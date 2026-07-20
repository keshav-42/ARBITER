# LOGIC — why ARBITER is built this way

Companion to [CONTEXT.md](CONTEXT.md), which covers *where the work stands*. This file
covers *why the design is what it is*, so a new session can extend the system without
re-deriving the reasoning or accidentally undoing a deliberate choice.

---

## 1. The reframe that drives everything

The problem statement asks for a system that "fairly weighs the card member vs. merchant
perspective" and "explains the basis for each dispute decision."

The obvious build is a classifier: features in, winner out, SHAP plot bolted on. That
fails the brief in three specific ways.

1. **It cannot state a burden of proof.** Chargeback law is not symmetric. Under C08 the
   merchant must prove delivery; under C31 the card member must prove misdescription. A
   classifier learns this only as a correlation, and cannot cite it.
2. **Its explanation is post-hoc.** SHAP explains *the model*, not *the decision*. If the
   model is wrong, the explanation is a confident account of a wrong answer.
3. **It always answers.** A dispute where a courier marked "delivered" and the parcel was
   stolen from a porch is genuinely undecidable. A classifier outputs 0.51 and moves on.

So ARBITER is modelled as a **court** rather than a classifier:

| Court concept | Implementation |
|---|---|
| Statute | `core/statute.py` — deterministic rules with guide citations |
| Burden of proof | `ReasonCodeSpec.prior_logodds` — the burden *is* the prior |
| Evidence weighed on the record | `core/ledger.py` — additive log-odds |
| Reasoned judgment | `core/explain.py` — assembled from the arithmetic, never generated |
| Declining to rule | `Verdict.CONTESTED` → settlement or human review |
| Settlement | Stage 6 — Nash bargaining |

Everything below follows from that framing.

---

## 2. Why log-odds, and why additive

The posterior that the Card Member prevails:

```
Λ = Λ₀(code) + Σᵢ wᵢ · λᵢ · qᵢ · dᵢ + reputation
```

Log-odds space is chosen for one reason above all others: **it makes the explanation and
the computation the same object.**

In probability space, "this exhibit moved the case by 0.2" is meaningless without knowing
where you started — 0.5→0.7 and 0.9→1.1 are not comparable. In log-odds, every exhibit
contributes a signed, additive, position-independent quantity. So:

- the waterfall chart is a *direct rendering* of the arithmetic, not a visualisation of it
- `Adjudication.audit_sum()` re-derives the posterior from its parts, and the test suite
  asserts the identity. A verdict can be checked by hand.
- counterfactual recourse (Stage 6) reduces to arithmetic: to flip the verdict, find
  exhibits summing to `−Λ`.

Contributions are reported in **decibans** (`10·log₁₀`) rather than nats, because a
base-10 scale reads better on a chart and is the conventional unit for weight of evidence.

### The four factors

Each exhibit's contribution is a product of four terms, and each exists to stop a specific
failure:

| Term | Guards against |
|---|---|
| `λᵢ` likelihood ratio | — the actual signal |
| `qᵢ` authenticity | winning by uploading unverifiable material |
| `wᵢ` reason-code relevance | a delivery receipt deciding a duplicate-charge case |
| `dᵢ` correlation damping | filing the same fact five times to manufacture certainty |

---

## 3. The burden of proof is a prior, not a feature

This is the single most defensible idea in the project, and the one most likely to be
accidentally removed by someone "simplifying" the code.

The AMEX Chargeback Code Guide already allocates the burden. ARBITER encodes that
allocation as `prior_cm_win`, converted to log-odds:

```
C08  goods not received   P(CM) = 0.72  →  Λ₀ = +0.94   merchant must prove delivery
C31  not as described     P(CM) = 0.44  →  Λ₀ = −0.24   card member must prove discrepancy
R13  no reply             P(CM) = 0.88  →  Λ₀ = +1.99   procedural default
```

Consequences worth understanding before changing these numbers:

- **An empty record is not neutral.** It resolves to whoever does *not* bear the burden.
  That is legally correct and is why C08 with no merchant evidence is a card-member win.
- **The priors are deliberately moderate** (|Λ₀| < 2). A prior should tilt the starting
  point, never decide the case. Strong claims are made by the statute layer, which can
  clamp, not by inflating a prior.
- `test_prior_logodds_sign_follows_burden` enforces the sign convention against the
  declared `BurdenOfProof`. If you add a reason code, that test will tell you if the prior
  contradicts the burden.

### The bug this design initially had

Stage 1 implemented only half of it. The prior said "the merchant must prove delivery",
but nothing penalised a *card member* who failed to prove what their own claim required.
When the disputed event never happened — no return was ever shipped — no exhibit existed,
nothing pushed toward the merchant, and the prior sat unopposed. Every such scenario
scored 0%.

Stage 3 added `BURDEN.CM_UNMET`: **failing to substantiate your own claim is itself
evidence.** Then the mirror rule (`BURDEN.MERCHANT_UNMET`) had to be guarded, because a
merchant cannot document an event that never occurred — where no refund was ever agreed,
there is no credit note to produce. Hence `_cm_claim_substantiated`: the card member must
offer more than a narrative before merchant silence counts against them.

The general principle: **absence of evidence is informative exactly when the party who
bears the burden is the one who is silent.**

---

## 4. Why a symbolic layer sits above the probabilistic one

Some facts are not evidence to be weighed. A return shipped on day 16 against a 14-day
policy does not make the merchant *more likely* to be right — it settles the matter.

`core/statute.py` holds these as rules with three force levels:

- **DISPOSITIVE** — clamps the posterior to ±12 log-odds, bypassing the evidence sum
- **STRONG** — a large fixed shift (±2.5), still rebuttable
- **ADVISORY** — recorded in the narration, moves nothing

This is a **neuro-symbolic** architecture: learned likelihoods underneath, deterministic
law on top. It matters because:

- date arithmetic must never be delegated to a neural model
- dispositive outcomes get a guide citation, not a probability
- rules are individually unit-testable, which learned weights are not

Conflicting dispositive rules resolve by magnitude, ties toward the merchant, and **both
findings are retained** so the conflict is visible in the audit trail rather than silently
resolved.

Clamps are ±12 rather than ±∞ so downstream sigmoid and settlement arithmetic stays
numerically well behaved.

### The `P07` lesson

Two rules both meant "late", in opposite directions:

- `filed_outside_dispute_window` — the *card member* waited too long to complain
- `P07.LATE_SUBMISSION` — the *merchant* settled the charge too long after authorisation

Stage 3 initially fed both from one field, and P07 scored 19% — worse than chance. They
now read separate fields (`days_since_transaction` vs `submission_delay_days`). **When
adding a temporal rule, be explicit about whose clock it measures.**

---

## 5. Calibration: three mechanisms, one goal

An end-to-end run in Stage 2 produced **98% confidence from two exhibits**. That is the
naive-Bayes independence assumption failing exactly as it always does. Three controls now
keep confidence honest:

**Correlation damping (`CORRELATION_DECAY = 0.55`).** A delivery confirmation and a
signature proof are two views of one carrier record. Within a `(correlation group, party)`
bucket, the strongest exhibit keeps full weight and the rest decay geometrically. Network
exhibits are exempt — they are independent measurements, not a party's own account.

**Per-exhibit clamp (`MAX_EXHIBIT_CONTRIBUTION = 3.0`).** One overconfident model output
cannot dominate a case that should turn on the whole record.

**Aggregate ceiling (`MAX_EVIDENCE_LOGODDS = 5.3`).** A long one-sided record cannot claim
more certainty than documentary evidence supports. Roughly P = 0.995.

Net effect: the demo case moved 98% → 96%, with the corroborating exhibit correctly
discounted from 9.5 to 5.2 decibans. This mattered to fix *before* Stage 5, because
conformal calibration would otherwise have been calibrating against inflated inputs.

---

## 6. Reputation is capped on purpose

A merchant with 4,000 clean deliveries deserves more credence than one who loses 40% of
disputes. But reputation is the most dangerous signal in the system — unchecked, it
becomes a loop where small merchants lose *because* they are small.

Three constraints, all deliberate:

- **Cap at ±0.40 log-odds**, and the *sum* over both parties is capped too, so two
  mediocre records cannot compound into a decisive prior.
- **Shrinkage below 10 observations**, so a new merchant is not judged on three cases.
- **Geometric decay (0.98/month) toward the prior**, so a merchant who fixes their
  operation recovers and one coasting on an old record does not.

`test_reputation_cannot_flip_a_decided_case` locks the guarantee: **reputation nudges,
evidence decides.** Stage 9 will re-run adjudications with `apply_reputation=False` to
measure how much any verdict depended on it.

---

## 7. Explanation is assembled, never generated

`core/explain.py` builds the `VerdictCard` from `LedgerEntry` and `RuleFinding` objects.
There is no language model in the path.

The failure mode being avoided: an LLM reads a decision and writes a plausible
justification. That is **rationalisation**, and it is worse than no explanation because it
is convincing — it will confidently explain a wrong verdict.

Here, every sentence traces to a number that actually entered the posterior. Reasons are
ranked by true magnitude, not by narrative appeal. An LLM may later *restyle* this text,
but it must not introduce a reason absent from the ledger.

---

## 8. Why the corpus generates the world before the evidence

The pipeline is deliberately causal:

```
scenario → world facts → exhibits → narratives → DisputeCase
              ↓
        ground truth label
```

The usual synthetic-data mistake is to write a narrative and then label it, which teaches
a model to read surface cues. Here the label is **causally upstream** of the text, so a
model that predicts it must learn the evidence→outcome relationship.

Ground truth is **what happened in the world**, deliberately *not* the ledger's output.
The engine only ever sees exhibits, so a gap between them is expected and measurable —
that gap is what Stage 5 calibrates and Stage 9 audits.

Three noise processes model why real dispute files are hard:

- **withholding** — a party never files what would have helped them
- **corruption** — the exhibit exists but will not verify
- **misleading** — authentic and pointing the wrong way (the porch-theft case)

Without these, ambiguous scenarios would not be ambiguous, and there would be nothing for
the abstention layer to learn.

### Why not CFPB

The original proposal suggested fine-tuning on the CFPB Consumer Complaint Database. That
is the wrong corpus: CFPB narratives are grievances **about banks** ("Amex denied my
dispute"), not **about merchants** ("the sofa never arrived"), and they carry no
ground-truth verdict. ARBITER needs `(evidence bundle → who should win)`, which does not
exist publicly and must be constructed. CFPB remains useful for *domain-adaptive
pretraining* — learning how consumers phrase financial grievances — but not for labels.

---

## 9. Abstention is a feature, and its threshold is not to be hand-tuned

`CONTESTED_BAND = 0.85` is a **placeholder**. Stage 5 replaces it with a split-conformal
threshold carrying a distribution-free coverage guarantee.

A band sweep over the Stage 3 corpus:

| band | ambiguous abstain | clear abstain | separation |
|---|---|---|---|
| 0.85 | 30.7% | 18.6% | +12.1% |
| 2.00 | 54.2% | 38.6% | +15.6% |
| 4.00 | 83.7% | 63.4% | +20.3% |

Separation is positive and grows monotonically — **the posterior already ranks uncertainty
correctly.** Stage 5 only has to pick the operating point.

> **Do not tune `CONTESTED_BAND` against the evaluation corpus.** That is exactly the
> overfitting conformal prediction exists to prevent. The honest claim is *"we auto-resolve
> X% at a guaranteed 1−α coverage"*, which is far stronger than a bare accuracy number and
> is what a bank actually needs.

---

## 10. Design decisions a future session should not silently reverse

| Decision | Why it is that way |
|---|---|
| Positive log-odds favour the Card Member | Fixed sign convention repo-wide; flipping it inverts every prior, rule, and test |
| Priors kept moderate (\|Λ₀\| < 2) | The statute decides hard cases, not inflated priors |
| Dispositive clamps are ±12, not ±∞ | Keeps sigmoid and settlement arithmetic finite |
| Unverified evidence capped at 0.65 authenticity | Stops winning by uploading more unverifiable material |
| A narrative alone never discharges a burden | Anyone can assert anything |
| Reputation capped and decaying | Prevents a self-reinforcing loop against small merchants |
| Explanation assembled, not generated | Avoids confident rationalisation of wrong verdicts |
| Ground truth ≠ ledger output | The gap is the thing being measured |
| Vision (SigLIP) is the designated cut | ~8% of disputes turn on it; the ledger and waterfall are what win |

---

## 11. How to extend the system

**Adding a reason code** — add a `ReasonCodeSpec` with `prior_cm_win` matching its
`BurdenOfProof` (a test enforces this), set `evidence_weights` for exhibits that matter,
and add scenarios in `data/scenarios.py` covering both outcomes plus an ambiguous case.

**Adding a statute rule** — decorate with `@rule`, return `None` when it does not apply,
always supply `rationale` and `guide_reference`. Prefer `STRONG` over `DISPOSITIVE` unless
the fact is genuinely conclusive. Be explicit about whose clock any temporal rule measures.

**Adding an evidence type** — add to `EvidenceType`, set its default polarity, decide
whether it belongs in `VERIFIABLE_TYPES` (machine-checkable against an external source)
and in a correlation group.

**Plugging in a model** — populate `metadata["entail"]` / `metadata["contradict"]` and
`compute_lambda` consumes it unchanged. Keep it behind an interface with a deterministic
stub so the 121 existing tests never depend on the network.

Always finish by running the measurement, not by asserting the change worked:

```powershell
python -m pytest tests/ -q
python -m arbiter.data.build_corpus --n 4000 --eval-only
```
