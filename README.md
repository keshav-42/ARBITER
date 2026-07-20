# ARBITER

**A**djudication via **R**eason-code **B**urden, **I**terative **T**rust & **E**vidence **R**econciliation

> A closed-loop dispute arbitration engine for American Express that resolves contested
> charges in ~90 seconds instead of ~45 days — and shows its work.

---

## The thesis

Most dispute systems are **classifiers**: evidence goes in, a winner comes out, and nobody
can explain why. ARBITER is a **court**. It has a statute (the AMEX Chargeback Code Guide),
a burden of proof that shifts by reason code, an evidence ledger where every exhibit carries
a signed, auditable weight, a judge that knows when it is not confident enough to rule, and
a settlement path for when the evidence is genuinely a coin flip.

The one-line pitch:

> Every other team is building a dispute classifier. We built a court of law that runs in
> 90 seconds, cites its statute, states its burden of proof, quantifies its own uncertainty,
> and settles rather than fights when settling is cheaper for everyone.

### Why this is not a fraud detector

The problem statement is explicit: this resolves charges **already disputed**. There is no
fraud score here. The question is never "is this transaction fraudulent" — it is "given two
parties who disagree, and a rulebook, who is right, how sure are we, and what is the fairest
disposition."

---

## Core mathematics

### 1. The Bayesian Evidence Ledger

A dispute is a hypothesis test between `H_CM` (card member is right) and `H_M` (merchant is
right). The posterior log-odds:

```
Λ = log[ P(H_CM) / P(H_M) ]  +  Σ  w_i · λ_i · q_i
    └──────── burden of proof ────┘   └── evidence i ──┘
```

| term  | meaning |
|-------|---------|
| `Λ₀`  | prior log-odds — **the burden of proof, set per reason code by the AMEX guide** |
| `λ_i` | log-likelihood ratio of exhibit `i`: `log[ P(e_i\|H_CM) / P(e_i\|H_M) ]` |
| `q_i` | authenticity / quality of exhibit `i` in `[0,1]` — is the tracking number verifiable, is the image tampered |
| `w_i` | reason-code-specific weight for that evidence type |

Because the model is **additive in log-space**, every exhibit has an exact, signed,
comparable contribution measured in **decibans**. The explanation is not generated after
the fact — it *is* the arithmetic. This is what makes the transparency layer real.

The prior is the detail that matters most: the AMEX Chargeback Code Guide already encodes
who must prove what. Under **C08 (goods not received)** the merchant must produce delivery
confirmation, so the prior favors the card member. Under **C31 (not as described)** the card
member must demonstrate the discrepancy, so the prior favors the merchant. ARBITER encodes
regulation directly as a Bayesian prior.

### 2. The Statute Layer (neuro-symbolic)

Some facts are not probabilistic. A return shipped on day 16 under a 14-day policy is a
**hard constraint**, not a piece of evidence. Dispositive rules clamp the posterior:

```
Λ_final =  −∞   if a merchant-dispositive rule fires
           +∞   if a card-member-dispositive rule fires
            Λ   otherwise
```

Symbolic law on top, learned likelihoods underneath.

### 3. Conformal prediction — knowing when to abstain

Split conformal calibration over nonconformity scores `s_j = 1 − p̂_j(y_j)`; take `q̂` as the
`⌈(n+1)(1−α)⌉`-th smallest. At inference emit the prediction set:

```
C(x) = { y : p̂(y|x) ≥ 1 − q̂ }
```

- `|C(x)| = 1` → **auto-resolve** (this is the "weeks to minutes")
- `|C(x)| = 2` → **abstain** → route to settlement or a human adjudicator

This yields a distribution-free, finite-sample guarantee `P(y ∈ C(x)) ≥ 1 − α`. The claim
becomes *"we auto-resolve 73% of disputes at a guaranteed 95% coverage, and escalate the
rest by design"* — which is what a bank actually needs, unlike a bare accuracy number.

### 4. Nash bargaining — settle instead of fight

When `Λ ≈ 0` a binary verdict just makes someone angry. Split the disputed amount `V` by
maximizing the Nash product over the card member's share `x`, with the disagreement point
set to the expected outcome of a full chargeback cycle net of its cost:

```
max ( u_CM(x) − d_CM ) · ( u_M(V−x) − d_M )
```

Under linear utilities this centers on the posterior, adjusted by each side's cost of
continuing:

```
x* = V · σ(Λ) + ½(c_M − c_CM)        σ(z) = 1/(1+e^(−z))
```

Both parties accept because both beat their expected outcome from fighting.

### 5. Reputation as a decaying Beta prior

```
θ_M ~ Beta(α_M + wins, β_M + losses)          α ← γ·α,  γ ≈ 0.98 / month
```

`E[θ_M]` enters as a **capped** prior adjustment so reputation can never override hard
evidence — that cap is a fairness constraint, and the decay lets merchants rehabilitate.

---

## Model stack

| Job | Model | Why |
|---|---|---|
| Reason-code classification | `microsoft/deberta-v3-base`, `nlpaueb/legal-bert-base-uncased` | Legal-BERT is pretrained on regulatory/contract register — the exact language of merchant ToS and the chargeback guide |
| **Evidence ↔ claim reconciliation** | **`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`** | ⭐ The keystone. Frame every check as entailment: premise = merchant tracking record, hypothesis = "goods were delivered to the cardholder's address". The entail/contradict ratio **is** `λ_i`. Works zero-shot on day one, improves with fine-tuning. |
| Guide retrieval | `BAAI/bge-m3` + FAISS | Hybrid dense/sparse retrieval to cite the exact clause |
| Clause reranking | `BAAI/bge-reranker-v2-m3` | Cross-encoder precision — cite the *right* clause, not a plausible one |
| Document parsing | LayoutLMv3 / Donut | Receipts and shipping labels are 2-D layout documents; flat OCR loses structure |
| Temporal extraction | spaCy + deterministic normalizer | Date arithmetic is never delegated to a neural model |
| Long ToS / chat logs | ModernBERT (8k ctx) | Ingests full terms of service in one pass |
| Visual discrepancy *(bonus)* | SigLIP | Better-calibrated similarity than CLIP's softmax contrastive |

### Fine-tuning curriculum

CFPB narratives are complaints *about banks*, not *about merchants* — the label space does
not map cleanly onto AMEX reason codes. So the strategy is three-stage:

1. **DAPT** — masked-LM over the AMEX Chargeback Code Guide, merchant ToS corpora, and
   network reason-code documentation. Cheap; teaches vocabulary.
2. **Weak supervision** — Snorkel-style labeling functions plus an LLM-generated synthetic
   dispute corpus with **known ground-truth verdicts**, which is the one thing CFPB cannot
   provide.
3. **Fine-tune** with focal loss (reason codes are heavily imbalanced) + LoRA, then
   **calibrate** with temperature scaling before conformal.

---

## Stages

Each stage is a commit. Each stage leaves the repo in a runnable state.

| # | Stage | Deliverable | Status |
|---|-------|-------------|--------|
| 0 | **Scaffold** | Repo layout, this README, tooling config | ✅ |
| 1 | **Domain core** | Reason-code registry, burden-of-proof priors, statute rules, evidence taxonomy | ⬜ |
| 2 | **Synthetic corpus** | Generator producing labeled disputes across all reason codes with ground-truth verdicts | ⬜ |
| 3 | **Evidence Ledger** | The log-odds arbitration engine + statute clamping | ⬜ |
| 4 | **NLP layer** | Zero-shot NLI evidence verifier, reason-code classifier, guide retrieval | ⬜ |
| 5 | **Conformal** | Split-conformal calibration, abstention routing, temperature scaling | ⬜ |
| 6 | **Settlement** | Nash bargaining engine + counterfactual recourse ("what would flip this") | ⬜ |
| 7 | **Backend** | FastAPI + SQLAlchemy, append-only event log, WebSocket status stream | ⬜ |
| 8 | **UI** | React + Vite, Amex design system, light **and** dark themes, Evidence Ledger waterfall | ⬜ |
| 9 | **Evaluation** | Fairness audit, ECE calibration curves, coverage plots, latency benchmarks | ⬜ |

---

## Differentiators

Things that will not appear in other submissions:

1. **The Evidence Ledger waterfall** — a horizontal chart starting at the burden-of-proof
   prior, each exhibit pushing left (merchant) or right (card member) by its exact deciban
   contribution, ending at the verdict with the conformal interval as a shaded band. One
   screen that *is* the explanation.
2. **"Challenge the Verdict"** — counterfactual recourse. Either party sees which exhibit
   hurt them most and exactly what would flip the outcome: *"provide signed delivery
   confirmation and the verdict reverses."* Regulators require actionable recourse; nobody
   builds it.
3. **Evidence authenticity scoring** — EXIF/ELA checks, carrier-API tracking verification,
   timestamp-versus-claim consistency. Gives `q_i` a real basis and blocks the "upload any
   photo" attack.
4. **Calibrated abstention** — the system declines to rule when it should, with a coverage
   guarantee rather than a vibe.
5. **Fairness as a measured quantity** — asymmetry gap across merchant size, card-member
   tenure, and value bands; plus a counterfactual role-swap test (swap the parties' evidence
   and the verdict must flip — if it doesn't, the model has a side-bias).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React + Vite  ·  Amex design system  ·  light + dark    │
│  card-member portal │ merchant console │ ledger waterfall│
└───────────────────────────┬─────────────────────────────┘
                            │ REST + WebSocket
┌───────────────────────────┴─────────────────────────────┐
│  FastAPI                                                 │
│  intake → aggregation → arbitration → verdict → settle   │
└───────────────────────────┬─────────────────────────────┘
                            │
      ┌─────────────────────┼─────────────────────┐
      │                     │                     │
┌─────┴──────┐      ┌───────┴────────┐    ┌───────┴───────┐
│  Ledger    │      │  NLP service   │    │  Postgres /   │
│  engine    │      │  NLI · cls ·   │    │  SQLite       │
│  statute   │      │  retrieval     │    │  event log    │
└────────────┘      └────────────────┘    └───────────────┘
```

### Database spine

An **append-only event log** — because "transparent" means auditable.

- `disputes` — reason code, state, prior/posterior log-odds, verdict, conformal set
- `evidence_items` — party, type, blob URI, `q_authenticity`, `lambda_LR`, weight,
  **`contribution_decibans`** ← the XAI, stored per row and queryable
- `dispute_events` — immutable `(from_state, to_state, actor, payload, ts)` audit trail;
  also drives the real-time tracker over WebSocket
- `statute_rules` / `rule_firings` — which guide clause fired, with page citation
- `settlements` — `x*`, acceptance flags per party
- `reputation` — `alpha`, `beta`, `last_decay_at`

Evidence blobs are content-hashed (SHA-256) so tampering is detectable and the audit trail
is provable.

---

## Stack

- **Backend** — Python 3.10, FastAPI, SQLAlchemy 2.x, Pydantic v2
- **ML** — PyTorch 2.10 (CUDA), Transformers 4.57, scikit-learn
- **Frontend** — React 18 + Vite + TypeScript
- **Database** — SQLite for the demo, PostgreSQL-compatible schema

## Getting started

```bash
# backend
pip install -r requirements.txt
uvicorn backend.app.main:app --reload

# frontend
cd frontend && npm install && npm run dev
```

## Layout

```
arbiter/
├── core/          domain model, reason codes, statute rules, ledger engine
├── nlp/           NLI verifier, classifier, retrieval
├── calibration/   conformal, temperature scaling
├── settlement/    Nash bargaining, counterfactual recourse
├── data/          synthetic corpus generator
└── eval/          fairness audit, calibration metrics
backend/           FastAPI application
frontend/          React + Vite UI
docs/              design notes, presentation material
```

---

## License

Hackathon submission. Not affiliated with American Express.
