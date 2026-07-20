# CONTEXT — session handoff

**Read this first if you are picking up ARBITER in a new session.**
Pair it with [LOGIC.md](LOGIC.md), which explains *why* the system is built the way it is.

Last updated: Stage 3 complete (`7cf8876`).

---

## What this project is

A hackathon submission for the American Express **Frictionless Dispute & Chargeback
Resolution** problem statement. The brief asks for a system that auto-gathers evidence,
fairly weighs Card Member against Merchant, and resolves contested charges transparently
— cutting the cycle from weeks to minutes. It is explicitly **not** fraud detection: the
charge is already disputed, and the question is who should prevail.

The project name: **A**djudication via **R**eason-code **B**urden, **I**terative **T**rust
& **E**vidence **R**econciliation.

The differentiating idea, in one line:

> Every other team builds a dispute classifier. ARBITER is a court — it has a statute, a
> burden of proof that shifts by reason code, an evidence ledger where every exhibit
> carries a signed auditable weight, a judge that knows when it is not confident enough
> to rule, and a settlement path for when the evidence is genuinely a coin flip.

---

## Environment (verified, Windows)

| Thing | Status |
|---|---|
| Python | `3.10.19` at `C:\Users\MVI\anaconda3\envs\tensorgpu\python.exe` |
| Torch | `2.10.0+cu130` — CUDA available |
| Transformers | `4.57.1` |
| Also installed | fastapi, uvicorn, sqlalchemy, pydantic v2, scikit-learn, pandas, datasets |
| Node | `v24.18.0`, npm `11.16.0` — **NOT on PATH** |
| Docker | present |

**Node gotcha.** `node` is not on the PATH inherited by tool sessions. It lives at
`C:\Program Files\nodejs`. Prefix commands:

```powershell
$env:PATH = "C:\Program Files\nodejs;$env:PATH"
```

**Shell gotchas.**
- PowerShell 5.1: no `&&`, no `??`, no ternary. Use `;` and `if ($?) { }`.
- Here-strings break on apostrophes in commit messages. Write the message to a file in
  the scratchpad and use `git commit -F <file>`.
- Running a script from the scratchpad needs `$env:PYTHONPATH="c:\Users\MVI\Desktop\amex"`.

---

## Git

- Remote: `https://github.com/keshav-42/ARBITER.git` (private)
- Repo-local identity is set to `keshav-42 / keshavtrivedit24@gmail.com`, so commits use
  it automatically. The global identity is different (`shivapreetham`) — do not "fix" it.
- **Nothing has been pushed yet.** Git cannot authenticate to GitHub non-interactively:
  no cached token, no `gh` CLI. The user pushes manually with `git push -u origin main`.
  Do not attempt to store a token.

---

## Commits so far

| Commit | Stage |
|---|---|
| `c9a227f` | Stage 0 — scaffold, README with the staged plan and full mathematics |
| `69188df` | Stage 1 — domain core: reason codes, evidence taxonomy, statute layer |
| `388b6c5` | chore — ignore `.claude/` |
| `4d0d496` | Stage 2 — Bayesian Evidence Ledger, reputation, verdict narration |
| `7cf8876` | Stage 3 — synthetic corpus generator, and five engine bugs it exposed |

**121 tests passing.** Run: `python -m pytest tests/ -q`

---

## Layout

```
arbiter/
  core/
    reason_codes.py   17 AMEX codes; burden of proof as a numeric prior
    evidence.py       40 exhibit types; authenticity ceilings
    statute.py        13 deterministic rules; DISPOSITIVE / STRONG / ADVISORY
    ledger.py         THE ENGINE — posterior log-odds, correlation damping, waterfall
    reputation.py     Beta posteriors, capped and decaying
    explain.py        VerdictCard — template-constrained, never generative
  data/
    scenarios.py      36 causal scenarios with ground truth
    generator.py      world facts -> noisy evidence bundles
    narratives.py     realistic Card Member / Merchant text
    build_corpus.py   CLI: build + score against ground truth
tests/                121 tests
docs/                 initial proposal, Amex design system spec
```

Nothing exists yet under `backend/`, `frontend/`, `arbiter/nlp/`,
`arbiter/calibration/`, `arbiter/settlement/`, or `arbiter/eval/`.

---

## Current measured state

```
python -m arbiter.data.build_corpus --n 4000 --eval-only
```

```
agreement on decided cases   87.2%
by difficulty     easy 92.0%   medium 72.2%   hard 88.5%
worst reason code C32 at 71%          (was 19% before Stage 3 fixes)
ambiguous abstention  ~31%            (deliberately untuned — see below)
```

Corpus distribution: 51% card-member-right, 34% merchant-right, 15% ambiguous. C08 is
the largest code, matching real chargeback volume.

---

## Where to pick up: Stage 4

**The NLI evidence verifier.** This is the keystone model decision and the plumbing is
already in place.

`arbiter/core/ledger.py::compute_lambda` resolves the likelihood ratio in this order:

1. `metadata["lambda_lr"]` — explicit
2. `metadata["entail"]` / `metadata["contradict"]` — **the NLI path, already wired**
3. exhibit-type polarity — weak fallback

So Stage 4 only has to populate `entail` / `contradict` and the ledger consumes it
unchanged.

**The idea.** Frame every evidence check as natural-language inference:

```
premise    = the merchant's carrier record
hypothesis = "the goods were delivered to the cardholder's address"
        ->  P(entail), P(neutral), P(contradict)
        ->  lambda_i = log( P(entail) / P(contradict) )
```

Entailment supports whoever filed the exhibit, so the sign flips for merchant-filed
evidence. `compute_lambda` already does this.

**Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli` (zero-shot on day one).
Reason-code classification: DeBERTa-v3-base or `nlpaueb/legal-bert-base-uncased`.

**Build notes.**
- Keep it behind an interface with a deterministic stub, so tests never download weights
  and the demo runs offline. The existing 121 tests must not become network-dependent.
- Hypotheses should be templated per `(reason_code, evidence_type)`.
- Batch on GPU; cache by content hash.

### Remaining stages

| # | Stage | Notes |
|---|---|---|
| 4 | NLI verifier + reason-code classifier | in progress |
| 5 | Conformal calibration + abstention | replaces `CONTESTED_BAND` heuristic |
| 6 | Nash settlement + counterfactual recourse | |
| 7 | FastAPI + SQLAlchemy + append-only event log | |
| 8 | React UI, Amex design system, light **and** dark | user explicitly asked for both |
| 9 | Fairness audit, ECE, coverage plots, latency | |

---

## Open decisions

**Ambiguous abstention (~31%) is deliberately untuned.** A band sweep showed separation
between ambiguous and clear-cut abstention is positive and grows monotonically (+12.1% at
band 0.85 up to +20.3% at 4.0). The posterior already ranks uncertainty correctly, so
Stage 5 picks the operating point via conformal calibration. **Do not hand-tune
`CONTESTED_BAND` against the test set** — that is precisely the overfitting the conformal
layer exists to prevent.

**Vision (SigLIP) is the designated cut.** `VISUAL_SIMILARITY` already flows through the
ledger as a `lambda_lr`, so a real model can drop in later. Only ~8% of disputes turn on
it. If time runs short, cut vision before cutting anything else — the Evidence Ledger,
conformal abstention, and the waterfall UI are what win.

---

## Working agreements

- The user wants **staged commits with real substance**, not a single dump.
- Report findings honestly, including regressions. The Stage 3 commit documents five bugs
  found in my own earlier code; that is the expected standard.
- Verify by running, not by asserting. Every stage so far ended with a measurement.
- The UI must follow `docs/amex-design-system.md` and support light **and** dark themes.
