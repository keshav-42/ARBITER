# CONTEXT — session handoff

**Read this first if you are picking up ARBITER in a new session.**
Pair it with [LOGIC.md](LOGIC.md), which explains *why* the system is built the way it is.

Last updated: Stage 4 complete.

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
| `35caedf` | docs — CONTEXT.md and LOGIC.md |
| (Stage 4) | NLI evidence verifier + reason-code classifier |

**172 tests passing.** Run: `python -m pytest tests/ -q`

All tests are offline — the NLI layer uses a deterministic stub backend, so nothing
downloads weights during a test run.

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
  nlp/
    hypotheses.py     what each exhibit is offered to prove, per (code, type)
    verifier.py       NLI backends: offline stub + DeBERTa MNLI
    classifier.py     reason-code classification from free text
tests/                172 tests, all offline
docs/                 initial proposal, Amex design system spec
```

Nothing exists yet under `backend/`, `frontend/`, `arbiter/calibration/`,
`arbiter/settlement/`, or `arbiter/eval/`.

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

Stage 4 NLP layer:

```
reason-code classifier   93.6% top-1, 97.1% top-3   (keyword/weak supervision)
ledger with stub NLI     87.6%   (baseline 87.3%)
ledger with real MNLI    87.2%   at parity, +23 self-defeating exhibits caught

on misleading-record scenarios only, real model vs baseline:  82.2% -> 92.8%
    C08.lost_in_transit   64.4% -> 100.0%
    C08.wrong_address     89.2% ->  95.0%
```

Corpus distribution: 51% card-member-right, 34% merchant-right, 15% ambiguous. C08 is
the largest code, matching real chargeback volume.

---

## Where to pick up: Stage 5

**Conformal calibration and abstention routing.** This replaces the hand-picked
`CONTESTED_BAND = 0.85` with a split-conformal threshold carrying a distribution-free
coverage guarantee.

The groundwork is done: a band sweep (see LOGIC.md §9) shows the posterior already
ranks uncertainty correctly, so Stage 5 only has to pick the operating point on that
curve. Do **not** hand-tune the band against the corpus.

The claim this unlocks: *"we auto-resolve X% of disputes at a guaranteed 1−α coverage,
and escalate the rest by design"* — far stronger than a bare accuracy number.

**Model download note.** The environment sets `HF_HUB_ENABLE_HF_TRANSFER=1` but
`hf_transfer` is not installed, so downloads fail. Set `$env:HF_HUB_ENABLE_HF_TRANSFER="0"`
first. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` is already cached locally.

### Remaining stages

| # | Stage | Notes |
|---|---|---|
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
