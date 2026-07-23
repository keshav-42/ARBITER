# CONTEXT — session handoff

**Read this first if you are picking up ARBITER in a new session.**
Pair it with [LOGIC.md](LOGIC.md), which explains *why* the system is built the way it is.

Last updated: Stage 9 complete — **all 9 stages done.**

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
| `ba98bdf` | Stage 4 — NLI evidence verifier + reason-code classifier |
| `4c0f92e` | Stage 5 — conformal abstention, temperature scaling, routing |
| `f08a8c8` | Stage 6 — Nash settlement + counterfactual recourse |
| `6ca5512` | Stage 7 — FastAPI backend, SQLAlchemy schema, append-only event log |
| `01c14d2` | Stage 8 — React UI, Amex design system, light + dark themes |
| (Stage 9) | fairness audit, calibration metrics, eval harness |

**284 tests passing** (Python). Run: `python -m pytest tests/ -q`
Frontend builds clean: `cd frontend && npm run build`
Eval report: `python -m arbiter.eval.run --n 5000 --out docs/eval-report`

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
  calibration/
    temperature.py    temperature scaling + ECE/MCE/reliability
    conformal.py      split-conformal abstention, routing, persistence
    fit_calibrator.py CLI: fit + validate + save the calibrator
  settlement/
    nash.py           Nash bargaining split of the disputed amount
    recourse.py       counterfactual "what would flip this verdict"
backend/
  db.py               SQLAlchemy schema + append-only event log guard
  service.py          the pipeline wired to persistence (state machine)
  schemas.py          Pydantic request/response models
  main.py             FastAPI app + WebSocket status hub
models/               calibrator.json (fitted artifact, committed)
tests/                262 tests, all offline
docs/                 initial proposal, Amex design system spec
```

frontend/
  src/theme.css       Amex tokens, both themes (dark is designed, not inverted)
  src/components/      Waterfall (hero), VerdictCard, StatusTracker, IntakeForm
  src/lib/             api client, theme hook, demo fixture
  src/App.tsx          shell: intake -> verdict + live tracker
models/               calibrator.json (fitted artifact, committed)
tests/                262 tests, all offline
docs/                 initial proposal, Amex design system spec
```

Run the API: `uvicorn backend.main:app --port 8000`
Run the UI: `cd frontend && npm run dev` (Node at C:\Program Files\nodejs, not on PATH)
Offline demo: `http://localhost:5173/?demo=1&theme=light`
Nothing exists yet under `arbiter/eval/`.

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

Stage 5 conformal (validated on a held-out split, `models/calibrator.json`):

```
COVERAGE GUARANTEE HOLDS AT EVERY ALPHA
    alpha=0.20  target 80%  empirical 81.6%   auto-resolve 97.9%
    alpha=0.10  target 90%  empirical 90.6%   auto-resolve 75.3%  acc 87.6%
    alpha=0.05  target 95%  empirical 95.1%   auto-resolve 44.4%
temperature T=1.78 (ledger was overconfident)   ECE 0.066
end-to-end routing: auto_resolve 64% | settlement 20% | statute 12% | human 3%
```

Stage 6 settlement + recourse:

```
Nash settlement offers: 100% mutually beneficial (both beat their fight outcome)
  at Lambda=0 the split is ~52.5% CM / 47.5% M (tilted to the higher-cost merchant)
  surplus each = the avoided cost of fighting, split fairly
counterfactual recourse: 271/271 'sufficient' options actually flip the verdict
  the arithmetic is exact, so recourse is a promise, not a model guess
```

Corpus distribution: 51% card-member-right, 34% merchant-right, 15% ambiguous. C08 is
the largest code, matching real chargeback volume.

---

## All nine stages complete

The system is end-to-end: intake → classify → verify → adjudicate → route →
settle/recourse → persist → stream → UI, with a full evaluation harness.

### If continuing, likely next steps (not yet built)

- **Fine-tune the models.** The NLI verifier and classifier run on their zero-shot /
  keyword backends. The three-stage curriculum (DAPT → weak supervision → LoRA) is
  described in README; the corpus and hooks exist. Would lift the misleading-record
  cases further.
- **Real doc parsing (LayoutLMv3/Donut) and vision (SigLIP).** `VISUAL_SIMILARITY`
  already flows through the ledger as a lambda; a real model drops in. Vision is the
  designated cut — only ~8% of disputes turn on it.
- **Merchant-side deflection console** (pre-dispute interception) and the WebSocket
  live-push wired into the UI tracker (currently polls on adjudicate).
- **Presentation deck + video.** `docs/eval-report.html` is the metrics slide;
  `?demo=1` drives the walkthrough.

Node is at `C:\Program Files\nodejs` (v24), NOT on the inherited PATH — prefix
`$env:PATH = "C:\Program Files\nodejs;$env:PATH"`. HF downloads need
`$env:HF_HUB_ENABLE_HF_TRANSFER="0"`.

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
