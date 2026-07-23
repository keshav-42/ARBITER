# PRODUCT.md — ARBITER

> Captured for the Impeccable design skill. Product truth, not visual decisions.

## What it is

ARBITER resolves **already-disputed** credit-card charges (chargebacks) between a Card
Member and a Merchant. It is not fraud detection — the charge is contested, and the
question is *who is right, how sure are we, and what is the fairest outcome*. Built for
the American Express "Frictionless Dispute & Chargeback Resolution" brief.

## The unique mechanism (one sentence)

It runs a dispute like a **court**: the AMEX Chargeback Code Guide is the statute, each
reason code sets a **burden of proof**, every piece of evidence carries a signed,
auditable weight, and the verdict is the sum — so the explanation *is* the arithmetic,
not a story told about it.

## Who uses it, and the real scene

- **Card Member (primary):** an ordinary person who feels wronged — a parcel never came,
  a refund never landed, an item arrived broken. Anxious, wants their money back, does
  **not** know or care what a "reason code" or a "deciban" is. On a phone or laptop,
  often frustrated, wants to know *what happens now and did I win*.
- **Merchant:** a business responding to a claim, more transactional.
- **Judges / reviewers (secondary):** want to see the rigor — the math, the calibration,
  the fairness proof. This audience wants depth **on demand**, never in the customer's face.

## What must be true (product facts, not inventable)

- Resolves in **minutes, not weeks** — and the honest reason is that the delay was
  *humans waiting on each other*, not compute. Reading a document (OCR/layout) and
  verifying it takes seconds-to-minutes; that is the pitch.
- ~60% of disputes resolve on **deterministic rule checks** (duplicate charge? return
  after the policy window? refund already posted?) — a lookup, not a model.
- Documents are **parsed** into structured evidence (tracking #, address, dates). In the
  demo this parse is simulated but framed truthfully and labelled.
- The engine states a **burden of proof**, quantifies its **own confidence** (calibrated,
  with a coverage guarantee), **abstains** when genuinely unsure, and proposes a **fair
  split** when the evidence is a coin flip.
- Every decision is **auditable**: an append-only event log, per-exhibit contributions.
- Measured: 88.5% agreement with ground truth, coverage guarantee holds, 100%
  identity-independent fairness, ~12,700 disputes/s.

## The job the interface must do

**Operate** (Card Member completing "get my dispute resolved") with a **Persuade** framing
on arrival (this is a hackathon submission that must land its thesis fast). Success:
a frustrated person files in plain words, watches the system *do visible work*, and gets
a clear answer they understand — with the option to disagree and to see the proof.

## Hard constraints / brand commitments

- **American Express is a fixed brand commitment.** Amex Blue (#006FCF), deep navy
  (#00175A), the restraint-and-trust posture, Benton Sans family. This is a real
  institution's identity, not a free aesthetic choice — it must read as Amex-grade:
  premium, calm, trustworthy, precise with money. (docs/amex-design-system.md)
- **Both light and dark themes** are required.
- Accessibility: AA minimum, financial figures as real text, visible focus, respects
  reduced-motion.

## What would make a polished result feel wrong

- Dumping technical jargon (decibans, log-odds) on the customer.
- An info-dump with no hierarchy — everything shouting at once.
- The old left/right "waterfall" bars as the *headline* — unintuitive to a normal user.
- Claiming "seconds" with no visible work — it reads as a magic trick, not a system.
- Looking like a generic fintech dashboard or a startup landing page. It must feel like
  Amex, and it must feel like a *decision being made*, not a form being submitted.
