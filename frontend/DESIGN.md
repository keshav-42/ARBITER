# DESIGN.md — ARBITER frontend

Durable visual decisions for the ARBITER interface. Product truth lives in
`/PRODUCT.md`; this file owns the look. Written for the Impeccable design skill.

## The world: "The Determination"

The interface **is an official American Express adjudication notice**, not a fintech
dashboard. A dispute is a *determination* — a statement of decision — so the surface
reads like one: a seal, a docket header, a ruling you can read top to bottom. This
refuses the category default (a grid of same-size cards with icons and border-accents).

## Non-negotiables

- **American Express is a pinned brand commitment**, not a free choice. Amex Blue
  (#006FCF) is the single action color. Deep navy (#00175A → #000C3D) is the document
  ground on arrival; warm paper (#fbfaf7 light / #071634 dark) is the ruling surface.
  Benton Sans stack. The posture is Amex-grade: calm, premium, precise with money.
- **Both themes are designed, not inverted.** Dark uses navy as canvas (white-on-navy
  passes AAA); the chart marks are validated separately for the dark lightness band.

## Structure, not decoration

- **Hairline rules and tabular figures do the work** that cards and colored
  border-accents used to. A `1px` navy-tint rule (`--rule`) separates; proximity groups.
  No card-left accent bars. No icon-tile grid. No hero-metric template.
- **Single reading column**, max 760px. The determination is read, not scanned across
  panels.
- **Spacing scale** `--s1..--s9` (4px base), reused semantically. More space above a
  heading than below it.

## The signature elements

- **The scale-of-justice meter** replaced the log-odds waterfall as the headline "how it
  decided" visual. A single track: Merchant ← balanced → You, with a needle and one
  plain confidence phrase. No decibans on this surface. The waterfall survives only
  inside "See the detailed reasoning" for reviewers.
- **Plain-language reasons** — 3–4 sentences, each with a small side-dot. The customer
  never sees "decibans" or "log-odds"; that vocabulary lives under progressive
  disclosure.
- **The pipeline** animates the actual work (reading documents, verifying, weighing) with
  per-stage timings, framing the honest claim: weeks became minutes because the delay
  was human hand-offs, not compute.
- **Record of proceedings** — the event log rendered as an official docket with an
  integrity attestation, not a progress ring.

## Motion

One authored moment: surfaces **settle in** (opacity + slight rise + brief blur clearing,
exponential ease-out). The needle animates to position. The active pipeline stage
pulses. Everything reduces to nothing under `prefers-reduced-motion`.

## Refused (category defaults, per the craft floor)

Same-size card grids; colored `border-left` accents; gradient text; glass/blur as
decoration; monospace-as-costume; the big-number hero-metric template; light/dark picked
by category rather than use scene. Theme is chosen from the scene: a person resolving a
money dispute, often anxious, on a phone or laptop — light by default, dark offered.

## Accessibility

AA minimum; financial figures are real tabular text, never images; focus ring always
visible (2px, blue on light, white on navy); status conveyed by text + shape, not color
alone; respects reduced motion.
