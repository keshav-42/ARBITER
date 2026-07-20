---
version: alpha
name: American Express
description: "A premium financial identity anchored by Amex Blue (#006FCF) and deep navy (#00175A), where restrained blue authority, generous whitespace, and Benton Sans's measured forms communicate trust, prestige, and institutional permanence across membership experiences."

colors:
  primary: "#006FCF"
  on-primary: "#FFFFFF"
  primary-hover: "#1374D4"
  primary-pressed: "#00509E"
  navy: "#00175A"
  navy-deep: "#000C3D"
  ink: "#1A1A1A"
  ink-muted: "#53565A"
  ink-subdued: "#86888C"
  ink-on-navy: "#FFFFFF"
  ink-on-navy-muted: "#B7C3D9"
  canvas: "#FFFFFF"
  surface-1: "#F7F8F9"
  surface-2: "#ECEDEE"
  border: "#D5D9DC"
  border-subtle: "#ECEDEE"
  success: "#00875A"
  warning: "#B95000"
  error: "#C52720"
  gold: "#BF9B30"

typography:
  display:
    fontFamily: "Benton Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
    fontSize: 40px
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: -0.01em
  body:
    fontFamily: "Benton Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: 0em

spacing:
  base: 8px
  scale: [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

radius:
  sm: 4px
  md: 8px
  lg: 12px
  pill: 9999px

shadows:
  card: "0 1px 4px rgba(0,23,90,0.10)"
  elevated: "0 6px 24px rgba(0,23,90,0.16)"

motion:
  duration-fast: 120ms
  duration-base: 240ms
  easing: cubic-bezier(0.4, 0, 0.2, 1)
---

## Rationale

**Blue as institutional trust** — American Express Blue (#006FCF) is one of finance's most established equity colors, carried for decades on the iconic Blue Box logo. In the digital product it functions as the color of authority and action: primary buttons, links, and the brand mark. Amex does not chase trendy palettes — the blue's consistency over time is precisely the point. It signals permanence, security, and a relationship that predates and will outlast any design fashion.

**Prestige through restraint** — The Amex brand promise is membership and premium service, which the design expresses not through ornamentation but through discipline. Generous whitespace, measured typography, a tight color palette, and clean alignment communicate that this is a serious financial institution worthy of trust with high-value spending. Where consumer fintechs shout, Amex speaks calmly — the confidence to use less is itself a luxury signal.

**Deep navy for premium and dark surfaces** — Beyond the everyday blue, Amex reaches for a deep navy (#00175A) on premium card tiers, statement headers, and dark hero surfaces. The navy adds gravitas — it reads as the boardroom and the Platinum tier rather than the everyday checking account. White type on navy is a recurring premium pairing that elevates membership moments above routine transactional screens.

**Clarity for high-stakes financial decisions** — Cardmembers manage real money: balances, payments, rewards, disputes. Every screen must present financial information with absolute legibility and zero ambiguity. The system uses comfortable body sizes, clear numeric hierarchy with tabular figures, and unambiguous semantic colors for status. Trust is earned transaction by transaction, and a misread balance or unclear due date erodes it instantly — clarity is a feature of the brand, not just the UI.

## 1. Visual Theme & Atmosphere
American Express feels composed, premium, and institutional. The interface is predominantly clean white with Amex Blue as the consistent action and brand color, punctuated by deep navy on premium and hero surfaces. The atmosphere is calm authority — generous space, restrained color, and confident typography that conveys decades of financial trust rather than startup energy.

The membership experience foregrounds the card and the cardmember's standing: card art, rewards balance, and account summary are presented with prominence and polish. Premium tiers (Gold, Platinum, Centurion) shift toward darker navy and metallic-gold accents to signal elevated status, while everyday servicing screens stay bright, clear, and blue. The depth model is gentle — soft navy-tinted shadows lift cards just enough to organize without drama.

## 2. Color System
**Brand blue system**:
- Amex Blue: #006FCF — primary buttons, links, brand mark, key actions
- Hover: #1374D4 — slightly brighter on interaction
- Pressed: #00509E — confirms the press

**Premium navy**:
- Navy: #00175A — premium card surfaces, statement headers, dark heroes
- Navy deep: #000C3D — deepest premium surfaces, gradients

**Light surface system**:
- Canvas: #FFFFFF — primary content, account surfaces
- Surface 1: #F7F8F9 — page background, grouped sections
- Surface 2: #ECEDEE — nested panels, input fields
- Border: #D5D9DC — card edges, dividers
- Border subtle: #ECEDEE — internal separators

**Text on light**:
- Primary ink: #1A1A1A — near-black, primary reading color
- Muted: #53565A — secondary metadata, labels
- Subdued: #86888C — fine print, timestamps

**Text on navy**:
- On-navy: #FFFFFF — headlines and figures on navy surfaces
- On-navy muted: #B7C3D9 — secondary labels on navy

**Premium accent**:
- Gold: #BF9B30 — Gold/Platinum tier accents, metallic membership cues (used sparingly)

**Semantic**:
- Success: #00875A — payment posted, rewards earned
- Warning: #B95000 — payment due soon, action required
- Error: #C52720 — declined, overdue, security alert

Amex Blue is the action and identity color; navy is the premium surface. Gold is reserved strictly for tier signaling — overusing it would cheapen the prestige cue it exists to protect.

## 3. Typography
American Express uses Benton Sans — a refined, highly legible American grotesque — as its corporate typeface, with Helvetica Neue, Helvetica, and Arial fallbacks. Benton Sans's measured proportions and even color on the page reinforce the brand's composed, trustworthy character: contemporary but never trendy, approachable but serious.

At display scale (account headlines, hero statements): 28–40px, weight 600, slightly tight −0.01em tracking. These establish premium membership moments and section identity.

At body scale (statement detail, servicing copy): 16px, weight 400, comfortable 1.55 line height. Amex favors legible, unhurried body text — high-value financial decisions deserve readable type, not density.

Monetary figures and account numbers use weight 600 with tabular figures so balances, payments, and rewards align precisely in statements and summaries. The system avoids hairline weights; everything reads grounded and authoritative.

## 4. Components & Patterns
**Card art tile (the atom)**:
- The cardmember's actual card rendered in brand color or premium navy/metal
- Tier-appropriate finish (Blue, Gold, Platinum); rounded corners, subtle sheen
- Tappable to reveal account summary and controls

**Account summary panel**:
- Current balance, available credit, payment due, minimum due, due date
- Large primary figure with clear secondary metrics in muted ink
- Prominent "Make a Payment" primary blue button

**Membership Rewards balance**:
- Points balance in large figures, with redemption and transfer entry points
- Premium framing for high-balance accounts

**Transaction statement list**:
- Rows: merchant, date, category, amount; pending vs. posted clearly marked
- Tappable rows expand to detail, dispute, and add-receipt actions
- Grouped by statement period with clear headers

**Primary blue button**:
- Amex Blue #006FCF fill, white text, restrained radius (4–8px)
- Secondary actions as blue-outline ghost buttons; one primary per screen

**Premium navy hero**:
- Full-width #00175A surface for Platinum/Centurion experiences and offers
- White headlines, gold accents, elevated benefit messaging

**Offers / benefits card**:
- White rounded card surfacing an Amex Offer or membership benefit
- Merchant logo, value statement, "Add to Card" action

**Payment flow**:
- Step-by-step: amount selection (full/min/custom), funding account, confirm
- Clear figures at each step, single primary blue action

**Status chip**:
- Pill chips in semantic colors with dark or white text
- Transaction and account states: pending / posted / past due / disputed

**Secure servicing banner**:
- Trust-reinforcing element: lock iconography, last-login, fraud-alert entry
- Calm, authoritative styling — security as a premium service, not a warning

## 5. Spacing & Layout
American Express uses an 8px base grid with generous, premium spacing. Content sits in clean white cards on a soft #F7F8F9 background, with comfortable padding (24–32px inside cards) and ample vertical rhythm (24–48px between sections). Whitespace is treated as a luxury signal — screens never feel crowded.

The account dashboard centers around the card art and the primary balance/payment summary, with statements, rewards, offers, and benefits arranged as a clean stack of cards beneath. Page gutters are 16px on mobile and widen substantially on desktop, keeping a calm, centered reading column. Premium navy heroes span full width to mark elevated membership moments.

Transaction rows are comfortably tall (around 56–64px) with clear separation and aligned tabular amounts, prioritizing precise, scannable financial information over packing density.

## 6. Motion & Interaction
**Composed transitions**: motion is calm and deliberate — 240ms eased transitions, never bouncy. The brand's restraint extends to movement: refined, not playful.

**Button press**: 120ms shift to the pressed blue tone with a subtle scale, springing back on release.

**Card reveal**: tapping the card art transitions smoothly to account controls; premium card surfaces have a gentle sheen on appear.

**Payment confirmation**: a successful payment animates a measured check-mark with a brief 240ms reveal on a confirmation surface — reassuring, not celebratory-loud.

**Statement expand**: transaction rows expand to detail with a smooth 200ms height transition; no abrupt jumps in a financial list.

## Accessibility

### Contrast Ratios
- **#FFFFFF on #006FCF Amex Blue**: 4.6:1 — passes AA
- **#FFFFFF on #00175A navy**: 15.4:1 — passes AAA
- **#1A1A1A ink on #FFFFFF canvas**: 17.9:1 — passes AAA
- **#53565A muted on #FFFFFF**: 7.4:1 — passes AAA
- **#86888C subdued on #FFFFFF**: 3.6:1 — fails AA; reserve for non-essential fine print
- **#006FCF blue on #FFFFFF**: 4.5:1 — passes AA; acceptable for links and large text
- **#B7C3D9 on-navy-muted on #00175A**: 8.9:1 — passes AAA
- **#BF9B30 gold on #FFFFFF**: 2.7:1 — fails AA; gold is a decorative tier accent only, never text
- **#1A1A1A ink on #F7F8F9 surface-1**: 17.0:1 — passes AAA

### Minimum Requirements
- **Touch target**: 44×44px minimum across all controls and statement actions
- **Financial figures**: balances, due dates, and amounts exposed as readable labeled text — never as images; tabular alignment must not break screen-reader order
- **Focus indicator**: 2px solid #006FCF outline on light surfaces; 2px solid #FFFFFF outline on navy surfaces — always visible against the active background
- **Status**: account and transaction status (pending/posted/past due) conveyed with text and icon, never semantic color alone

### Motion
- Respects `prefers-reduced-motion`: yes — transitions, card reveal sheen, statement expand, and confirmation reveal reduce to instant or simple fades
- All motion is non-essential; financial state changes apply without animation under reduced motion

### Notes
- Amex Blue (#006FCF) on white at 4.5:1 passes AA but is borderline for small text — prefer it for buttons, large links, and fills; pair with the darker pressed tone (#00509E) where smaller blue text is unavoidable
- Gold (#BF9B30) at 2.7:1 must never carry text — it is strictly a metallic tier accent; tier identity must also be conveyed with a text label
- Premium navy surfaces are a strength for contrast (white-on-navy passes AAA) — favor navy over saturated blue for any dark surface carrying meaningful text
- Security and fraud messaging must remain calm and clear; do not rely on red alone for alerts — pair error color with explicit text and an icon for accessibility and to preserve the composed, trustworthy tone
