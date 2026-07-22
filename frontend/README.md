# ARBITER frontend

React + Vite + TypeScript UI for the ARBITER dispute engine, built on the American
Express design system with light **and** dark themes.

## Run

```bash
# 1. start the backend (from the repo root)
uvicorn backend.main:app --port 8000

# 2. start the frontend
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api and /ws to :8000
```

## Demo mode

`http://localhost:5173/?demo=1` renders a canned C08 verdict with no backend running —
useful for the walkthrough video and screenshots. Add `&theme=light` or `&theme=dark`
to pin the theme.

## What's here

| File | Role |
|---|---|
| `src/theme.css` | Amex design tokens, both themes. Dark is designed, not inverted. |
| `src/components/Waterfall.tsx` | **The Evidence Ledger waterfall** — the hero. A diverging chart where each bar is one exhibit's exact deciban contribution. |
| `src/components/VerdictCard.tsx` | Verdict, burden statement, reasoning, settlement, recourse — all from ledger fields. |
| `src/components/StatusTracker.tsx` | Live timeline off the immutable event log, with the audit-intact badge and the 45-days-vs-seconds clock. |
| `src/components/IntakeForm.tsx` | Filing with live reason-code classification as you type. |
| `src/lib/api.ts` | Typed client mirroring `backend/schemas.py`. |

## Design notes

- The waterfall's diverging pair (Card Member blue / Merchant gold) was validated with
  the dataviz skill's palette checker: CVD ΔE 26.4 (light), 23.7 (dark), both passing.
  Chart marks are validated separately from the UI blue because dark mode has a
  tighter lightness band.
- Identity is carried by side + label + value, never colour alone (accessibility).
- Tabular figures on all monetary and numeric values, per the Amex spec.
- Respects `prefers-reduced-motion`; focus rings are always visible.
