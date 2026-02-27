# Figma Handoff (MAGI UI)

## 1. Scope
- Source screen: `frontend/app/page.tsx`
- Decomposed components:
  - `frontend/app/components/chamber-visualization.tsx`
  - `frontend/app/components/consensus-panel.tsx`
  - `frontend/app/components/agent-results-grid.tsx`
  - `frontend/app/components/thread-sidebar.tsx`
  - `frontend/app/components/prompt-form.tsx`
  - `frontend/app/components/run-meta-bar.tsx`
  - `frontend/app/components/routing-info-panel.tsx`
  - `frontend/app/components/feedback-panel.tsx`
- Design token source: `frontend/design/tokens.json`

## 2. Recommended import flow
1. Run app (`npm run dev` in `frontend/`) and open the target page.
2. In Figma, use `html.to.design` (or equivalent) to import the page frame.
3. Import `frontend/design/tokens.json` into a token plugin (Tokens Studio etc.).
4. Map token groups to Figma Variables:
   - `color.terminal.*` -> Color
   - `typography.*` -> Typography
   - `spacing.*` / `radius.*` -> Number
   - `shadow.*` -> Effect
5. Convert repeated layers to components:
   - Agent result card
   - Consensus panel
   - Thread item (sidebar)

## 3. Notes for this repo
- This UI uses heavy gradient/effect styling in `frontend/app/globals.css`.
- Some visual effects (animated node states, progress lines) need manual adjustment in Figma.
- `Space Grotesk` is loaded via `next/font/google`; install the same font in Figma for close parity.

## 4. Update policy
When color/spacing/typography changes in `tailwind.config.ts` or `globals.css`, update `frontend/design/tokens.json` in the same PR.
