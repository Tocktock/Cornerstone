# Cornerstone Color System Refresh

## Summary

This spec records the earlier dark-first visual-system refresh that preceded the workspace-plus-studios redesign.

The current active ownership for reader-versus-studio visual tone now lives in [./workspace-plus-studios-redesign.md](./workspace-plus-studios-redesign.md).

## Scope and owned behavior

This spec owns:
- shared color tokens and surface hierarchy
- typography stack and core shape language
- semantic status and interaction color rules
- route-specific color treatment for dashboard, glossary, graph, decisions, review, and sources
- focus-visible treatment and contrast expectations

## Theme token contract

The frontend must use one canonical semantic token layer for colors, radius, and surface shadow.

Required tokens:
- `--color-canvas`
- `--color-shell`
- `--color-surface-1`
- `--color-surface-2`
- `--color-surface-3`
- `--color-border-subtle`
- `--color-border-strong`
- `--color-text-primary`
- `--color-text-secondary`
- `--color-text-muted`
- `--color-brand`
- `--color-brand-hover`
- `--color-brand-soft`
- `--color-info`
- `--color-info-soft`
- `--color-accent-sage`
- `--color-accent-sage-soft`
- `--color-success`
- `--color-warning`
- `--color-danger`
- `--color-neutral-state`
- `--shadow-surface`
- `--radius-lg`
- `--radius-md`
- `--radius-sm`

The app remains dark-first in this pass. No theme switcher or alternate color mode is added.

## Styling requirements

### Shared shell and components

- The sidebar must use a solid shell surface, not a glass-heavy translucent treatment.
- Panels, nav items, cards, and empty states must read from the semantic surface hierarchy rather than bespoke shades.
- Primary buttons must use a solid brand fill, not a decorative gradient.
- Secondary buttons must use neutral surfaces and stronger borders.
- Links and focus states must use the info color family.
- Eyebrow labels must use a muted warm brand treatment.
- Buttons and inputs must use rounded rectangles rather than fully pill-shaped controls.
- Chips and pills may stay pill-shaped.

### Typography and shape

- The primary stack must start with `IBM Plex Sans`.
- The monospace stack must start with `IBM Plex Mono`.
- This pass uses one sans family for UI and headings; no serif display pairing is introduced yet.
- Heading tracking and component softness should be tightened to reduce the current generic SaaS feel.

### Semantic colors

- Brand is reserved for primary actions and top-level emphasis.
- Info is used for links, focus, query/result guidance, and outbound graph emphasis.
- Accent sage is used for calm secondary emphasis and inbound graph emphasis.
- Success communicates positive or healthy states.
- Warning communicates pending, stale, drifted, or cautionary states.
- Danger communicates errors, degraded states, removed states, and destructive actions.
- Neutral-state communicates superseded, restricted, evidence-only, or otherwise disclosed-but-not-failing states.

## Route-level requirements

### Dashboard

- The search surface should feel inviting but controlled.
- Stat cards should be quieter than the primary action surface.
- Answer and result panels should use cooler secondary accents rather than brand-heavy styling.

### Glossary and Decisions

- These routes should feel the most editorial and readable.
- Surfaces should be lower-chroma and less decorative than dashboard or graph.
- Selected cards must remain clear without relying on neon or glow effects.

### Graph

- Graph is the most intentionally color-coded route.
- Outbound relation emphasis uses the info family.
- Inbound relation emphasis uses the accent-sage family.
- Selected node treatment uses brand-soft rather than a loud brand fill.

### Review

- Review remains controlled and high-stakes.
- Warning is used for pending or cautionary review context.
- Danger is reserved for reject/error states.
- Brand is reserved for the primary officialization path.

### Sources

- Sources should feel operational, with semantics dominating over brand.
- Alert rows must clearly read as danger states.
- Locator and timestamp text remain subdued but readable.

## Constraints and non-goals

- No backend endpoint or payload changes.
- No CSS framework or third-party design dependency additions.
- No route structure rewrite.
- No light theme in this pass.
- No illustration or marketing-style art direction in this pass.

## Related docs

- [./workspace-plus-studios-redesign.md](./workspace-plus-studios-redesign.md)
- [./craft-ui-ux-benchmark.md](./craft-ui-ux-benchmark.md)
- [./ui-ux-polish.md](./ui-ux-polish.md)
- [../graph-and-relations/spec.md](../graph-and-relations/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
