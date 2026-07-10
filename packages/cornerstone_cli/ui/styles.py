from __future__ import annotations

import json
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


def render_styles(root: Path) -> str:
    token_path = root / "docs" / "design" / "tokens" / "cornerstone_design_tokens_v0_3.json"
    tokens = json.loads(token_path.read_text())
    variables: list[tuple[str, str]] = []

    def flatten(prefix: list[str], value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                flatten([*prefix, _css_name(key)], child)
        elif isinstance(value, list):
            variables.append(("--cs-" + "-".join(prefix), ", ".join(str(item) for item in value)))
        else:
            variables.append(("--cs-" + "-".join(prefix), str(value)))

    flatten([], tokens)
    aliases = {
        "--cs-color-evidence-600": "var(--cs-color-evidence-700)",
        "--cs-color-surface-primary": "var(--cs-color-background-surface)",
        "--cs-color-surface-subtle": "var(--cs-color-background-subtle)",
        "--cs-radius-xs": "var(--cs-radius-sm)",
        "--cs-shadow-sm": "var(--cs-shadow-card)",
        "--cs-state-draft-text": "var(--cs-state-draft-fg)",
        "--cs-state-evidenceBacked-text": "var(--cs-state-evidenceBacked-fg)",
        "--cs-state-searchable-text": "var(--cs-state-searchable-fg)",
        "--cs-state-underReview-text": "var(--cs-state-underReview-fg)",
        "--cs-typography-weight-bold": "var(--cs-typography-display-fontWeight)",
        "--cs-typography-weight-medium": "500",
        "--cs-typography-weight-semibold": "var(--cs-typography-label-fontWeight)",
    }
    variables.extend(aliases.items())
    var_block = "\n".join(f"  {name}: {value};" for name, value in variables)
    return f"""
:root {{
{var_block}
}}
* {{ box-sizing: border-box; }}
html {{ min-height: 100%; background: var(--cs-color-background-app); }}
body {{
  margin: 0;
  min-height: 100%;
  background: var(--cs-color-background-app);
  color: var(--cs-color-text-primary);
  font-family: var(--cs-typography-fontFamily);
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
}}
a {{ color: inherit; text-decoration: none; }}
button, input, textarea, select {{ font: inherit; }}
:where(a, button, input, textarea, select, summary):focus-visible {{
  outline: 2px solid var(--cs-color-border-focus);
  outline-offset: 2px;
}}
.cs-shell {{
  min-height: 100vh;
  min-height: 100dvh;
  display: grid;
  grid-template-columns: var(--cs-layout-sidebarWidth) minmax(0, 1fr);
}}
.cs-sr-only {{
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}}
.cs-icon {{
  display: block;
  width: 20px;
  height: 20px;
  flex: 0 0 auto;
}}
.cs-icon.is-small {{ width: 16px; height: 16px; }}
.cs-icon.is-inverse,
.cs-brand-mark .cs-icon,
.cs-button:not(.secondary):not(.ghost) .cs-icon {{ filter: brightness(0) invert(1); }}
.cs-skip-link {{
  position: fixed;
  left: var(--cs-space-4);
  top: var(--cs-space-4);
  z-index: 10;
  transform: translateY(-160%);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  padding: var(--cs-space-2) var(--cs-space-4);
}}
.cs-skip-link:focus-visible {{ transform: translateY(0); outline: 3px solid var(--cs-color-primary-100); }}
.cs-sidebar {{
  position: sticky;
  top: 0;
  height: 100vh;
  height: 100dvh;
  border-right: 1px solid var(--cs-color-border-default);
  background:
    linear-gradient(180deg, var(--cs-color-surface-primary), color-mix(in srgb, var(--cs-color-surface-subtle) 68%, var(--cs-color-surface-primary)));
  padding: var(--cs-space-6) var(--cs-space-4);
  display: flex;
  flex-direction: column;
  gap: var(--cs-space-5);
  overflow-y: auto;
}}
.cs-brand {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-brand-mark {{
  width: 38px;
  height: 38px;
  border-radius: var(--cs-radius-md);
  background:
    linear-gradient(135deg, var(--cs-color-primary-600), var(--cs-color-primary-700));
  color: var(--cs-color-text-inverse);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-bold);
  box-shadow: 0 10px 24px rgba(37, 87, 209, .18);
}}
.cs-brand-mark .cs-icon {{ width: 22px; height: 22px; }}
.cs-brand-name {{ font-weight: var(--cs-typography-weight-bold); font-size: var(--cs-typography-sectionTitle-fontSize); }}
.cs-brand-sub {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-nav {{ display: grid; gap: var(--cs-space-4); }}
.cs-nav-group {{ display: grid; gap: var(--cs-space-2); }}
.cs-nav-label {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
  font-weight: var(--cs-typography-weight-semibold);
  text-transform: uppercase;
}}
.cs-nav a {{
  min-height: 40px;
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-2) var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  display: grid;
  grid-template-columns: 26px minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-2);
  font-weight: var(--cs-typography-weight-medium);
  transition: background .18s ease, color .18s ease, box-shadow .18s ease;
}}
.cs-nav a:hover, .cs-nav a:focus-visible {{ background: var(--cs-color-surface-primary); box-shadow: var(--cs-shadow-sm); }}
.cs-nav a[aria-current="page"] {{
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  box-shadow: inset 3px 0 0 var(--cs-color-primary-600);
}}
.cs-nav-mark {{
  width: 26px;
  height: 26px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--cs-color-surface-primary) 72%, var(--cs-color-surface-subtle));
  color: var(--cs-color-text-muted);
  opacity: .72;
}}
.cs-nav a[aria-current="page"] .cs-nav-mark {{
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-primary-700);
  opacity: 1;
}}
.cs-nav-count {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-workspace-switcher {{
  margin-top: auto;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid !important;
  grid-template-columns: auto minmax(0, 1fr) auto !important;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-secondary);
  box-shadow: none !important;
}}
.cs-workspace-switcher:hover,
.cs-workspace-switcher:focus-visible {{
  border-color: var(--cs-color-border-strong);
  background: var(--cs-color-surface-primary) !important;
}}
.cs-workspace-switcher > span:nth-child(2) {{ min-width: 0; display: grid; gap: 1px; }}
.cs-workspace-switcher strong,
.cs-workspace-switcher small {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.cs-workspace-switcher strong {{ color: var(--cs-color-text-primary); }}
.cs-workspace-switcher small {{
  font-size: var(--cs-typography-metadata-fontSize);
  color: var(--cs-color-text-muted);
}}
.cs-workspace-icon {{
  width: 32px;
  height: 32px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
}}
.cs-main {{ min-width: 0; }}
.cs-topbar {{
  min-height: var(--cs-layout-headerHeight);
  border-bottom: 1px solid var(--cs-color-border-default);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 92%, transparent);
  backdrop-filter: blur(12px);
  display: flex;
  align-items: center;
  gap: var(--cs-space-4);
  justify-content: space-between;
  padding: var(--cs-space-4) var(--cs-layout-contentGutter);
  position: sticky;
  top: 0;
  z-index: 2;
}}
.cs-command {{
  flex: 1 1 auto;
  max-width: 860px;
  min-width: 280px;
}}
.cs-search {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: 2px var(--cs-space-3);
  width: 100%;
  min-height: 48px;
}}
.cs-search:focus-within {{
  border-color: var(--cs-color-border-focus);
  box-shadow: 0 0 0 3px var(--cs-color-primary-50);
}}
.cs-search-icon {{ color: var(--cs-color-text-muted); display: grid; place-items: center; }}
.cs-search input {{ border: 0; outline: 0; min-width: 0; flex: 1; color: var(--cs-color-text-primary); background: transparent; }}
.cs-search button, .cs-button {{
  border: 1px solid var(--cs-color-primary-600);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  padding: var(--cs-space-2) var(--cs-space-4);
  min-height: 44px;
  font-weight: var(--cs-typography-weight-semibold);
  cursor: pointer;
  transition: background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--cs-space-2);
}}
.cs-search button {{ width: 44px; min-width: 44px; min-height: 44px; padding: 0; }}
.cs-search button:hover, .cs-button:hover {{ box-shadow: var(--cs-shadow-sm); transform: translateY(-1px); }}
.cs-search button:active, .cs-button:active {{ transform: translateY(0); }}
.cs-topbar-actions {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-topbar-workspace {{
  display: grid;
  gap: 1px;
  padding-inline: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: 1.25;
}}
.cs-topbar-workspace strong {{ color: var(--cs-color-text-primary); font-weight: var(--cs-typography-weight-semibold); }}
.cs-review-link {{
  min-height: 44px;
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2) var(--cs-space-3);
  color: var(--cs-color-text-secondary);
}}
.cs-review-link:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-review-link:focus-visible {{ border-color: var(--cs-color-border-focus); box-shadow: var(--cs-shadow-focus); }}
.cs-review-link strong {{
  min-width: 20px;
  height: 20px;
  border-radius: var(--cs-radius-pill);
  display: grid;
  place-items: center;
  padding-inline: 5px;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-help-menu {{ position: relative; }}
.cs-help-menu > summary {{ list-style: none; cursor: pointer; }}
.cs-help-menu > summary::-webkit-details-marker {{ display: none; }}
.cs-help-menu[open] > summary,
.cs-help-menu > summary:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-help-menu > summary:focus-visible {{ border-color: var(--cs-color-border-focus); box-shadow: var(--cs-shadow-focus); }}
.cs-help-glyph {{
  display: block;
  width: 20px;
  height: 20px;
  color: var(--cs-color-text-primary);
  font-size: 20px;
  font-weight: var(--cs-typography-weight-bold);
  line-height: 20px;
  text-align: center;
}}
.cs-help-popover {{
  position: absolute;
  z-index: 4;
  inset: calc(100% + var(--cs-space-2)) 0 auto auto;
  width: min(320px, calc(100vw - var(--cs-space-6)));
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  box-shadow: var(--cs-shadow-card);
  padding: var(--cs-space-4);
}}
.cs-help-popover p {{ margin: var(--cs-space-2) 0; color: var(--cs-color-text-secondary); }}
.cs-help-popover a {{ color: var(--cs-color-primary-700); font-weight: var(--cs-typography-weight-semibold); }}
.cs-icon-button {{
  width: 44px;
  height: 44px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-secondary);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-avatar {{
  min-width: 44px;
  height: 44px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-subtle);
  border: 1px solid var(--cs-color-border-default);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-button.secondary {{
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-primary);
  border-color: var(--cs-color-border-strong);
}}
.cs-button.ghost {{
  background: transparent;
  color: var(--cs-color-text-secondary);
  border-color: transparent;
}}
.cs-kicker {{ color: var(--cs-color-primary-700); font-weight: var(--cs-typography-weight-semibold); font-size: var(--cs-typography-label-fontSize); }}
.cs-content {{ padding: var(--cs-layout-contentGutter); max-width: 1360px; width: 100%; min-width: 0; margin-inline: auto; }}
.cs-page-head {{ display: grid; gap: var(--cs-space-2); margin-bottom: var(--cs-space-6); max-width: 760px; }}
.cs-page-head h1, .cs-hero h1 {{
  margin: 0;
  font-size: var(--cs-typography-pageTitle-fontSize);
  line-height: var(--cs-typography-pageTitle-lineHeight);
  letter-spacing: 0;
}}
.cs-hero h1 {{ font-size: var(--cs-typography-display-fontSize); line-height: var(--cs-typography-display-lineHeight); }}
.cs-page-head p, .cs-hero p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 760px; }}
.cs-grid-hero {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(320px, .55fr); gap: var(--cs-space-6); align-items: start; }}
.cs-grid-two {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 380px); gap: var(--cs-space-6); align-items: start; }}
.cs-stack {{ display: grid; gap: var(--cs-space-4); }}
.cs-panel {{
  background: var(--cs-color-surface-primary);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  box-shadow: var(--cs-shadow-sm);
  padding: var(--cs-layout-cardPadding);
}}
.cs-panel.flat {{ box-shadow: none; }}
.cs-panel-header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: var(--cs-space-4); margin-bottom: var(--cs-space-4); }}
.cs-panel-header h2, .cs-section-title {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-muted {{ color: var(--cs-color-text-muted); }}
.cs-meta {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); line-height: var(--cs-typography-metadata-lineHeight); }}
.cs-home-intro {{
  min-height: calc(100vh - var(--cs-layout-headerHeight) - (var(--cs-layout-contentGutter) * 2));
  min-height: calc(100dvh - var(--cs-layout-headerHeight) - (var(--cs-layout-contentGutter) * 2));
  align-content: start;
}}
.cs-home-layout {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--cs-space-6); align-items: start; }}
.cs-home-layout.has-activity {{ grid-template-columns: minmax(0, 1fr) minmax(280px, 340px); }}
.cs-home-primary {{ min-width: 0; }}
.cs-home-activity {{ position: sticky; top: calc(var(--cs-layout-headerHeight) + var(--cs-layout-contentGutter)); }}
.cs-home-canvas {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  box-shadow: var(--cs-shadow-sm);
  padding: var(--cs-space-5);
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-home-canvas .cs-panel-header {{ margin-bottom: 0; flex-wrap: wrap; }}
.cs-home-canvas p {{ max-width: 62ch; }}
.cs-home-workspace {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-home-source-row {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  justify-content: flex-start;
}}
.cs-home-paste-row {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: stretch;
}}
.cs-home-source-note {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
  text-align: center;
}}
.cs-drop {{
  min-height: 166px;
  border: 1px dashed var(--cs-color-border-strong);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--cs-color-primary-50) 34%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary));
  display: grid;
  gap: var(--cs-space-3);
  padding: var(--cs-space-4);
  align-content: center;
}}
.cs-drop.is-hot {{ border-color: var(--cs-color-primary-600); background: var(--cs-color-primary-50); }}
.cs-drop textarea, .cs-field {{
  width: 100%;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-primary);
  padding: var(--cs-space-3);
  outline: none;
}}
.cs-drop textarea {{ min-height: 130px; resize: vertical; }}
.cs-drop textarea:focus, .cs-field:focus {{ border-color: var(--cs-color-border-focus); box-shadow: 0 0 0 3px var(--cs-color-primary-50); }}
.cs-drop-target {{
  display: grid;
  gap: var(--cs-space-2);
  grid-template-columns: auto minmax(0, 1fr);
  place-items: center start;
  text-align: left;
  padding: 0;
}}
.cs-drop-mark {{
  width: 46px;
  height: 46px;
  border-radius: var(--cs-radius-pill);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
  border: 1px solid var(--cs-color-primary-100);
  font-size: 18px;
}}
.cs-drop textarea.cs-drop-input {{
  min-height: 44px;
  background: var(--cs-color-surface-primary);
}}
.cs-or-divider {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: center;
  gap: var(--cs-space-3);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-or-divider::before, .cs-or-divider::after {{
  content: "";
  height: 1px;
  background: var(--cs-color-border-default);
}}
.cs-ask-bar {{
  border: 1px solid var(--cs-color-border-focus);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2) var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(140px, .72fr) minmax(160px, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
  box-shadow: var(--cs-shadow-sm);
}}
.cs-ask-bar > * {{ min-width: 0; }}
.cs-ask-mark {{
  width: 34px;
  height: 34px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-ask-bar .cs-field {{
  border: 0;
  padding: var(--cs-space-2);
  box-shadow: none;
}}
.cs-ask-bar .cs-field:focus {{ box-shadow: none; }}
.cs-suggestion-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: var(--cs-space-2); }}
.cs-suggestion-row .cs-button {{ min-width: 0; justify-content: center; white-space: normal; }}
.cs-home-loop-inline {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--cs-space-2);
  max-width: 460px;
}}
.cs-home-loop-inline span {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-secondary);
  padding: var(--cs-space-1) var(--cs-space-2);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
  font-weight: var(--cs-typography-weight-medium);
  white-space: nowrap;
}}
.cs-home-loop-inline span:first-child {{
  border-color: var(--cs-color-primary-100);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
}}
.cs-home-loop-inline strong {{
  color: var(--cs-color-text-primary);
}}
.cs-home-item-list {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  overflow: hidden;
  background: var(--cs-color-surface-primary);
}}
.cs-home-item {{
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  min-height: 72px;
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-home-item:last-child {{ border-bottom: 0; }}
.cs-home-item:hover {{ background: var(--cs-color-surface-subtle); }}
.cs-home-item-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-home-item-icon .cs-icon, .cs-activity-icon .cs-icon, .cs-collection-icon .cs-icon, .cs-empty-mark .cs-icon {{ width: 20px; height: 20px; }}
.cs-home-item h3 {{ margin: 0; font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-home-item p {{ margin: 0; color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-activity-list {{
  display: grid;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  overflow: hidden;
  background: var(--cs-color-surface-primary);
}}
.cs-activity-row {{
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
  min-height: 74px;
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-activity-row:last-child {{ border-bottom: 0; }}
.cs-activity-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-activity-row strong, .cs-activity-row p {{ margin: 0; }}
.cs-next-step-list {{ display: grid; gap: var(--cs-space-2); }}
.cs-next-step {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  padding: var(--cs-space-3);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
}}
.cs-next-step strong {{ font-size: var(--cs-typography-body-fontSize); }}
.cs-row {{ display: flex; align-items: center; gap: var(--cs-space-3); flex-wrap: wrap; }}
.cs-module-grid {{ display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(280px, .95fr); gap: var(--cs-space-4); }}
.cs-list {{ display: grid; gap: var(--cs-space-3); }}
.cs-list-row {{
  min-height: var(--cs-layout-listRowHeight);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-list-row:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-list-row h3 {{ margin: 0 0 var(--cs-space-1); font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-list-row p {{ margin: 0; color: var(--cs-color-text-secondary); }}
.cs-list-row.compact {{ padding: var(--cs-space-3); min-height: auto; }}
.cs-search-page {{ display: grid; gap: var(--cs-space-4); max-width: 1120px; }}
.cs-search-page > .cs-page-head {{ margin-bottom: 0; }}
.cs-search-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(300px, 340px);
  gap: var(--cs-space-5);
  align-items: start;
}}
.cs-search-main {{
  display: grid;
  gap: var(--cs-space-4);
  min-width: 0;
}}
.cs-search-canvas {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-search-command {{
  display: grid;
  gap: var(--cs-space-2);
  align-items: start;
}}
.cs-search-back {{
  color: var(--cs-color-primary-700);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-search-copy {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-search-titleline {{
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: var(--cs-space-3);
}}
.cs-search-copy h1 {{
  margin: 0;
  max-width: 34ch;
  font-size: 26px;
  line-height: 1.18;
  text-wrap: balance;
}}
.cs-search-mode {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-search-hero {{
  border: 1px solid var(--cs-color-border-focus);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  box-shadow: var(--cs-shadow-focus);
  padding: var(--cs-space-2);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-2);
  align-items: center;
}}
.cs-search-lens {{
  min-width: 42px;
  min-height: 42px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-search-hero input {{
  min-height: 50px;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--cs-color-text-primary);
  padding: 0 var(--cs-space-2);
  font-size: 15px;
}}
.cs-search-submit {{
  min-width: 44px;
  min-height: 44px;
  justify-content: center;
}}
.cs-search-tabs, .cs-filter-row {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-search-tab, .cs-filter-chip {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-height: 34px;
  border-radius: var(--cs-radius-md);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-secondary);
  padding: 0 var(--cs-space-3);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-medium);
}}
.cs-search-tab {{ min-height: 44px; }}
.cs-search-tab.is-active {{
  background: var(--cs-color-primary-50);
  border-color: var(--cs-color-primary-100);
  color: var(--cs-color-primary-700);
}}
.cs-search-filterbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
}}
.cs-search-context {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-search-context h2 {{
  margin: 0;
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
  color: var(--cs-color-text-muted);
}}
.cs-result-list {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-result-list-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-result-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr) minmax(150px, 210px);
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-result-row:hover {{ border-color: var(--cs-color-border-strong); box-shadow: var(--cs-shadow-sm); }}
.cs-result-icon {{
  width: 34px;
  height: 38px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-size: 11px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-result-icon .cs-icon {{ width: 18px; height: 18px; }}
.cs-result-icon.is-source {{ background: var(--cs-color-primary-50); color: var(--cs-color-primary-700); }}
.cs-result-icon.is-brief {{ background: var(--cs-state-underReview-bg); color: var(--cs-state-underReview-text); }}
.cs-result-icon.is-claim {{ background: var(--cs-state-searchable-bg); color: var(--cs-state-searchable-text); }}
.cs-result-icon.is-action {{ background: var(--cs-state-draft-bg); color: var(--cs-state-draft-text); }}
.cs-result-body {{ display: grid; gap: var(--cs-space-1); }}
.cs-result-body h3 {{ margin: 0; font-size: 16px; line-height: 1.35; }}
.cs-result-body h3 a {{ color: var(--cs-color-text-primary); }}
.cs-result-body h3 a:hover, .cs-result-body h3 a:focus-visible {{ color: var(--cs-color-primary-700); }}
.cs-result-body p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 78ch; }}
.cs-result-meta {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-result-type {{
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-result-support {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-result-support .cs-meta {{
  line-height: 1.35;
  text-align: right;
  max-width: 180px;
}}
.cs-result-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-result-actions .cs-button {{ min-height: 44px; padding: var(--cs-space-1) var(--cs-space-3); }}
.cs-search-rail {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-right-stat {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  padding: var(--cs-space-2) 0;
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-right-stat:last-child {{ border-bottom: 0; }}
.cs-right-stat-label {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-right-stat-icon {{
  width: 24px;
  height: 24px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-size: 10px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-suggested-query {{
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: start;
  color: var(--cs-color-text-secondary);
  padding: var(--cs-space-2) 0;
}}
.cs-suggested-query span:first-child {{
  width: 20px;
  height: 20px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-semibold);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-artifact-hero {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
  margin-bottom: var(--cs-space-5);
}}
.cs-artifact-title {{ display: grid; gap: var(--cs-space-2); }}
.cs-artifact-title h1 {{
  margin: 0;
  font-size: var(--cs-typography-pageTitle-fontSize);
  line-height: var(--cs-typography-pageTitle-lineHeight);
}}
.cs-artifact-actions {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); justify-content: flex-end; }}
.cs-artifact-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 400px);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-artifact-compact-hero {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
  padding-bottom: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-artifact-compact-hero > * {{
  min-width: 0;
}}
.cs-artifact-compact-hero .cs-artifact-title h1 {{
  max-width: 44ch;
  font-size: 27px;
  line-height: 1.12;
  text-wrap: balance;
  overflow-wrap: anywhere;
}}
.cs-artifact-compact-hero .cs-artifact-actions {{
  justify-content: flex-start;
  max-width: 100%;
}}
.cs-artifact-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  min-width: 0;
}}
.cs-artifact-breadcrumb a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-breadcrumb span:last-child {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.cs-artifact-title-row {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-artifact-file-mark {{
  width: 46px;
  height: 46px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  font-size: var(--cs-typography-label-fontSize);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-artifact-file-mark .cs-icon {{ filter: brightness(0) invert(1); }}
.cs-metadata-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-5);
}}
.cs-metadata-item {{
  border-left: 1px solid var(--cs-color-border-default);
  padding-left: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-metadata-strip.is-artifact {{
  grid-template-columns: repeat(5, minmax(0, 1fr));
  border-bottom: 1px solid var(--cs-color-border-default);
  padding: var(--cs-space-3) 0;
  margin-bottom: 0;
}}
.cs-metadata-item strong {{
  word-break: break-word;
}}
.cs-artifact-inspection-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-artifact-inspection-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-artifact-inspection-card strong {{
  font-size: 18px;
  line-height: 1.25;
}}
.cs-artifact-viewer {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
  box-shadow: var(--cs-shadow-sm);
  margin-top: var(--cs-space-2);
}}
.cs-artifact-toolbar {{
  min-height: 48px;
  border-bottom: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-primary);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  padding: var(--cs-space-2) var(--cs-space-3);
}}
.cs-artifact-toolbar-label {{
  display: grid;
  gap: 2px;
}}
.cs-artifact-toolgroup {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-artifact-tool {{
  min-width: 32px;
  height: 32px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-tool.is-muted {{
  color: var(--cs-color-text-muted);
  background: var(--cs-color-surface-subtle);
}}
.cs-artifact-page-count {{
  min-width: 72px;
  height: 32px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-document-frame {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-4);
}}
.cs-document-frame.has-rail {{
  border: 0;
  border-radius: 0;
  padding: 0;
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  min-height: 600px;
}}
.cs-artifact-page-rail {{
  border-right: 1px solid var(--cs-color-border-default);
  background: color-mix(in srgb, var(--cs-color-surface-subtle) 74%, var(--cs-color-surface-primary));
  padding: var(--cs-space-3) var(--cs-space-2);
  display: grid;
  align-content: start;
  gap: var(--cs-space-2);
}}
.cs-artifact-page-rail-label {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-thumb {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  min-height: 92px;
  padding: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-1);
  color: var(--cs-color-text-muted);
  font-size: 10px;
}}
.cs-artifact-thumb.is-active {{
  border-color: var(--cs-color-primary-500);
  box-shadow: var(--cs-shadow-focus);
}}
.cs-artifact-thumb-line {{
  height: 5px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-border-default);
}}
.cs-artifact-thumb span {{
  text-align: center;
  margin-top: var(--cs-space-1);
}}
.cs-artifact-page-area {{
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--cs-color-surface-subtle) 72%, var(--cs-color-surface-primary)), var(--cs-color-surface-subtle));
  padding: var(--cs-space-4);
  overflow: auto;
}}
.cs-document-page {{
  max-width: 760px;
  min-height: 540px;
  margin: 0 auto;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  box-shadow: 0 18px 40px rgba(15, 23, 42, .08);
  padding: clamp(var(--cs-space-5), 5vw, var(--cs-space-8));
}}
.cs-document-heading {{
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-5);
  padding-bottom: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-document-heading h3 {{
  margin: 0;
  font-size: 20px;
  line-height: 1.35;
}}
.cs-artifact-source-note {{
  display: flex;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-document-page .cs-source-text {{
  border: 0;
  border-radius: 0;
  background: transparent;
  padding: 0;
  line-height: 1.75;
}}
.cs-artifact-rail {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-artifact-rail-tabs {{
  display: flex;
  gap: var(--cs-space-5);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-4);
  overflow-x: auto;
}}
.cs-artifact-rail-tab {{
  padding: 0 0 var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
  white-space: nowrap;
}}
.cs-artifact-rail-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-bottom: 2px solid var(--cs-color-primary-600);
}}
.cs-artifact-panel-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-artifact-summary-lead {{
  border: 1px solid var(--cs-color-primary-100);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-primary-50) 42%, var(--cs-color-surface-primary));
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-artifact-summary-lead strong {{
  color: var(--cs-color-text-primary);
  line-height: 1.35;
}}
.cs-artifact-summary-lead p {{
  line-height: 1.65;
}}
.cs-artifact-side-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-artifact-side-card h2 {{
  margin: 0;
}}
.cs-artifact-side-card p {{
  margin: 0;
}}
.cs-artifact-side-card summary {{
  cursor: pointer;
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-artifact-side-card[open] summary {{
  margin-bottom: var(--cs-space-2);
}}
.cs-keyword-list {{ display: grid; gap: var(--cs-space-2); }}
.cs-keyword-row {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
}}
.cs-inbox-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(320px, .8fr);
  gap: var(--cs-space-6);
  align-items: start;
}}
.cs-inbox-list-heading {{
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-3);
}}
.cs-inbox-list-heading h2, .cs-inbox-list-heading p {{ margin: 0; }}
.cs-inbox-lane-summary {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  margin-bottom: var(--cs-space-3);
  color: var(--cs-color-text-secondary);
}}
.cs-inbox-summary-main {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-inbox-summary-main strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-inbox-summary-pills {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-inbox-summary-pill {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-primary);
  padding: 4px var(--cs-space-2);
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-inbox-summary-pill.is-active {{
  border-color: var(--cs-color-border-focus);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
}}
.cs-inbox-summary-pill strong {{
  color: var(--cs-color-text-primary);
  font-variant-numeric: tabular-nums;
}}
.cs-inbox-tabs {{
  display: flex;
  gap: var(--cs-space-4);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-4);
  overflow-x: auto;
  scrollbar-width: thin;
}}
.cs-inbox-tab {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-height: 42px;
  color: var(--cs-color-text-secondary);
  border-bottom: 2px solid transparent;
  font-weight: var(--cs-typography-weight-medium);
  white-space: nowrap;
}}
.cs-inbox-tab strong {{
  min-width: 22px;
  height: 22px;
  padding-inline: 6px;
  border-radius: var(--cs-radius-pill);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-inbox-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-color: var(--cs-color-primary-600);
}}
.cs-inbox-tab.is-active strong {{ background: var(--cs-color-primary-50); color: var(--cs-color-primary-700); }}
.cs-inbox-toolbar {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
  margin-bottom: var(--cs-space-4);
}}
.cs-inbox-toolbar .cs-filter-row {{ margin-top: 0; }}
.cs-inbox-filter-label {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-height: 34px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: 0 var(--cs-space-3);
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-inbox-filter-label span {{
  color: var(--cs-color-primary-700);
}}
.cs-inbox-table {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
}}
.cs-inbox-row {{
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-inbox-row {{
  padding: var(--cs-space-4);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-inbox-row:last-child {{ border-bottom: 0; }}
.cs-inbox-row:hover {{ background: var(--cs-color-surface-subtle); }}
.cs-inbox-row.is-selected {{
  background: var(--cs-color-primary-50);
  box-shadow: inset 3px 0 0 var(--cs-color-primary-600);
}}
.cs-inbox-select {{
  width: 16px;
  height: 16px;
  border: 1px solid var(--cs-color-border-strong);
  border-radius: var(--cs-radius-xs);
  display: grid;
  place-items: center;
  color: var(--cs-color-surface-primary);
  font-size: 11px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-inbox-row.is-selected .cs-inbox-select {{
  border-color: var(--cs-color-primary-600);
  background: var(--cs-color-primary-600);
}}
.cs-inbox-item-title {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 2px;
  align-items: start;
}}
.cs-inbox-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-inbox-icon .cs-icon {{ width: 18px; height: 18px; }}
.cs-inbox-item-title strong {{ display: block; }}
.cs-inbox-item-title .cs-meta {{ display: block; }}
.cs-inbox-row-state {{ display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: var(--cs-space-2); }}
.cs-inbox-type-cell {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-secondary);
}}
.cs-inbox-type-mark {{
  width: 20px;
  height: 20px;
  border-radius: var(--cs-radius-xs);
  display: grid;
  place-items: center;
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-primary-700);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-inbox-owner {{
  display: inline-grid;
  grid-template-columns: 24px minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: center;
  min-width: 0;
}}
.cs-inbox-owner-mark {{
  width: 22px;
  height: 22px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
  border: 1px solid var(--cs-color-border-default);
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-inbox-detail {{
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-inbox-detail h2 {{ margin: 0; font-size: var(--cs-typography-sectionTitle-fontSize); }}
.cs-inbox-detail-title {{
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-inbox-context {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-3);
}}
.cs-inbox-context > summary {{ cursor: pointer; min-height: 36px; color: var(--cs-color-text-secondary); }}
.cs-inbox-context[open] > summary {{ margin-bottom: var(--cs-space-3); color: var(--cs-color-text-primary); }}
.cs-inbox-context > p {{ color: var(--cs-color-text-secondary); }}
.cs-audit-filters {{
  display: grid;
  grid-template-columns: minmax(220px, 1fr) minmax(180px, .7fr) auto auto;
  gap: var(--cs-space-3);
  align-items: end;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  margin-bottom: var(--cs-space-5);
}}
.cs-audit-filters label {{ display: grid; gap: var(--cs-space-1); color: var(--cs-color-text-secondary); font-size: var(--cs-typography-metadata-fontSize); font-weight: var(--cs-typography-weight-semibold); }}
.cs-audit-filters select {{ min-height: 44px; border: 1px solid var(--cs-color-border-default); border-radius: var(--cs-radius-md); background: var(--cs-color-surface-primary); color: var(--cs-color-text-primary); padding: 0 var(--cs-space-3); min-width: 0; }}
.cs-audit-pagination {{ display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: var(--cs-space-3); margin-top: var(--cs-space-4); color: var(--cs-color-text-secondary); }}
.cs-audit-icon .cs-icon {{ width: 18px; height: 18px; }}
.cs-inbox-close {{
  width: 28px;
  height: 28px;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  color: var(--cs-color-text-muted);
  background: var(--cs-color-surface-primary);
}}
.cs-inbox-action-panel {{
  border: 1px solid var(--cs-color-border-focus);
  border-radius: var(--cs-radius-md);
  background: linear-gradient(180deg, var(--cs-color-primary-50), var(--cs-color-surface-primary));
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-inbox-action-panel .cs-section-title {{ margin: 0; }}
.cs-inbox-preview-note {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-inbox-preview-note h3 {{
  margin: 0;
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
}}
.cs-inbox-preview-note p {{ margin: 0; color: var(--cs-color-text-secondary); }}
.cs-journey-timeline {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
  min-width: 0;
}}
.cs-journey-timeline.is-recovery {{
  border-color: var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
}}
.cs-journey-header {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  min-width: 0;
}}
.cs-journey-header h3 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-journey-header p {{ margin: var(--cs-space-1) 0 0; color: var(--cs-color-text-secondary); }}
.cs-journey-stage-list {{
  list-style: none;
  margin: 0;
  padding: 0;
  min-width: 0;
}}
.cs-journey-stage-list > li {{ min-width: 0; }}
.cs-journey-stage {{
  border-left: 3px solid var(--cs-color-border-strong);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  min-width: 0;
}}
.cs-journey-stage.is-ready {{
  border-left-color: var(--cs-state-saved-border);
  background: var(--cs-state-saved-bg);
}}
.cs-journey-stage.is-needs-review {{
  border-left-color: var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
}}
.cs-journey-stage.is-blocked {{
  border-left-color: var(--cs-state-policyBlocked-border);
  background: var(--cs-state-policyBlocked-bg);
}}
.cs-journey-stage.is-ready .cs-dot {{ background: var(--cs-state-saved-fg); }}
.cs-journey-stage.is-needs-review .cs-dot {{ background: var(--cs-state-underReview-fg); }}
.cs-journey-stage.is-blocked .cs-dot {{ background: var(--cs-state-policyBlocked-fg); }}
.cs-journey-stage-body {{ display: grid; gap: var(--cs-space-2); min-width: 0; }}
.cs-journey-stage-body p {{ margin: 0; color: var(--cs-color-text-secondary); overflow-wrap: anywhere; }}
.cs-journey-stage-heading {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-journey-ref-grid {{
  margin: var(--cs-space-2) 0 0;
  display: grid;
  grid-template-columns: minmax(88px, auto) minmax(0, 1fr);
  gap: var(--cs-space-1) var(--cs-space-2);
}}
.cs-journey-ref-grid dt {{ color: var(--cs-color-text-muted); }}
.cs-journey-ref-grid dd {{ margin: 0; min-width: 0; overflow-wrap: anywhere; word-break: break-word; }}
.cs-journey-ref-grid code {{ display: inline-block; max-width: 100%; overflow-wrap: anywhere; word-break: break-word; }}
.cs-journey-recovery {{ border-top: 1px solid var(--cs-color-border-default); padding-top: var(--cs-space-2); }}
.cs-journey-recovery summary {{
  cursor: pointer;
  min-height: 32px;
  padding: var(--cs-space-1) 0;
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-journey-recovery summary:focus-visible,
.cs-journey-stage summary:focus-visible {{
  outline: 2px solid var(--cs-color-border-focus);
  outline-offset: 2px;
}}
.cs-journey-recovery-list {{ display: grid; gap: var(--cs-space-2); margin-top: var(--cs-space-2); }}
.cs-journey-recovery-list > div {{
  border-left: 2px solid var(--cs-state-underReview-border);
  padding-left: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-1);
  color: var(--cs-color-text-secondary);
}}
.cs-journey-recovery-list strong {{ color: var(--cs-color-text-primary); }}
.cs-inbox-linked-list {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-inbox-linked-row {{
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: start;
  color: var(--cs-color-text-secondary);
}}
.cs-inbox-linked-row strong {{
  display: block;
  color: var(--cs-color-text-primary);
}}
.cs-inbox-receipt-strip {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-inbox-receipt {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-inbox-receipt strong {{ font-size: var(--cs-typography-metadata-fontSize); }}
.cs-inbox-receipt span {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-inbox-actions {{ display: grid; gap: var(--cs-space-2); }}
.cs-inbox-actions .cs-button {{ justify-content: center; text-align: center; }}
.cs-inbox-foot {{
  padding: var(--cs-space-3) var(--cs-space-4);
  border-top: 1px solid var(--cs-color-border-default);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-collection-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
  gap: var(--cs-space-6);
  align-items: start;
}}
.cs-collection-summary {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-collection-stat {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-collection-stat strong {{
  font-size: 22px;
  line-height: 1.2;
  font-variant-numeric: tabular-nums;
}}
.cs-collection-toolbar {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  margin-bottom: var(--cs-space-4);
}}
.cs-collection-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-collection-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) minmax(176px, auto);
  gap: var(--cs-space-4);
  align-items: start;
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}}
.cs-collection-row:hover {{
  border-color: var(--cs-color-border-strong);
  box-shadow: var(--cs-shadow-sm);
  transform: translateY(-1px);
}}
.cs-collection-icon {{
  width: 32px;
  height: 32px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-collection-body {{ display: grid; gap: var(--cs-space-2); }}
.cs-collection-body h3 {{ margin: 0; font-size: 16px; line-height: 1.35; }}
.cs-collection-body p {{ margin: 0; color: var(--cs-color-text-secondary); max-width: 82ch; }}
.cs-collection-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-collection-actions {{
  display: grid;
  gap: var(--cs-space-3);
  justify-items: end;
  align-content: start;
}}
.cs-collection-actions .cs-row {{ justify-content: flex-end; }}
.cs-collection-cta {{
  min-height: 32px;
  border-radius: var(--cs-radius-sm);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  padding: 6px 10px;
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-collection-footrail {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-2);
  margin-top: var(--cs-space-1);
}}
.cs-collection-stage {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
  min-width: 0;
}}
.cs-collection-stage strong {{
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
  color: var(--cs-color-text-primary);
}}
.cs-collection-stage span {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  overflow-wrap: anywhere;
}}
.cs-queue-focus {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-2);
  margin-bottom: var(--cs-space-3);
}}
.cs-queue-focus-head {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  align-items: center;
}}
.cs-queue-focus h2 {{
  margin: 0;
  font-size: 17px;
  line-height: 1.35;
}}
.cs-queue-focus p {{
  margin: 2px 0 0;
  color: var(--cs-color-text-secondary);
  max-width: 72ch;
}}
.cs-queue-lanes {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
  padding-top: var(--cs-space-2);
  border-top: 1px solid var(--cs-color-border-default);
}}
.cs-queue-lane {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-pill);
  background: var(--cs-color-surface-subtle);
  padding: 4px var(--cs-space-2);
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-secondary);
}}
.cs-queue-lane strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-queue-lane span {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-empty {{
  border: 1px dashed var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  padding: var(--cs-space-6);
  color: var(--cs-color-text-muted);
  background: var(--cs-color-surface-subtle);
}}
.cs-empty-state {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, var(--cs-color-surface-primary), var(--cs-color-surface-subtle));
  padding: var(--cs-space-6);
  display: grid;
  gap: var(--cs-space-4);
  color: var(--cs-color-text-primary);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.68);
}}
.cs-empty-state-main {{
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-empty-mark {{
  width: 44px;
  height: 44px;
  border-radius: var(--cs-radius-lg);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-empty-copy {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-empty-copy h2 {{
  margin: 0;
  font-size: 20px;
  line-height: 1.3;
  text-wrap: balance;
}}
.cs-empty-copy p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  max-width: 64ch;
  text-wrap: pretty;
}}
.cs-empty-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-empty-steps {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-empty-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 72%, var(--cs-color-surface-subtle));
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-empty-briefing {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 280px);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-empty-briefing h3 {{
  margin: 0 0 var(--cs-space-2);
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
}}
.cs-empty-receipts {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-empty-receipt {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-empty-receipt strong {{
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
}}
.cs-empty-receipt span {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-chip {{
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  border-radius: var(--cs-radius-pill);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-text-secondary);
  padding: 0 var(--cs-space-2);
  font-size: var(--cs-typography-label-fontSize);
  line-height: var(--cs-typography-label-lineHeight);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-chip-saved {{ background: var(--cs-state-saved-bg); border-color: var(--cs-state-saved-border); color: var(--cs-state-saved-fg); }}
.cs-chip-searchable {{ background: var(--cs-state-searchable-bg); border-color: var(--cs-state-searchable-border); color: var(--cs-state-searchable-fg); }}
.cs-chip-draft {{ background: var(--cs-state-draft-bg); border-color: var(--cs-state-draft-border); color: var(--cs-state-draft-fg); }}
.cs-chip-evidenceBacked {{ background: var(--cs-state-evidenceBacked-bg); border-color: var(--cs-state-evidenceBacked-border); color: var(--cs-state-evidenceBacked-fg); }}
.cs-chip-underReview {{ background: var(--cs-state-underReview-bg); border-color: var(--cs-state-underReview-border); color: var(--cs-state-underReview-fg); }}
.cs-chip-approved {{ background: var(--cs-state-approved-bg); border-color: var(--cs-state-approved-border); color: var(--cs-state-approved-fg); }}
.cs-chip-executed {{ background: var(--cs-state-executed-bg); border-color: var(--cs-state-executed-border); color: var(--cs-state-executed-fg); }}
.cs-chip-failed {{ background: var(--cs-state-failed-bg); border-color: var(--cs-state-failed-border); color: var(--cs-state-failed-fg); }}
.cs-chip-policyBlocked {{ background: var(--cs-state-policyBlocked-bg); border-color: var(--cs-state-policyBlocked-border); color: var(--cs-state-policyBlocked-fg); }}
.cs-detail-orientation {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  margin-bottom: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: center;
}}
.cs-detail-context {{
  display: grid;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-detail-path {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-detail-path a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-detail-path span[aria-hidden="true"] {{ color: var(--cs-color-text-muted); }}
.cs-detail-summary {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-detail-summary-head {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-detail-current {{
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
  overflow-wrap: anywhere;
}}
.cs-detail-summary p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  max-width: 68ch;
  text-wrap: pretty;
}}
.cs-detail-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
}}
.cs-source-text {{
  white-space: pre-wrap;
  word-break: break-word;
  border-radius: var(--cs-radius-md);
  border: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-4);
}}
.cs-detail-grid {{
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr);
  gap: var(--cs-space-2) var(--cs-space-3);
  margin-top: var(--cs-space-4);
}}
.cs-detail-grid dt {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-detail-grid dd {{ margin: 0; color: var(--cs-color-text-secondary); min-width: 0; word-break: break-word; }}
.cs-finding-list {{ display: grid; gap: var(--cs-space-3); margin: 0; padding: 0; list-style: none; }}
.cs-finding {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-finding-head {{
  display: flex;
  justify-content: space-between;
  gap: var(--cs-space-3);
  align-items: flex-start;
}}
.cs-finding-index {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
}}
.cs-citation-rail {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-citation-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 82%, var(--cs-color-primary-50));
  overflow: hidden;
}}
.cs-citation-card summary {{
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
  padding: var(--cs-space-3);
}}
.cs-citation-card summary::-webkit-details-marker {{ display: none; }}
.cs-citation-title {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-citation-title strong {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.cs-citation-action {{
  color: var(--cs-color-primary-700);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-semibold);
  flex: 0 0 auto;
}}
.cs-citation-body {{
  border-top: 1px solid var(--cs-color-border-default);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-citation-snippet {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  text-wrap: pretty;
}}
.cs-citation-meta {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-citation-meta div {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  min-width: 0;
}}
.cs-citation-meta strong {{
  display: block;
  overflow-wrap: anywhere;
}}
.cs-citation-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-brief-hero {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
  margin-bottom: var(--cs-space-5);
}}
.cs-brief-title {{ display: grid; gap: var(--cs-space-2); }}
.cs-brief-title h1 {{
  margin: 0;
  font-size: var(--cs-typography-pageTitle-fontSize);
  line-height: var(--cs-typography-pageTitle-lineHeight);
}}
.cs-brief-meta {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2); color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-brief-actions {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--cs-space-2);
}}
.cs-brief-hero.is-stacked {{
  grid-template-columns: 1fr;
  gap: var(--cs-space-3);
}}
.cs-brief-hero.is-stacked .cs-brief-actions {{ justify-content: flex-start; }}
.cs-brief-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
  gap: var(--cs-space-6);
  align-items: start;
}}
.cs-brief-workbench > *, .cs-brief-workbench .cs-stack, .cs-brief-titlebar, .cs-brief-titlebar > *, .cs-brief-heading-row {{
  min-width: 0;
  max-width: 100%;
}}
.cs-brief-titlebar {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-2);
  align-items: start;
  padding-bottom: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-3);
}}
.cs-brief-titlebar .cs-brief-actions {{
  justify-content: flex-start;
}}
.cs-brief-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  min-width: 0;
}}
.cs-brief-breadcrumb a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-brief-breadcrumb span:last-child {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.cs-brief-heading-row {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
}}
.cs-brief-titlebar h1 {{
  margin: 0;
  flex: 1 1 24rem;
  min-width: 0;
  max-width: 58ch;
  font-size: 28px;
  line-height: 1.16;
  text-wrap: balance;
  overflow-wrap: anywhere;
}}
.cs-brief-titlebar p {{
  max-width: 68ch;
}}
.cs-brief-fact-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-brief-fact {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-brief-fact strong {{ font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-brief-answer-panel {{
  border: 1px solid var(--cs-color-primary-100);
  border-radius: var(--cs-radius-lg);
  background: color-mix(in srgb, var(--cs-color-primary-50) 48%, var(--cs-color-surface-primary));
  padding: clamp(var(--cs-space-4), 3vw, var(--cs-space-6));
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-brief-answer-head {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
}}
.cs-brief-answer-head h2 {{ margin: var(--cs-space-1) 0 0; font-size: 20px; line-height: 1.3; }}
.cs-brief-answer-text {{
  margin: 0;
  max-width: 72ch;
  color: var(--cs-color-text-primary);
  font-size: 18px;
  line-height: 1.65;
  text-wrap: pretty;
}}
.cs-brief-answer-meta {{ display: flex; flex-wrap: wrap; gap: var(--cs-space-2) var(--cs-space-4); color: var(--cs-color-text-secondary); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-brief-answer-meta strong {{ color: var(--cs-color-text-primary); }}
.cs-claim-statement {{ display: grid; gap: var(--cs-space-3); }}
.cs-claim-statement h2 {{
  margin: 0;
  max-width: 42ch;
  font-size: 24px;
  line-height: 1.4;
  text-wrap: pretty;
}}
.cs-claim-rationale {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-claim-rationale p {{ margin: 0; color: var(--cs-color-text-secondary); }}
.cs-decision-panel {{ display: grid; gap: var(--cs-space-3); }}
.cs-confirm-dialog {{
  width: min(520px, calc(100vw - var(--cs-space-6)));
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  color: var(--cs-color-text-primary);
  box-shadow: var(--cs-shadow-popover);
  padding: 0;
}}
.cs-confirm-dialog::backdrop {{ background: rgba(15, 23, 42, .38); }}
.cs-confirm-dialog form {{ padding: var(--cs-space-5); display: grid; gap: var(--cs-space-5); }}
.cs-confirm-dialog-copy {{ display: grid; gap: var(--cs-space-2); }}
.cs-confirm-dialog-copy h2, .cs-confirm-dialog-copy p {{ margin: 0; }}
.cs-confirm-dialog-actions {{ display: flex; justify-content: flex-end; flex-wrap: wrap; gap: var(--cs-space-2); }}
.cs-action-preview, .cs-action-outcome {{ display: grid; gap: var(--cs-space-3); }}
.cs-action-preview h2, .cs-action-outcome h2 {{ margin: 0; }}
.cs-action-change-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--cs-space-3); }}
.cs-action-change {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-action-change p {{ margin: 0; color: var(--cs-color-text-secondary); overflow-wrap: anywhere; }}
.cs-brief-receipt-panel {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--cs-color-primary-50) 50%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary) 58%);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-3);
}}
.cs-brief-lead-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(260px, .75fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-brief-answer-card, .cs-brief-receipt-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 88%, white);
  padding: var(--cs-space-3);
  display: grid;
  align-content: start;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-brief-answer-card.is-primary {{
  min-height: 100%;
  border-color: var(--cs-color-primary-100);
  background: var(--cs-color-surface-primary);
}}
.cs-brief-answer-card p, .cs-brief-receipt-card p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  line-height: 1.55;
  text-wrap: pretty;
}}
.cs-brief-answer-card p {{
  color: var(--cs-color-text-primary);
  font-size: 16px;
  line-height: 1.65;
}}
.cs-brief-receipt-stack {{
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-brief-receipt-card strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
  overflow-wrap: anywhere;
}}
.cs-summary-card {{
  background: color-mix(in srgb, var(--cs-color-primary-50) 48%, var(--cs-color-surface-primary));
  border-color: var(--cs-color-primary-100);
}}
.cs-summary-card p {{
  margin: 0;
  font-size: 16px;
  line-height: 1.7;
  color: var(--cs-color-text-primary);
}}
.cs-brief-note-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: var(--cs-space-4);
}}
.cs-brief-note-list {{
  margin: 0;
  padding-left: var(--cs-space-5);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-stat-list {{ display: grid; gap: var(--cs-space-3); }}
.cs-stat-row {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
}}
.cs-stat-icon {{
  width: 34px;
  height: 34px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-source-card summary {{
  cursor: pointer;
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-source-card[open] summary {{ margin-bottom: var(--cs-space-3); }}
.cs-provenance {{
  border-top: 1px solid var(--cs-color-border-default);
  margin-top: var(--cs-space-3);
  padding-top: var(--cs-space-3);
}}
.cs-trust-ladder {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin: var(--cs-space-4) 0;
}}
.cs-trust-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-trust-step strong {{ display: flex; align-items: center; gap: var(--cs-space-2); }}
.cs-trust-step strong::before {{
  content: "";
  width: 10px;
  height: 10px;
  border-radius: var(--cs-radius-pill);
  border: 2px solid var(--cs-color-border-strong);
  background: var(--cs-color-surface-primary);
}}
.cs-trust-step.is-active {{
  border-color: var(--cs-state-evidenceBacked-border);
  background: var(--cs-state-evidenceBacked-bg);
}}
.cs-trust-step.is-active strong::before {{ border-color: var(--cs-state-evidenceBacked-fg); background: var(--cs-state-evidenceBacked-fg); }}
.cs-trust-step.is-locked {{ opacity: .76; }}
.cs-claim-workbench {{
  grid-template-columns: minmax(0, 1fr) minmax(340px, 400px);
}}
.cs-claim-workbench, .cs-claim-workbench > *, .cs-claim-workbench .cs-stack, .cs-claim-hero, .cs-claim-hero > *, .cs-claim-titlebar, .cs-claim-titlebar > *, .cs-claim-heading-row {{
  min-width: 0;
  max-width: 100%;
}}
.cs-claim-hero, .cs-claim-titlebar, .cs-claim-titlebar > *, .cs-claim-actions {{
  width: 100%;
}}
.cs-claim-workbench .cs-stack, .cs-claim-hero, .cs-claim-titlebar, .cs-claim-titlebar .cs-brief-title {{
  grid-template-columns: minmax(0, 1fr);
}}
.cs-claim-hero {{
  display: grid;
  gap: var(--cs-space-4);
  margin-bottom: var(--cs-space-4);
}}
.cs-claim-hero.is-compact {{
  padding-bottom: var(--cs-space-4);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-claim-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  min-width: 0;
}}
.cs-claim-breadcrumb a {{ color: var(--cs-color-primary-700); font-weight: var(--cs-typography-weight-semibold); }}
.cs-claim-breadcrumb span:last-child {{
  flex: 1 1 180px;
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
}}
.cs-claim-titlebar {{
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-claim-heading-row {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-3);
  flex-wrap: wrap;
}}
.cs-claim-titlebar h1 {{
  margin: 0;
  flex: 1 1 22rem;
  min-width: 0;
  max-width: 44ch;
  font-size: 28px;
  line-height: 1.16;
  text-wrap: balance;
  overflow-wrap: anywhere;
}}
.cs-claim-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
  width: 100%;
  max-width: 100%;
  min-width: 0;
}}
.cs-button.is-disabled {{
  cursor: not-allowed;
  opacity: .68;
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-text-muted);
  border-color: var(--cs-color-border-default);
  box-shadow: none;
}}
.cs-button.is-disabled:hover {{ transform: none; box-shadow: none; }}
.cs-claim-progress {{
  position: relative;
  border: 0;
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3) var(--cs-space-4);
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-claim-progress::before {{
  content: "";
  position: absolute;
  left: var(--cs-space-8);
  right: var(--cs-space-8);
  top: 24px;
  border-top: 1px dashed var(--cs-color-border-strong);
}}
.cs-claim-progress-step {{
  position: relative;
  z-index: 1;
  display: grid;
  justify-items: center;
  gap: var(--cs-space-2);
  text-align: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-claim-dot {{
  width: 16px;
  height: 16px;
  border-radius: var(--cs-radius-pill);
  border: 2px solid var(--cs-color-border-strong);
  background: var(--cs-color-surface-primary);
}}
.cs-claim-progress-step.is-active {{ color: var(--cs-color-text-primary); font-weight: var(--cs-typography-weight-semibold); }}
.cs-claim-progress-step.is-active .cs-claim-dot {{
  border-color: var(--cs-color-primary-600);
  background: var(--cs-color-primary-600);
  box-shadow: 0 0 0 4px var(--cs-color-primary-100);
}}
.cs-claim-pathbar {{
  display: grid;
  grid-template-columns: minmax(170px, .32fr) minmax(0, 1fr);
  gap: var(--cs-space-4);
  align-items: center;
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-claim-pathbar-title {{
  display: grid;
  gap: var(--cs-space-1);
  min-width: 0;
}}
.cs-claim-pathbar-title strong {{
  color: var(--cs-color-text-primary);
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
}}
.cs-claim-pathbar .cs-claim-progress {{
  background: transparent;
  border-radius: 0;
  padding: var(--cs-space-1) 0;
}}
.cs-claim-pathbar .cs-claim-progress::before {{
  left: 8%;
  right: 8%;
  top: 14px;
}}
.cs-claim-pathbar .cs-claim-progress-step {{
  gap: var(--cs-space-1);
}}
.cs-claim-pathbar .cs-claim-dot {{
  width: 14px;
  height: 14px;
}}
.cs-claim-tabs {{
  display: flex;
  gap: var(--cs-space-5);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin-bottom: var(--cs-space-4);
  overflow-x: auto;
}}
.cs-claim-tab {{
  padding: 0 0 var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
  white-space: nowrap;
}}
.cs-claim-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-bottom: 2px solid var(--cs-color-primary-600);
}}
.cs-claim-form-card {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-claim-review-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-top: var(--cs-space-3);
}}
.cs-claim-review-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-claim-review-card strong {{
  font-size: var(--cs-typography-body-fontSize);
  line-height: var(--cs-typography-body-lineHeight);
}}
.cs-claim-field {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-claim-field.is-primary {{
  border-color: var(--cs-color-primary-100);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--cs-color-primary-500) 18%, transparent);
}}
.cs-claim-field-head {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-claim-text {{
  margin: 0;
  color: var(--cs-color-text-primary);
  font-size: 15px;
  line-height: 1.7;
}}
.cs-claim-text.is-statement {{
  font-size: 16px;
  line-height: 1.75;
}}
.cs-claim-field-foot {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  text-align: right;
}}
.cs-claim-taxonomy {{
  display: grid;
  grid-template-columns: minmax(180px, .42fr) minmax(0, 1fr);
  gap: var(--cs-space-3);
}}
.cs-claim-frameworks {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  min-height: 42px;
  padding: var(--cs-space-2) var(--cs-space-3);
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-claim-select, .cs-claim-tags {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  min-height: 42px;
  padding: var(--cs-space-2) var(--cs-space-3);
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
}}
.cs-claim-footrail {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-claim-footrail div {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-claim-footrail strong {{
  color: var(--cs-color-text-primary);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-claim-save-note {{
  display: inline-flex;
  align-items: center;
  gap: var(--cs-space-2);
  color: var(--cs-state-evidenceBacked-text);
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-medium);
}}
.cs-claim-save-note::before {{
  content: "";
  width: 8px;
  height: 8px;
  border-radius: var(--cs-radius-pill);
  background: var(--cs-state-evidenceBacked-fg);
}}
.cs-claim-control-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-claim-control-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-claim-control-mark {{
  width: 26px;
  height: 26px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-surface-subtle);
  color: var(--cs-color-text-secondary);
  border: 1px solid var(--cs-color-border-default);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-claim-control-row.is-ready .cs-claim-control-mark {{
  background: var(--cs-state-evidenceBacked-bg);
  color: var(--cs-state-evidenceBacked-fg);
  border-color: var(--cs-state-evidenceBacked-border);
}}
.cs-claim-control-row.is-review .cs-claim-control-mark {{
  background: var(--cs-state-underReview-bg);
  color: var(--cs-state-underReview-fg);
  border-color: var(--cs-state-underReview-border);
}}
.cs-claim-control-row strong, .cs-claim-control-row p {{ margin: 0; }}
.cs-form-surface {{
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-field-block {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-field-block p {{ margin: 0; }}
.cs-evidence-picker {{ display: grid; gap: var(--cs-space-3); }}
.cs-evidence-toolbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-2);
  flex-wrap: wrap;
  margin-bottom: var(--cs-space-3);
}}
.cs-evidence-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-evidence-row.is-selected {{
  background: color-mix(in srgb, var(--cs-state-underReview-bg) 42%, var(--cs-color-surface-primary));
  border-color: var(--cs-state-underReview-border);
}}
.cs-checkmark {{
  width: 20px;
  height: 20px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  border: 1px solid var(--cs-color-primary-600);
  background: var(--cs-color-primary-600);
  color: var(--cs-color-text-inverse);
  font-size: 12px;
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-review-box {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-action-summary {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-action-workbench {{
  grid-template-columns: minmax(0, 1fr) minmax(340px, 400px);
}}
.cs-action-hero {{
  display: grid;
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-3);
  padding-bottom: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-action-breadcrumb {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  align-items: center;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-action-breadcrumb a {{
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-action-titlebar {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-action-titlebar h1 {{
  margin: 0;
  max-width: 44ch;
  font-size: 28px;
  line-height: 1.14;
  text-wrap: balance;
}}
.cs-action-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
}}
.cs-action-rail {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-action-review-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin: var(--cs-space-3) 0 0;
}}
.cs-action-review-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-action-review-card strong {{ font-size: var(--cs-typography-body-fontSize); line-height: var(--cs-typography-body-lineHeight); }}
.cs-action-receipt-panel {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--cs-color-primary-50) 46%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary) 62%);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-action-receipt-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-action-receipt-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 90%, white);
  padding: var(--cs-space-3);
  display: grid;
  align-content: start;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-action-receipt-card strong {{
  color: var(--cs-color-text-primary);
  line-height: 1.4;
  overflow-wrap: anywhere;
}}
.cs-action-receipt-card p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  line-height: 1.55;
  text-wrap: pretty;
}}
.cs-action-mini-diff {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  overflow: hidden;
  display: grid;
}}
.cs-action-mini-diff div {{
  display: grid;
  gap: var(--cs-space-1);
  padding: var(--cs-space-2);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-action-mini-diff div:last-child {{ border-bottom: 0; }}
.cs-action-route-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-top: var(--cs-space-3);
}}
.cs-action-route-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-action-route-step.is-current {{
  border-color: var(--cs-color-border-focus);
  background: linear-gradient(180deg, var(--cs-color-primary-50), var(--cs-color-surface-primary));
  box-shadow: inset 3px 0 0 var(--cs-color-primary-600);
}}
.cs-action-route-top {{
  display: flex;
  gap: var(--cs-space-2);
  align-items: center;
}}
.cs-action-route-index {{
  width: 24px;
  height: 24px;
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  display: grid;
  place-items: center;
  font-size: var(--cs-typography-metadata-fontSize);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-action-route-step p {{
  margin: 0;
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: 1.35;
}}
.cs-owner-overview {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0;
  margin-bottom: var(--cs-space-5);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
}}
.cs-owner-tabs {{
  display: flex;
  gap: var(--cs-space-5);
  border-bottom: 1px solid var(--cs-color-border-default);
  margin: var(--cs-space-4) 0 var(--cs-space-5);
  overflow-x: auto;
}}
.cs-owner-tab {{
  padding: 0 0 var(--cs-space-3);
  color: var(--cs-color-text-secondary);
  font-weight: var(--cs-typography-weight-semibold);
  white-space: nowrap;
}}
.cs-owner-tab.is-active {{
  color: var(--cs-color-primary-700);
  border-bottom: 2px solid var(--cs-color-primary-600);
}}
.cs-owner-metric {{
  border-right: 1px solid var(--cs-color-border-default);
  background: linear-gradient(180deg, var(--cs-color-surface-primary), var(--cs-color-surface-subtle));
  padding: var(--cs-space-3) var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-owner-metric:last-child {{ border-right: 0; }}
.cs-owner-metric strong {{
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-reference-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-4);
}}
.cs-reference-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  overflow: hidden;
  display: grid;
  min-width: 0;
}}
.cs-reference-card img {{
  width: 100%;
  aspect-ratio: 16 / 10;
  object-fit: cover;
  object-position: top left;
  border-bottom: 1px solid var(--cs-color-border-default);
  background: var(--cs-color-surface-subtle);
}}
.cs-reference-body {{
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-reference-body h2 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-reference-body p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
}}
.cs-connector-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
  gap: var(--cs-space-4);
  align-items: start;
}}
.cs-owner-main-stack, .cs-connector-list, .cs-admin-stack, .cs-policy-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-admin-stack {{
  position: sticky;
  top: calc(var(--cs-space-4) + 72px);
}}
.cs-connector-table-head {{
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(0, .7fr) minmax(0, .8fr) minmax(0, .9fr) auto;
  gap: var(--cs-space-3);
  padding: 0 var(--cs-space-3) var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-connector-card {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(0, .7fr) minmax(0, .8fr) minmax(0, .9fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-connector-card h3 {{ margin: 0 0 var(--cs-space-1); font-size: var(--cs-typography-body-fontSize); }}
.cs-connector-card p {{ margin: 0; color: var(--cs-color-text-secondary); font-size: var(--cs-typography-metadata-fontSize); line-height: 1.45; }}
.cs-connector-source {{
  display: grid;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-connector-title {{
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-connector-icon {{
  width: 34px;
  height: 34px;
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-bold);
  font-size: var(--cs-typography-metadata-fontSize);
  flex: 0 0 auto;
}}
.cs-connector-cell {{
  display: grid;
  gap: var(--cs-space-1);
  min-width: 0;
}}
.cs-connector-cell span, .cs-connector-cell strong {{
  display: block;
  min-width: 0;
  overflow-wrap: anywhere;
}}
.cs-policy-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-policy-row strong, .cs-policy-row p {{
  margin: 0;
}}
.cs-policy-row p {{
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: 1.5;
}}
.cs-owner-scope-table {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  overflow: hidden;
}}
.cs-owner-scope-row {{
  display: grid;
  grid-template-columns: minmax(150px, .45fr) minmax(0, 1fr);
  gap: var(--cs-space-3);
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-owner-scope-row:last-child {{ border-bottom: 0; }}
.cs-owner-scope-row strong, .cs-owner-scope-row span {{ min-width: 0; word-break: break-word; }}
.cs-admin-note {{
  border: 1px solid var(--cs-state-underReview-border);
  border-radius: var(--cs-radius-md);
  background: var(--cs-state-underReview-bg);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-action-metric {{
  display: grid;
  gap: var(--cs-space-1);
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-3);
  background: var(--cs-color-surface-subtle);
}}
.cs-action-object-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--cs-space-3);
}}
.cs-action-object-icon {{
  width: 30px;
  height: 30px;
  border-radius: var(--cs-radius-sm);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-diff-view {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  overflow: hidden;
  background: var(--cs-color-surface-primary);
}}
.cs-action-preview-frame {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-action-preview-meta {{
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-diff-line {{
  display: grid;
  grid-template-columns: 84px minmax(0, 1fr);
  gap: var(--cs-space-3);
  padding: var(--cs-space-3);
  border-bottom: 1px solid var(--cs-color-border-default);
}}
.cs-diff-line:last-child {{ border-bottom: 0; }}
.cs-diff-line.before {{ background: color-mix(in srgb, var(--cs-state-failed-bg) 58%, var(--cs-color-surface-primary)); }}
.cs-diff-line.after {{ background: color-mix(in srgb, var(--cs-state-evidenceBacked-bg) 62%, var(--cs-color-surface-primary)); }}
.cs-call-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: center;
}}
.cs-call-facts {{
  margin-top: var(--cs-space-3);
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-call-fact {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-sm);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-call-fact strong {{ font-size: var(--cs-typography-metadata-fontSize); }}
.cs-call-fact span {{ color: var(--cs-color-text-muted); font-size: var(--cs-typography-metadata-fontSize); }}
.cs-approval-note {{
  border: 1px solid var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-policy-card {{
  border: 1px solid var(--cs-state-underReview-border);
  background: var(--cs-state-underReview-bg);
  border-radius: var(--cs-radius-md);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-2);
}}
.cs-policy-checks {{
  display: grid;
  gap: var(--cs-space-2);
  margin-top: var(--cs-space-3);
}}
.cs-policy-check {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-3);
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-policy-check-mark {{
  width: 24px;
  height: 24px;
  border-radius: var(--cs-radius-sm);
  background: var(--cs-state-underReview-bg);
  color: var(--cs-state-underReview-fg);
  border: 1px solid var(--cs-state-underReview-border);
  display: grid;
  place-items: center;
  font-weight: var(--cs-typography-weight-bold);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-timeline {{ display: grid; gap: var(--cs-space-3); }}
.cs-timeline-item {{ display: grid; grid-template-columns: 16px minmax(0, 1fr); gap: var(--cs-space-3); }}
.cs-dot {{ width: 10px; height: 10px; margin-top: 7px; border-radius: var(--cs-radius-pill); background: var(--cs-color-evidence-600); }}
.cs-audit-workbench {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
  gap: var(--cs-space-5);
  align-items: start;
}}
.cs-audit-hero {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, var(--cs-color-surface-primary), var(--cs-color-surface-subtle));
  padding: var(--cs-space-6);
  margin-bottom: var(--cs-space-4);
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: var(--cs-space-5);
  align-items: end;
}}
.cs-audit-hero h1 {{
  margin: 0;
  font-size: 34px;
  line-height: 1.12;
  text-wrap: balance;
}}
.cs-audit-hero p {{
  max-width: 72ch;
  text-wrap: pretty;
}}
.cs-audit-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  justify-content: flex-end;
}}
.cs-audit-status-strip {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-3);
  margin-bottom: var(--cs-space-4);
}}
.cs-audit-overview {{
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(360px, .95fr);
  gap: var(--cs-space-4);
  margin-bottom: var(--cs-space-4);
  align-items: stretch;
}}
.cs-audit-latest {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-lg);
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--cs-color-primary-50) 44%, var(--cs-color-surface-primary)), var(--cs-color-surface-primary) 62%);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
  min-width: 0;
}}
.cs-audit-latest h2 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-audit-latest-title {{
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-audit-latest-title p {{
  margin: var(--cs-space-1) 0 0;
  color: var(--cs-color-text-secondary);
  text-wrap: pretty;
}}
.cs-audit-latest-actions {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
}}
.cs-audit-overview-side {{
  display: grid;
  gap: var(--cs-space-3);
  min-width: 0;
}}
.cs-audit-summary {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-audit-receipt {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-2);
  display: grid;
  gap: 2px;
}}
.cs-audit-receipt strong {{
  font-size: 18px;
  line-height: 1.15;
  font-variant-numeric: tabular-nums;
}}
.cs-audit-lifecycle {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--cs-space-2);
}}
.cs-audit-lane {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: color-mix(in srgb, var(--cs-color-surface-primary) 78%, var(--cs-color-surface-subtle));
  padding: var(--cs-space-2);
  display: grid;
  gap: var(--cs-space-2);
  min-width: 0;
}}
.cs-audit-lane-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-audit-lane-count {{
  font-variant-numeric: tabular-nums;
  font-weight: var(--cs-typography-weight-bold);
  color: var(--cs-color-primary-700);
}}
.cs-audit-lane p {{
  margin: 0;
  color: var(--cs-color-text-secondary);
  font-size: var(--cs-typography-metadata-fontSize);
  line-height: var(--cs-typography-metadata-lineHeight);
}}
.cs-audit-list {{
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-audit-list-panel {{
  scroll-margin-top: 92px;
}}
.cs-audit-row {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-4);
  display: grid;
  gap: var(--cs-space-3);
}}
.cs-audit-row:hover {{
  border-color: var(--cs-color-border-strong);
  box-shadow: var(--cs-shadow-sm);
}}
.cs-audit-row-main {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: var(--cs-space-3);
  align-items: start;
}}
.cs-audit-row-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--cs-space-3);
}}
.cs-audit-row-position {{
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}
.cs-audit-icon {{
  width: 36px;
  height: 36px;
  border-radius: var(--cs-radius-md);
  display: grid;
  place-items: center;
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-bold);
}}
.cs-audit-row h2 {{
  margin: 0;
  font-size: var(--cs-typography-sectionTitle-fontSize);
  line-height: var(--cs-typography-sectionTitle-lineHeight);
}}
.cs-audit-row-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: var(--cs-space-2);
  margin-top: var(--cs-space-2);
  color: var(--cs-color-text-muted);
  font-size: var(--cs-typography-metadata-fontSize);
}}
.cs-audit-row-note {{
  margin: var(--cs-space-2) 0 0;
  max-width: 64ch;
  color: var(--cs-color-text-secondary);
}}
.cs-audit-side-list {{
  display: grid;
  gap: var(--cs-space-3);
  margin-top: var(--cs-space-3);
}}
.cs-audit-side-item {{
  border-left: 1px solid var(--cs-color-border-default);
  padding-left: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-audit-detail {{
  border-top: 1px solid var(--cs-color-border-default);
  padding-top: var(--cs-space-3);
}}
.cs-audit-detail summary {{
  cursor: pointer;
  color: var(--cs-color-primary-700);
  font-weight: var(--cs-typography-weight-semibold);
}}
.cs-audit-raw-grid {{
  margin-top: var(--cs-space-3);
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-audit-raw-item {{
  border-left: 1px solid var(--cs-color-border-default);
  padding-left: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-audit-empty {{
  border: 1px dashed var(--cs-color-border-strong);
  border-radius: var(--cs-radius-lg);
  background: var(--cs-color-surface-primary);
  padding: var(--cs-space-6);
  display: grid;
  gap: var(--cs-space-4);
}}
.cs-audit-empty-steps {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--cs-space-3);
}}
.cs-audit-empty-step {{
  border: 1px solid var(--cs-color-border-default);
  border-radius: var(--cs-radius-md);
  background: var(--cs-color-surface-subtle);
  padding: var(--cs-space-3);
  display: grid;
  gap: var(--cs-space-1);
}}
.cs-status {{
  min-height: 34px;
  border: 1px solid var(--cs-color-border-default);
  border-left-width: 3px;
  border-radius: var(--cs-radius-md);
  color: var(--cs-color-text-secondary);
  padding: var(--cs-space-2) var(--cs-space-3);
  background: var(--cs-color-surface-primary);
  display: flex;
  align-items: center;
  gap: var(--cs-space-2);
}}
.cs-status::before {{
  content: "";
  width: 8px;
  height: 8px;
  border-radius: var(--cs-radius-pill);
  background: currentColor;
  flex: 0 0 auto;
}}
.cs-status.is-idle {{
  border-left-color: var(--cs-color-primary-600);
  background: var(--cs-color-primary-50);
  color: var(--cs-color-primary-700);
}}
.cs-status.is-loading {{
  border-left-color: var(--cs-state-underReview-fg);
  background: var(--cs-state-underReview-bg);
  color: var(--cs-state-underReview-fg);
}}
.cs-status.is-success {{
  border-left-color: var(--cs-state-evidenceBacked-fg);
  background: var(--cs-state-evidenceBacked-bg);
  color: var(--cs-state-evidenceBacked-fg);
}}
.cs-status.is-error {{
  border-left-color: var(--cs-state-failed-fg);
  background: var(--cs-state-failed-bg);
  color: var(--cs-state-failed-fg);
}}
.cs-status[hidden] {{ display: none; }}
.cs-button:disabled {{
  cursor: progress;
  opacity: .72;
  transform: none;
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
  }}
}}
@media (max-width: 1180px) and (min-width: 981px) {{
  .cs-shell {{ grid-template-columns: 220px minmax(0, 1fr); }}
  .cs-content {{ padding: var(--cs-space-5); }}
  .cs-topbar {{ padding-inline: var(--cs-space-5); }}
  .cs-home-layout.has-activity {{ grid-template-columns: minmax(0, 1fr) minmax(240px, 290px); gap: var(--cs-space-4); }}
  .cs-topbar-workspace {{ display: none; }}
}}
@media (max-width: 1360px) and (min-width: 981px) {{
  .cs-home-layout.has-activity {{ grid-template-columns: 1fr; }}
  .cs-home-activity {{ position: static; }}
}}
.cs-pagination {{
  margin-top: var(--cs-space-4);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--cs-space-3);
  color: var(--cs-color-text-secondary);
}}
.cs-pagination > span:not(.cs-button) {{ font-variant-numeric: tabular-nums; }}
.cs-pagination .cs-button[aria-disabled="true"] {{ opacity: .5; cursor: default; pointer-events: none; }}
.cs-search-receipt, .cs-access-receipt {{ margin-top: var(--cs-space-5); }}
@media (max-width: 980px) {{
  .cs-shell {{ grid-template-columns: 1fr; padding-bottom: calc(68px + env(safe-area-inset-bottom)); }}
  .cs-main {{ order: 1; display: flex; flex-direction: column; }}
  .cs-sidebar {{
    order: 3;
    position: static;
    height: 0;
    border-right: 0;
    border-bottom: 0;
    padding: 0;
    overflow: visible;
  }}
  .cs-sidebar > .cs-brand, .cs-nav > .cs-workspace-switcher {{ display: none !important; }}
  .cs-nav {{
    position: fixed;
    inset: auto 0 0;
    z-index: 5;
    display: block;
    border-top: 1px solid var(--cs-color-border-default);
    background: color-mix(in srgb, var(--cs-color-surface-primary) 96%, transparent);
    backdrop-filter: blur(12px);
    padding: var(--cs-space-1) var(--cs-space-2) calc(var(--cs-space-1) + env(safe-area-inset-bottom));
  }}
  .cs-nav-group {{ grid-template-columns: repeat(5, minmax(0, 1fr)); gap: var(--cs-space-1); }}
  .cs-nav-label {{ display: none; }}
  .cs-nav a {{
    min-height: 58px;
    grid-template-columns: 1fr;
    justify-items: center;
    align-content: center;
    gap: 0;
    padding: var(--cs-space-1);
    text-align: center;
    font-size: var(--cs-typography-metadata-fontSize);
  }}
  .cs-nav a[aria-current="page"] {{ box-shadow: inset 0 3px 0 var(--cs-color-primary-600); }}
  .cs-nav-mark {{ width: 22px; height: 22px; }}
  .cs-nav-count {{ display: none; }}
  .cs-topbar {{
    order: 1;
    position: static;
    min-height: 64px;
    padding: var(--cs-space-2) var(--cs-space-3);
    align-items: center;
    flex-direction: row;
    gap: var(--cs-space-2);
  }}
  .cs-command {{
    flex: 1 1 auto;
    max-width: none;
    min-width: 0;
  }}
  .cs-topbar-actions {{ flex: 0 0 auto; justify-content: flex-start; flex-wrap: nowrap; }}
  .cs-topbar-workspace, .cs-avatar {{ display: none; }}
  .cs-review-link span {{ display: none; }}
  .cs-review-link {{ width: 44px; padding: 0; justify-content: center; position: relative; }}
  .cs-review-link strong {{ position: absolute; right: -4px; top: -4px; min-width: 18px; height: 18px; }}
  .cs-search {{ max-width: none; min-height: 48px; flex-basis: auto; padding: var(--cs-space-1) var(--cs-space-2); }}
  .cs-search span[aria-hidden="true"] {{ display: none; }}
  .cs-content {{ order: 2; padding: var(--cs-space-3); }}
  .cs-grid-hero, .cs-grid-two, .cs-module-grid, .cs-detail-orientation, .cs-brief-hero, .cs-brief-workbench, .cs-brief-titlebar, .cs-brief-lead-grid, .cs-search-workbench, .cs-search-command, .cs-artifact-hero, .cs-artifact-workbench, .cs-artifact-compact-hero, .cs-artifact-title-row, .cs-metadata-strip, .cs-metadata-strip.is-artifact, .cs-artifact-inspection-strip, .cs-inbox-workbench, .cs-inbox-lane-summary, .cs-inbox-receipt-strip, .cs-collection-workbench, .cs-collection-summary, .cs-collection-footrail, .cs-queue-lanes, .cs-empty-state-main, .cs-empty-steps, .cs-empty-briefing, .cs-brief-fact-strip, .cs-brief-note-grid, .cs-action-workbench, .cs-action-titlebar, .cs-action-review-strip, .cs-action-receipt-grid, .cs-action-route-strip, .cs-call-facts, .cs-audit-hero, .cs-audit-overview, .cs-audit-workbench, .cs-audit-status-strip, .cs-audit-summary, .cs-audit-lifecycle, .cs-audit-empty-steps, .cs-audit-raw-grid, .cs-owner-overview, .cs-reference-grid, .cs-connector-grid, .cs-connector-card, .cs-policy-row, .cs-owner-scope-row, .cs-claim-workbench, .cs-claim-titlebar, .cs-claim-pathbar, .cs-claim-progress, .cs-claim-review-strip, .cs-claim-taxonomy, .cs-claim-footrail {{ grid-template-columns: 1fr; }}
  .cs-owner-metric {{ border-right: 0; border-bottom: 1px solid var(--cs-color-border-default); }}
  .cs-owner-metric:last-child {{ border-bottom: 0; }}
  .cs-connector-table-head {{ display: none; }}
  .cs-page-head {{ margin-bottom: var(--cs-space-3); }}
  .cs-hero h1 {{ font-size: var(--cs-typography-pageTitle-fontSize); line-height: var(--cs-typography-pageTitle-lineHeight); }}
  .cs-home-intro {{ min-height: auto; }}
  .cs-home-layout, .cs-home-layout.has-activity {{ grid-template-columns: 1fr; gap: var(--cs-space-4); }}
  .cs-home-activity {{ position: static; }}
  .cs-home-canvas {{ padding: var(--cs-space-3); gap: var(--cs-space-2); }}
  .cs-home-canvas > .cs-panel-header {{ gap: var(--cs-space-2); }}
  .cs-home-canvas > .cs-panel-header p {{ display: none; }}
  .cs-home-source-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .cs-home-source-row .cs-button {{ justify-content: center; padding-inline: var(--cs-space-2); }}
  .cs-home-item, .cs-next-step, .cs-home-paste-row {{ grid-template-columns: 1fr; }}
  .cs-home-loop-inline {{ display: none; }}
  .cs-home-source-note {{ display: none; }}
  .cs-drop {{ min-height: auto; padding: var(--cs-space-2); gap: var(--cs-space-2); }}
  .cs-drop-target {{ grid-template-columns: auto minmax(0, 1fr); place-items: center start; text-align: left; }}
  .cs-drop-mark {{ width: 40px; height: 40px; font-size: 15px; }}
  .cs-drop textarea.cs-drop-input {{ min-height: 64px; }}
  .cs-ask-bar {{ grid-template-columns: auto minmax(0, 1fr); gap: var(--cs-space-2); }}
  .cs-ask-bar .cs-field, .cs-ask-bar .cs-button {{ grid-column: 1 / -1; }}
  .cs-suggestion-row {{ grid-template-columns: 1fr; }}
  .cs-inbox-lane-summary {{ align-items: flex-start; }}
  .cs-journey-ref-grid {{ grid-template-columns: 1fr; }}
  .cs-empty-actions {{ flex-direction: column; align-items: stretch; }}
  .cs-empty-actions .cs-button {{ justify-content: center; }}
  .cs-detail-actions {{ justify-content: flex-start; }}
  .cs-brief-actions {{ justify-content: flex-start; }}
  .cs-brief-titlebar h1 {{ font-size: 26px; }}
  .cs-brief-actions {{
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    justify-content: stretch;
  }}
  .cs-brief-fact-strip {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
  .cs-brief-actions .cs-button {{
    justify-content: center;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    text-align: center;
  }}
  .cs-claim-actions {{ justify-content: flex-start; }}
  .cs-action-actions {{ justify-content: flex-start; }}
  .cs-action-rail {{ position: static; }}
  .cs-audit-actions {{ justify-content: flex-start; }}
  .cs-audit-filters {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .cs-admin-stack {{ position: static; }}
  .cs-claim-progress::before {{ display: none; }}
  .cs-trust-ladder, .cs-action-summary, .cs-citation-meta {{ grid-template-columns: 1fr; }}
  .cs-diff-line, .cs-call-row, .cs-result-row, .cs-inbox-head, .cs-inbox-row, .cs-collection-row, .cs-action-object-row, .cs-connector-card, .cs-claim-control-row {{ grid-template-columns: 1fr; }}
  .cs-collection-actions {{ justify-items: start; }}
  .cs-collection-actions .cs-row {{ justify-content: flex-start; }}
  .cs-inbox-head {{ display: none; }}
  .cs-inbox-row {{ grid-template-columns: 34px minmax(0, 1fr); align-items: start; }}
  .cs-inbox-row-state {{ grid-column: 2; justify-content: flex-start; }}
  .cs-inbox-detail {{ scroll-margin-top: var(--cs-space-3); }}
  .cs-artifact-compact-hero .cs-artifact-title h1 {{ font-size: 26px; }}
  .cs-artifact-compact-hero .cs-artifact-actions {{ padding-top: 0; }}
  .cs-artifact-actions {{ justify-content: flex-start; }}
  .cs-artifact-rail {{ position: static; }}
  .cs-artifact-toolbar {{ align-items: stretch; flex-direction: column; }}
  .cs-document-frame.has-rail {{ grid-template-columns: 1fr; min-height: auto; }}
  .cs-artifact-page-rail {{ display: none; }}
  .cs-artifact-page-area {{ padding: var(--cs-space-3); }}
  .cs-search-rail {{ position: static; }}
  .cs-search-titleline {{ align-items: start; flex-direction: column; }}
  .cs-search-mode {{ min-width: 0; width: 100%; }}
  .cs-claim-breadcrumb {{
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr);
    align-items: center;
  }}
  .cs-claim-breadcrumb span:last-child {{
    grid-column: 1 / -1;
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .cs-claim-actions {{
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    justify-content: stretch;
  }}
  .cs-claim-actions .cs-button {{
    justify-content: center;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    text-align: center;
  }}
  .cs-claim-actions .is-disabled {{ grid-column: auto; }}
  .cs-result-support {{ justify-items: start; text-align: left; }}
  .cs-audit-row-main {{ grid-template-columns: auto minmax(0, 1fr); }}
  .cs-audit-row-main .cs-chip {{ justify-self: start; }}
  .cs-audit-row-top {{ align-items: flex-start; flex-direction: column; gap: var(--cs-space-1); }}
  .cs-document-page {{ min-height: auto; }}
  .cs-list-row {{ grid-template-columns: 1fr; }}
  .cs-detail-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 620px) {{
  .cs-content {{ padding: var(--cs-space-3) var(--cs-space-2); }}
  .cs-topbar {{ padding-inline: var(--cs-space-2); }}
  .cs-search input {{ font-size: 16px; }}
  .cs-search button {{ width: 44px; }}
  .cs-home-canvas {{ padding: var(--cs-space-3); }}
  .cs-home-source-row {{ grid-template-columns: 1fr; }}
  .cs-home-source-row .cs-button {{ width: 100%; }}
  .cs-drop-target p {{ display: none; }}
  .cs-hero h1 {{ font-size: 28px; line-height: 1.15; }}
  .cs-action-change-grid {{ grid-template-columns: 1fr; }}
  .cs-confirm-dialog-actions {{ display: grid; grid-template-columns: 1fr; }}
  .cs-confirm-dialog-actions .cs-button {{ width: 100%; }}
  .cs-audit-filters {{ grid-template-columns: 1fr; padding: var(--cs-space-3); }}
  .cs-audit-filters .cs-button {{ width: 100%; }}
}}
"""


@lru_cache(maxsize=8)
def _style_asset_cached(root_value: str) -> tuple[str, bytes, str]:
    css = render_styles(Path(root_value).resolve()).encode("utf-8")
    digest = hashlib.sha256(css).hexdigest()
    return f"cornerstone.{digest[:16]}.css", css, digest


def style_asset(root: Path) -> tuple[str, bytes, str]:
    """Return the immutable content-addressed product stylesheet."""

    return _style_asset_cached(str(root.resolve()))


def _css_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
