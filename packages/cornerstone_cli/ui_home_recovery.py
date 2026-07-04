"""Reference-aligned VS4-H01 Home recovery target.

This module is intentionally side-effect free. It gives the rejected VS4-H01
landing/Home work a concrete, source-verifiable target before the runtime Home
is swapped over. It does not claim VS4-H01 acceptance and does not execute any
external action.
"""

from __future__ import annotations

import html

PRODUCT_NAV = ("Home", "Search", "Artifacts", "Claims", "Actions")

REQUIRED_HOME_REGIONS = (
    "sidebar-nav",
    "top-search",
    "hero",
    "drop-zone",
    "ask-box",
    "suggested-prompts",
    "recent-items",
    "knowledge-state",
    "suggested-next-steps",
    "recent-activity",
)

SAFETY_CHIPS = (
    "Local preview",
    "No external send",
    "Sources preserved",
    "Review before action",
)

FORBIDDEN_FIRST_VIEWPORT_TERMS = (
    "scenario verifier",
    "human-gate",
    "human gate",
    "review packet",
    "package paths",
    "raw command",
    "local_scenario_ready",
    "vs0_runtime_ready",
    "production_release_ready",
    "dense audit id",
    "connector admin",
    "ontology editor",
    "policy editor",
)

FORBIDDEN_READINESS_CLAIMS = (
    "production ready",
    "production readiness",
    "on-prem ready",
    "on-prem readiness",
    "final security accepted",
    "live provider ready",
    "live-provider readiness",
    "human ux accepted",
    "vs5 complete",
    "value verified external",
)

SUGGESTED_PROMPTS = (
    "What changed in this source?",
    "What claims can I defend from this?",
    "What is missing before a decision?",
)

RECENT_ITEMS = (
    ("Vendor renewal notes", "Source saved · searchable"),
    ("Operations review draft", "Brief draft · needs review"),
    ("Follow-up action", "Preview only · approval required"),
)

NEXT_STEPS = (
    "Create a brief from the pasted source.",
    "Open a receipt before promoting a finding.",
    "Request review before any action changes state.",
)

RECENT_ACTIVITY = (
    "Source preserved locally",
    "Search snapshot available",
    "Review queue waiting",
)


def _chip(text: str, state: str = "neutral") -> str:
    return f'<span class="cs-chip" data-hrec-safety-chip="{html.escape(text)}" data-state="{html.escape(state)}">{html.escape(text)}</span>'


def _region(name: str, body: str, *, tag: str = "section", extra_class: str = "") -> str:
    classes = f"cs-card {extra_class}".strip()
    return f'<{tag} class="{classes}" data-hrec-region="{html.escape(name)}">{body}</{tag}>'


def render_vs4_h01_home_recovery() -> str:
    """Return the static recovery target for the Product Alpha Home.

    The returned markup is deliberately small enough for static checks and human
    review. The runtime implementation should preserve these region markers or
    equivalent DOM markers when the hardcoded Product Alpha Home is rebuilt.
    """

    nav = "\n".join(
        f'<a href="#{item.lower().replace(" ", "-")}" data-hrec-nav-item="{html.escape(item)}">{html.escape(item)}</a>'
        for item in PRODUCT_NAV
    )
    safety = "\n".join(_chip(label, "safe") for label in SAFETY_CHIPS)
    prompts = "\n".join(f'<button type="button">{html.escape(prompt)}</button>' for prompt in SUGGESTED_PROMPTS)
    recent_items = "\n".join(
        f"<li><strong>{html.escape(title)}</strong><span>{html.escape(state)}</span></li>"
        for title, state in RECENT_ITEMS
    )
    next_steps = "\n".join(f"<li>{html.escape(step)}</li>" for step in NEXT_STEPS)
    activity = "\n".join(f"<li>{html.escape(item)}</li>" for item in RECENT_ACTIVITY)

    drop_zone = _region(
        "drop-zone",
        """
        <p class="cs-label">Drop</p>
        <h2>Paste, upload, or drag a source.</h2>
        <textarea aria-label="Paste source text" placeholder="Paste messy notes, email fragments, meeting text, or source material..."></textarea>
        <button type="button">Create brief</button>
        """,
        extra_class="cs-primary-card",
    )

    ask_box = _region(
        "ask-box",
        """
        <p class="cs-label">Ask</p>
        <h2>Ask across saved sources.</h2>
        <input aria-label="Ask CornerStone" placeholder="What do my sources say?" />
        <button type="button">Ask with sources</button>
        <p class="cs-muted">If sources do not support an answer, CornerStone should say so.</p>
        """,
        extra_class="cs-primary-card",
    )

    first_view = f"""
    <div class="cs-first-viewport" data-hrec-first-viewport="true">
      <aside class="cs-sidebar" data-hrec-region="sidebar-nav" aria-label="Primary navigation">
        <div class="cs-logo">CornerStone</div>
        <nav>{nav}</nav>
      </aside>
      <main class="cs-main">
        <header class="cs-topbar" data-hrec-region="top-search">
          <input aria-label="Global search" placeholder="Search sources, briefs, claims, and actions..." />
          <div class="cs-safety-row" aria-label="Safety boundary">{safety}</div>
        </header>
        <section class="cs-hero" data-hrec-region="hero">
          <p class="cs-label">Briefs with receipts</p>
          <h1>Drop anything, or ask what we know.</h1>
          <p>CornerStone preserves your sources, builds a brief you can defend, and keeps every important statement tied to receipts.</p>
        </section>
        <div class="cs-primary-grid">
          {drop_zone}
          {ask_box}
        </div>
        {_region("suggested-prompts", '<h2>Suggested prompts</h2><div class="cs-prompt-row">' + prompts + '</div>')}
        <div class="cs-lower-grid">
          {_region("recent-items", '<h2>Recent items</h2><ul>' + recent_items + '</ul>')}
          {_region("knowledge-state", '<h2>Knowledge state</h2><div class="cs-state-list"><span>Saved</span><span>Searchable</span><span>Draft brief</span><span>Needs review</span></div>')}
          {_region("suggested-next-steps", '<h2>Suggested next steps</h2><ol>' + next_steps + '</ol>')}
        </div>
      </main>
      <aside class="cs-activity" data-hrec-region="recent-activity">
        <h2>Recent activity</h2>
        <ul>{activity}</ul>
      </aside>
    </div>
    <span hidden data-hrec-first-viewport-end="true"></span>
    """

    proof = """
    <section class="cs-proof" data-hrec-progressive-proof="review-drawer">
      <details>
        <summary>Review proof and safety detail</summary>
        <p>Evidence, policy, approval, audit, and proof details stay reachable after the user sees the primary Drop / Ask workspace.</p>
        <ul>
          <li>Evidence: source receipts and cited spans are shown before decision promotion.</li>
          <li>Policy: risky or external changes require a policy decision.</li>
          <li>Approval: actions remain preview-only until reviewed.</li>
          <li>Audit: source, brief, decision, and action history remain inspectable.</li>
          <li>Proof: VS4-H01 remains human-owned until a dated retry record exists.</li>
        </ul>
      </details>
    </section>
    """

    return f"""<!doctype html>
<html lang="en" data-hrec-prototype="vs4-h01-home-recovery">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CornerStone Home Recovery Target</title>
  <style>
    :root {{
      color-scheme: light;
      --cs-bg: #f7f9fb;
      --cs-surface: #ffffff;
      --cs-border: #d8e0e7;
      --cs-ink: #1f2933;
      --cs-muted: #5f6f7a;
      --cs-accent: #285e61;
      --cs-safe: #216e4e;
      --cs-review: #8f5f00;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--cs-bg); color: var(--cs-ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .cs-first-viewport {{ min-height: 100vh; display: grid; grid-template-columns: 232px minmax(0, 1fr) 320px; gap: 24px; padding: 24px; }}
    .cs-sidebar, .cs-activity, .cs-card, .cs-topbar, .cs-hero, .cs-proof {{ background: var(--cs-surface); border: 1px solid var(--cs-border); border-radius: 14px; }}
    .cs-sidebar {{ padding: 20px; }}
    .cs-logo {{ font-weight: 800; margin-bottom: 24px; }}
    nav {{ display: grid; gap: 8px; }}
    nav a {{ color: var(--cs-accent); font-weight: 700; text-decoration: none; padding: 8px 10px; border-radius: 8px; }}
    .cs-main {{ display: grid; gap: 18px; align-content: start; }}
    .cs-topbar {{ display: flex; gap: 12px; align-items: center; justify-content: space-between; padding: 14px; }}
    input, textarea {{ width: 100%; border: 1px solid var(--cs-border); border-radius: 10px; padding: 12px; font: inherit; background: #fbfcfd; }}
    textarea {{ min-height: 152px; resize: vertical; }}
    .cs-hero {{ padding: 28px; }}
    .cs-hero h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 4vw, 4rem); letter-spacing: -0.04em; }}
    .cs-hero p {{ max-width: 760px; }}
    .cs-label {{ margin: 0 0 8px; color: var(--cs-accent); font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
    .cs-muted {{ color: var(--cs-muted); }}
    .cs-primary-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }}
    .cs-card {{ padding: 20px; }}
    .cs-primary-card {{ min-height: 280px; display: grid; gap: 12px; align-content: start; }}
    .cs-lower-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }}
    .cs-prompt-row, .cs-safety-row, .cs-state-list {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .cs-chip, .cs-state-list span {{ border: 1px solid var(--cs-border); border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
    .cs-chip[data-state="safe"] {{ color: var(--cs-safe); background: #ecfdf5; border-color: #bbf7d0; }}
    button {{ border: 1px solid var(--cs-accent); background: var(--cs-accent); color: #fff; border-radius: 10px; padding: 10px 14px; font: inherit; font-weight: 800; }}
    .cs-activity {{ padding: 20px; align-self: start; }}
    li {{ margin: 8px 0; }}
    li span {{ display: block; color: var(--cs-muted); margin-top: 2px; }}
    .cs-proof {{ margin: 0 24px 24px; padding: 16px; }}
    .cs-proof summary {{ color: var(--cs-accent); cursor: pointer; font-weight: 800; }}
    @media (max-width: 900px) {{
      .cs-first-viewport {{ grid-template-columns: minmax(0, 1fr); padding: 16px; }}
      .cs-sidebar nav {{ grid-template-columns: repeat(5, minmax(0, auto)); overflow-x: auto; }}
      .cs-topbar, .cs-primary-grid, .cs-lower-grid {{ grid-template-columns: 1fr; display: grid; }}
      .cs-activity {{ order: 4; }}
    }}
  </style>
</head>
<body>
{first_view}
{proof}
</body>
</html>"""


if __name__ == "__main__":
    print(render_vs4_h01_home_recovery())
