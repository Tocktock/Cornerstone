#!/usr/bin/env python3
"""Verify the VS4-H01 Home recovery target.

The check is intentionally static: it prepares AI-verifiable evidence for the
human-owned VS4-H01 retry without claiming human acceptance.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli.ui_home_recovery import (  # noqa: E402
    FORBIDDEN_FIRST_VIEWPORT_TERMS,
    FORBIDDEN_READINESS_CLAIMS,
    REQUIRED_HOME_REGIONS,
    SAFETY_CHIPS,
    render_vs4_h01_home_recovery,
)

DOC_PATH = ROOT / "docs" / "design" / "VS4_H01_HOME_RECOVERY_CONTRACT_2026-07-04.md"


def _between(text: str, start: str, end: str) -> str:
    try:
        start_index = text.index(start)
        end_index = text.index(end, start_index)
    except ValueError:
        return ""
    return text[start_index:end_index]


def main() -> int:
    html = render_vs4_h01_home_recovery()
    first_view = _between(html, 'data-hrec-first-viewport="true"', 'data-hrec-first-viewport-end="true"')
    lower_first_view = first_view.lower()
    lower_html = html.lower()

    region_results = {
        region: f'data-hrec-region="{region}"' in html
        for region in REQUIRED_HOME_REGIONS
    }
    safety_results = {chip: chip in html for chip in SAFETY_CHIPS}
    first_view_forbidden_hits = [term for term in FORBIDDEN_FIRST_VIEWPORT_TERMS if term in lower_first_view]
    readiness_hits = [term for term in FORBIDDEN_READINESS_CLAIMS if term in lower_html]
    proof_terms = ["Evidence", "policy", "approval", "audit", "proof"]
    proof_results = {term: re.search(term, html, flags=re.IGNORECASE) is not None for term in proof_terms}

    doc_text = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
    doc_results = {
        "contract_doc_present": DOC_PATH.exists(),
        "scenario_contract_table": "| HREC-S01 | MUST_PASS |" in doc_text,
        "human_required_gate": "| HREC-H01 | HUMAN_REQUIRED |" in doc_text,
        "no_vs4_h01_pass_claim": "does **not** mark `VS4-H01` as accepted" in doc_text,
    }

    checks = {
        "required_regions_present": all(region_results.values()),
        "first_value_copy_present": "Drop anything, or ask what we know." in html and "Briefs with receipts" in html,
        "safety_chips_present": all(safety_results.values()),
        "first_view_internal_terms_absent": not first_view_forbidden_hits,
        "readiness_claims_absent": not readiness_hits,
        "progressive_proof_present": 'data-hrec-progressive-proof="review-drawer"' in html and all(proof_results.values()),
        "responsive_rule_present": "@media (max-width: 900px)" in html,
        "contract_doc_valid": all(doc_results.values()),
    }
    payload = {
        "schema_version": "cs.vs4_h01_home_recovery_static_verification.v0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "region_results": region_results,
        "safety_results": safety_results,
        "first_view_forbidden_hits": first_view_forbidden_hits,
        "readiness_hits": readiness_hits,
        "proof_results": proof_results,
        "doc_results": doc_results,
        "human_acceptance_claimed": False,
        "vs4_h01_human_gate_status": "HUMAN_REQUIRED",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
