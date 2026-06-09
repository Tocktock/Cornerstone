from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalTestProvider:
    """Deterministic local provider for scenario plumbing, never a PASS judge."""

    name: str = "local_test"
    model: str = "local_test.v0"

    def brief_for(self, text: str) -> dict[str, object]:
        return {
            "provider": self.name,
            "model": self.model,
            "summary": "Deterministic fixture brief.",
            "evidence_terms": sorted(set(word.strip(".,:").lower() for word in text.split() if len(word) > 6))[:8],
            "unsupported_assertions": [],
            "assumptions": [],
        }
