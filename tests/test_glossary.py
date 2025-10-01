from __future__ import annotations

from pathlib import Path

import pytest

from cornerstone.glossary import Glossary, GlossaryEntry, load_glossary


def test_load_glossary_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "glossary.yaml"
    yaml_path.write_text(
        """
        - term: API
          definition: Application Programming Interface
          synonyms: [interface]
        - term: SLA
          definition: Service level agreement
        """
    )

    glossary = load_glossary(yaml_path)
    assert len(glossary) == 2

    matches = glossary.top_matches("We need to review the API", limit=2)
    assert matches[0].term == "API"


def test_glossary_matches_synonyms() -> None:
    glossary = Glossary([GlossaryEntry(term="Incident", definition="issue", synonyms=["outage"])])
    matches = glossary.top_matches("We experienced an outage", limit=1)
    assert matches
    assert matches[0].term == "Incident"


def test_glossary_missing_file_returns_empty(tmp_path: Path) -> None:
    glossary = load_glossary(tmp_path / "missing.yaml")
    assert len(glossary) == 0


def test_glossary_prompt_section() -> None:
    glossary = Glossary([GlossaryEntry(term="SLA", definition="agreement")])
    section = glossary.to_prompt_section("What SLA do we have?", limit=1)
    assert "SLA" in section
