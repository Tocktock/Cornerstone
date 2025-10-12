#!/usr/bin/env python3
"""CLI utility to generate query expansion hints using the configured LLM backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cornerstone.config import Settings
from cornerstone.glossary import load_glossary, load_query_hints
from cornerstone.query_hints import QueryHintGenerator, merge_hint_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate query expansion hints using the project glossary.")
    parser.add_argument(
        "--glossary",
        type=Path,
        default=None,
        help="Path to glossary YAML (defaults to settings glossary path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("glossary/query_hints.generated.yaml"),
        help="Destination YAML file for generated hints",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help="Existing hints file to merge with generated results",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=6,
        help="Maximum glossary entries per LLM prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged hints to stdout instead of writing to disk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    glossary_path = args.glossary or Path(settings.glossary_path)
    glossary = load_glossary(glossary_path)
    generator = QueryHintGenerator(settings, max_terms_per_prompt=args.max_terms)

    report = generator.generate(glossary.entries())
    existing = load_query_hints(args.merge) if args.merge else {}
    merged = merge_hint_sources(existing, report.hints)

    if args.dry_run:
        yaml.safe_dump(merged, stream=sys.stdout, allow_unicode=True, sort_keys=True)
        return 0

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, allow_unicode=True, sort_keys=True)

    print(
        f"Generated {len(report.hints)} hint buckets using backend={report.backend or 'custom'} "
        f"(prompts={report.prompts_sent}) -> {output_path}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
