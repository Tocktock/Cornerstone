#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DATE = "2026-06-23"
DEFAULT_SOURCE_DIR = ROOT / "reports/scenario"
DEFAULT_OUTPUT_DIR = ROOT / "reports/scenario/connector-contract-adapter"
SHARED_SECTION_KEYS = [
    "command_evidence",
    "connector_contract_evidence",
    "negative_evidence",
    "readiness_dimensions",
    "audit_refs",
    "evidence_refs",
    "policy_decision_refs",
]
COMPACT_REPORT_SCHEMA = "cs.connector_contract_adapter.compact_report.v1"
COMPACT_MANIFEST_SCHEMA = "cs.connector_contract_adapter.compact_manifest.v1"
SHARED_EVIDENCE_SCHEMA = "cs.connector_contract_adapter.shared_evidence.v1"
SHARED_EVIDENCE_INDEX_SCHEMA = "cs.connector_contract_adapter.shared_evidence_index.v1"
COMPACT_EVIDENCE_LAYOUT = "content_addressed_objects_v1"
PATH_PORTABILITY_CLAIM_BOUNDARY = (
    "absolute_paths_are_historical_transcript_metadata_not_portable_evidence"
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{_relative(path)} must contain a JSON object")
    return payload


def _load_content_object(ref: dict[str, Any]) -> Any:
    path_value = ref.get("path")
    expected_sha = ref.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("content object ref missing path")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError(f"content object ref missing sha256: {path_value}")
    object_path = ROOT / path_value
    payload = json.loads(object_path.read_text())
    actual_sha = _sha256_payload(payload)
    if actual_sha != expected_sha:
        raise ValueError(f"content object sha256 mismatch for {path_value}: {actual_sha} != {expected_sha}")
    return payload


def _focused_report_scenario_id(path: Path) -> str:
    match = re.fullmatch(
        r"connector-contract-adapter-cs-ch-(?P<number>\d{3})-\d{4}-\d{2}-\d{2}\.json",
        path.name,
    )
    if not match:
        raise ValueError(f"unexpected focused report name: {_relative(path)}")
    return f"CS-CH-{match.group('number')}"


def _scenario_summary(row: dict[str, Any]) -> dict[str, int]:
    status = row.get("status")
    owner = row.get("owner")
    return {
        "scenario_count": 1,
        "pass": 1 if status == "PASS" else 0,
        "fail": 1 if status == "FAIL" else 0,
        "not_verified": 1 if status == "NOT_VERIFIED" else 0,
        "human_required": 1 if owner == "Human" else 0,
        "blocking": 0 if status == "PASS" or owner == "Human" else 1,
    }


def _source_section_refs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for key in SHARED_SECTION_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        count = len(value) if isinstance(value, (dict, list)) else None
        ref: dict[str, Any] = {
            "sha256": _sha256_payload(value),
            "size_bytes": len(_canonical_bytes(value)),
        }
        if count is not None:
            ref["count"] = count
        refs[key] = ref
    return refs


def _source_report_ref(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "output_path": payload.get("output_path"),
        "omitted_duplicate_section_refs": _source_section_refs(payload),
    }


def _path_portability_ref(*, output_path: Path, source_path: Path) -> dict[str, Any]:
    return {
        "portable_report_path": _relative(output_path),
        "portable_source_report_path": _relative(source_path),
        "portable_evidence_root": "reports/scenario/connector-contract-adapter",
        "historical_absolute_path_fields": [
            "output_path",
            "source_report.output_path",
        ],
        "regenerable_transcript_path_prefixes": [
            "tmp/scenario/",
        ],
        "claim_boundary": PATH_PORTABILITY_CLAIM_BOUNDARY,
    }


def _shared_section_refs(shared_sections: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for key, value in shared_sections.items():
        count = len(value) if isinstance(value, (dict, list)) else None
        ref: dict[str, Any] = {
            "sha256": _sha256_payload(value),
            "size_bytes": len(_canonical_bytes(value)),
        }
        if count is not None:
            ref["count"] = count
        refs[key] = ref
    return refs


def _content_object_path(output_dir: Path, sha256: str) -> Path:
    return output_dir / "objects" / "sha256" / sha256[:2] / f"{sha256}.json"


def _content_object_ref(output_dir: Path, payload: Any) -> dict[str, Any]:
    sha256 = _sha256_payload(payload)
    encoded = _canonical_bytes(payload)
    object_path = _content_object_path(output_dir, sha256)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(encoded + b"\n")
    return {
        "path": _relative(object_path),
        "sha256": sha256,
        "size_bytes": len(encoded),
    }


def _section_object_refs(output_dir: Path, section_name: str, payload: Any) -> dict[str, Any]:
    object_ref = _content_object_ref(output_dir, payload)
    section_ref: dict[str, Any] = {
        "sha256": _sha256_payload(payload),
        "size_bytes": len(_canonical_bytes(payload)),
        "object": object_ref,
    }
    if isinstance(payload, list):
        section_ref["type"] = "list"
        section_ref["count"] = len(payload)
        return section_ref
    if isinstance(payload, dict):
        section_ref["type"] = "dict"
        section_ref["count"] = len(payload)
        return section_ref
    raise ValueError(f"shared section {section_name} must be a JSON list or object")


def _shared_evidence_summary(section_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for section_name, section_ref in section_index.items():
        summaries[section_name] = {
            "count": section_ref.get("count"),
            "sha256": section_ref.get("sha256"),
            "size_bytes": section_ref.get("size_bytes"),
            "type": section_ref.get("type"),
        }
    return summaries


def _expand_shared_evidence_index(shared_index: dict[str, Any]) -> dict[str, Any]:
    sections = shared_index.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("shared evidence index missing sections")
    expanded: dict[str, Any] = {}
    for section_name, section_ref in sections.items():
        if not isinstance(section_ref, dict):
            raise ValueError(f"shared evidence section ref must be an object: {section_name}")
        section_type = section_ref.get("type")
        section_object = section_ref.get("object")
        if isinstance(section_object, dict):
            payload = _load_content_object(section_object)
            if section_type == "list" and not isinstance(payload, list):
                raise ValueError(f"shared evidence section expected list: {section_name}")
            if section_type == "dict" and not isinstance(payload, dict):
                raise ValueError(f"shared evidence section expected dict: {section_name}")
            expanded[section_name] = payload
        elif section_type == "list":
            items = section_ref.get("items")
            if not isinstance(items, list):
                raise ValueError(f"shared evidence list section missing items: {section_name}")
            expanded[section_name] = [
                _load_content_object(item)
                for item in sorted(items, key=lambda item: item.get("index") if isinstance(item, dict) else -1)
            ]
        elif section_type == "dict":
            entries = section_ref.get("entries")
            if not isinstance(entries, list):
                raise ValueError(f"shared evidence dict section missing entries: {section_name}")
            expanded[section_name] = {
                entry["key"]: _load_content_object(entry)
                for entry in entries
                if isinstance(entry, dict) and isinstance(entry.get("key"), str)
            }
        else:
            raise ValueError(f"unsupported shared evidence section type: {section_name}={section_type}")
        actual_sha = _sha256_payload(expanded[section_name])
        expected_sha = section_ref.get("sha256")
        if actual_sha != expected_sha:
            raise ValueError(f"shared evidence section sha256 mismatch for {section_name}: {actual_sha} != {expected_sha}")
    return expanded


def _write_shared_evidence_index(
    *,
    output_dir: Path,
    shared_sections: dict[str, Any],
    report_date: str,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    object_root = output_dir / "objects" / "sha256"
    if object_root.exists():
        shutil.rmtree(object_root)
    section_index = {
        section_name: _section_object_refs(output_dir, section_name, payload)
        for section_name, payload in shared_sections.items()
    }
    object_paths = {
        section_ref["object"]["path"]
        for section_ref in section_index.values()
        if isinstance(section_ref.get("object"), dict)
    }
    index_payload = {
        "schema_version": SHARED_EVIDENCE_INDEX_SCHEMA,
        "scenario_set": "connector-contract-adapter",
        "report_date": report_date,
        "layout": COMPACT_EVIDENCE_LAYOUT,
        "object_root": _relative(object_root),
        "path_portability": {
            "portable_evidence_root": "reports/scenario/connector-contract-adapter",
            "historical_absolute_path_fields": [
                "objects referenced by sections.command_evidence may include historical transcript source paths",
                "objects referenced by sections.connector_contract_evidence may include historical transcript source paths",
            ],
            "regenerable_transcript_path_prefixes": [
                "tmp/scenario/",
            ],
            "claim_boundary": PATH_PORTABILITY_CLAIM_BOUNDARY,
        },
        "summary": {
            "section_count": len(section_index),
            "object_count": len(object_paths),
            "deduplicated_object_ref_count": len(object_paths),
        },
        "sections": section_index,
    }
    index_path = output_dir / f"shared-evidence-index-{report_date}.json"
    _write_json(index_path, index_payload)
    return index_path, index_payload, _shared_evidence_summary(section_index)


def _validate_shared_sections(
    aggregate: dict[str, Any],
    focused_reports: list[tuple[Path, str, dict[str, Any]]],
) -> dict[str, Any]:
    shared_sections: dict[str, Any] = {}
    for key in SHARED_SECTION_KEYS:
        if key not in aggregate:
            raise ValueError(f"aggregate report missing shared section {key}")
        shared_sections[key] = aggregate[key]
    for path, scenario_id, payload in focused_reports:
        for key, expected in shared_sections.items():
            focused_value = payload.get(key)
            if type(focused_value) is not type(expected):  # noqa: E721
                raise ValueError(
                    f"{_relative(path)} ({scenario_id}) section {key} has incompatible type"
                )
            if isinstance(expected, (dict, list)) and len(focused_value) != len(expected):
                raise ValueError(
                    f"{_relative(path)} ({scenario_id}) section {key} count mismatch: "
                    f"{len(focused_value)} != {len(expected)}"
                )
    return shared_sections


def _compact_envelope(
    *,
    source_path: Path,
    source_payload: dict[str, Any],
    output_path: Path,
    shared_index_path: Path,
    shared_index_sha256: str,
    shared_section_summaries: dict[str, dict[str, Any]],
    scenario_results: list[dict[str, Any]],
    summary: dict[str, Any],
    scenario_filter: list[str] | None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {}
    for key in [
        "schema_version",
        "cli_schema_version",
        "product",
        "version",
        "mode",
        "command",
        "owner_id",
        "tenant_id",
        "workspace_id",
        "namespace_id",
        "scenario_set",
        "status",
        "ids",
        "errors",
        "human_required",
        "self_command_transcript",
    ]:
        if key in source_payload:
            envelope[key] = source_payload[key]
    envelope["compact_schema_version"] = COMPACT_REPORT_SCHEMA
    envelope["compact_evidence_layout"] = COMPACT_EVIDENCE_LAYOUT
    envelope["source_report"] = _source_report_ref(source_path, source_payload)
    envelope["path_portability"] = _path_portability_ref(
        output_path=output_path,
        source_path=source_path,
    )
    envelope["shared_evidence_ref"] = {
        "path": _relative(shared_index_path),
        "sha256": shared_index_sha256,
        "layout": COMPACT_EVIDENCE_LAYOUT,
        "object_root": "reports/scenario/connector-contract-adapter/objects/sha256",
        "sections": shared_section_summaries,
    }
    envelope["summary"] = summary
    envelope["scenario_results"] = scenario_results
    if scenario_filter is not None:
        envelope["scenario_filter"] = scenario_filter
    envelope["output_path"] = str(output_path.resolve())
    return envelope


def compact_reports(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_date: str = DEFAULT_REPORT_DATE,
    expected_focused_count: int = 40,
    delete_sources: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    aggregate_path = source_dir / f"connector-contract-adapter-{report_date}.json"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"missing aggregate report: {aggregate_path}")
    aggregate = _load_json(aggregate_path)

    focused_paths = sorted(
        source_dir.glob(f"connector-contract-adapter-cs-ch-[0-9][0-9][0-9]-{report_date}.json")
    )
    if len(focused_paths) != expected_focused_count:
        raise ValueError(
            f"expected {expected_focused_count} focused reports, found {len(focused_paths)}"
        )
    focused_reports = [
        (path, _focused_report_scenario_id(path), _load_json(path))
        for path in focused_paths
    ]
    aggregate_ids = [
        row.get("id")
        for row in aggregate.get("scenario_results", [])
        if isinstance(row, dict)
    ]
    focused_ids = [scenario_id for _, scenario_id, _ in focused_reports]
    if focused_ids != aggregate_ids:
        raise ValueError(f"focused report IDs do not match aggregate IDs: {focused_ids} != {aggregate_ids}")

    for path, scenario_id, payload in focused_reports:
        scenario_filter = payload.get("scenario_filter")
        if scenario_filter != [scenario_id]:
            raise ValueError(f"{_relative(path)} scenario_filter expected [{scenario_id}]")
        scenario_results = payload.get("scenario_results")
        if (
            not isinstance(scenario_results, list)
            or len(scenario_results) != 1
            or not isinstance(scenario_results[0], dict)
            or scenario_results[0].get("id") != scenario_id
        ):
            raise ValueError(f"{_relative(path)} must contain exactly one matching scenario_result")

    shared_sections = _validate_shared_sections(aggregate, focused_reports)
    shared_index_path, shared_index_payload, shared_section_summaries = _write_shared_evidence_index(
        output_dir=output_dir,
        shared_sections=shared_sections,
        report_date=report_date,
    )
    legacy_shared_path = output_dir / f"shared-evidence-{report_date}.json"
    if legacy_shared_path.exists():
        legacy_shared_path.unlink()
    shared_index_sha256 = _sha256_file(shared_index_path)

    aggregate_output_path = output_dir / f"aggregate-{report_date}.json"
    aggregate_summary = dict(aggregate.get("summary") or {})
    aggregate_envelope = _compact_envelope(
        source_path=aggregate_path,
        source_payload=aggregate,
        output_path=aggregate_output_path,
        shared_index_path=shared_index_path,
        shared_index_sha256=shared_index_sha256,
        shared_section_summaries=shared_section_summaries,
        scenario_results=aggregate["scenario_results"],
        summary=aggregate_summary,
        scenario_filter=None,
    )
    _write_json(aggregate_output_path, aggregate_envelope)

    scenario_entries: list[dict[str, Any]] = []
    for source_path, scenario_id, source_payload in focused_reports:
        output_path = output_dir / "scenarios" / f"{scenario_id}.json"
        row = source_payload["scenario_results"][0]
        focused_summary = dict(source_payload.get("summary") or {})
        focused_summary.update(_scenario_summary(row))
        focused_summary["product_feature_claims"] = aggregate_summary.get("product_feature_claims")
        envelope = _compact_envelope(
            source_path=source_path,
            source_payload=source_payload,
            output_path=output_path,
            shared_index_path=shared_index_path,
            shared_index_sha256=shared_index_sha256,
            shared_section_summaries=shared_section_summaries,
            scenario_results=[row],
            summary=focused_summary,
            scenario_filter=[scenario_id],
        )
        _write_json(output_path, envelope)
        scenario_entries.append(
            {
                "scenario_id": scenario_id,
                "path": _relative(output_path),
                "sha256": _sha256_file(output_path),
                "source_report": envelope["source_report"],
            }
        )

    manifest_path = output_dir / f"manifest-{report_date}.json"
    manifest = {
        "schema_version": COMPACT_MANIFEST_SCHEMA,
        "scenario_set": "connector-contract-adapter",
        "report_date": report_date,
        "layout": COMPACT_EVIDENCE_LAYOUT,
        "summary": {
            "source_full_report_count": len(focused_reports) + 1,
            "compact_report_count": len(scenario_entries) + 1,
            "focused_scenario_count": len(scenario_entries),
            "shared_section_count": len(shared_sections),
            "shared_object_count": shared_index_payload["summary"]["object_count"],
            "source_total_size_bytes": aggregate_path.stat().st_size
            + sum(path.stat().st_size for path, _, _ in focused_reports),
            "compact_total_size_bytes": shared_index_path.stat().st_size
            + sum(
                (ROOT / section_ref["object"]["path"]).stat().st_size
                for section_ref in shared_index_payload["sections"].values()
                if isinstance(section_ref.get("object"), dict)
            )
            + aggregate_output_path.stat().st_size
            + sum((output_dir / "scenarios" / f"{entry['scenario_id']}.json").stat().st_size for entry in scenario_entries),
        },
        "path_portability": {
            "portable_evidence_root": "reports/scenario/connector-contract-adapter",
            "historical_absolute_path_fields": [
                "aggregate_report.source_report.output_path",
                "scenario_reports[].source_report.output_path",
            ],
            "claim_boundary": PATH_PORTABILITY_CLAIM_BOUNDARY,
        },
        "shared_evidence": {
            "path": _relative(shared_index_path),
            "sha256": shared_index_sha256,
            "layout": COMPACT_EVIDENCE_LAYOUT,
            "object_root": _relative(output_dir / "objects" / "sha256"),
            "sections": shared_section_summaries,
        },
        "aggregate_report": {
            "path": _relative(aggregate_output_path),
            "sha256": _sha256_file(aggregate_output_path),
            "source_report": aggregate_envelope["source_report"],
        },
        "scenario_reports": scenario_entries,
    }
    manifest["summary"]["compact_total_size_bytes"] += len(_canonical_bytes(manifest)) + 1
    _write_json(manifest_path, manifest)

    if delete_sources:
        aggregate_path.unlink()
        for path, _, _ in focused_reports:
            path.unlink()

    return manifest


def rewrite_existing_compact_layout(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_date: str = DEFAULT_REPORT_DATE,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / f"manifest-{report_date}.json"
    manifest = _load_json(manifest_path)
    shared_ref = manifest.get("shared_evidence")
    if not isinstance(shared_ref, dict) or not isinstance(shared_ref.get("path"), str):
        raise ValueError("compact manifest missing shared_evidence.path")
    shared_index = _load_json(ROOT / shared_ref["path"])
    shared_sections = _expand_shared_evidence_index(shared_index)
    shared_index_path, shared_index_payload, shared_section_summaries = _write_shared_evidence_index(
        output_dir=output_dir,
        shared_sections=shared_sections,
        report_date=report_date,
    )
    shared_index_sha256 = _sha256_file(shared_index_path)
    shared_evidence_ref = {
        "path": _relative(shared_index_path),
        "sha256": shared_index_sha256,
        "layout": COMPACT_EVIDENCE_LAYOUT,
        "object_root": _relative(output_dir / "objects" / "sha256"),
        "sections": shared_section_summaries,
    }

    aggregate_report = manifest.get("aggregate_report")
    if not isinstance(aggregate_report, dict) or not isinstance(aggregate_report.get("path"), str):
        raise ValueError("compact manifest missing aggregate_report.path")
    aggregate_path = ROOT / aggregate_report["path"]
    aggregate_payload = _load_json(aggregate_path)
    aggregate_payload["shared_evidence_ref"] = shared_evidence_ref
    _write_json(aggregate_path, aggregate_payload)
    aggregate_report["sha256"] = _sha256_file(aggregate_path)

    scenario_reports = manifest.get("scenario_reports")
    if not isinstance(scenario_reports, list):
        raise ValueError("compact manifest scenario_reports must be a list")
    for entry in scenario_reports:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("compact manifest scenario report entry missing path")
        report_path = ROOT / entry["path"]
        report_payload = _load_json(report_path)
        report_payload["shared_evidence_ref"] = shared_evidence_ref
        _write_json(report_path, report_payload)
        entry["sha256"] = _sha256_file(report_path)

    manifest["shared_evidence"] = shared_evidence_ref
    manifest["aggregate_report"] = aggregate_report
    manifest["scenario_reports"] = scenario_reports
    summary = dict(manifest.get("summary") or {})
    object_paths = {
        section_ref["object"]["path"]
        for section_ref in shared_index_payload["sections"].values()
        if isinstance(section_ref.get("object"), dict)
    }
    summary["shared_object_count"] = shared_index_payload["summary"]["object_count"]
    summary["compact_total_size_bytes"] = (
        shared_index_path.stat().st_size
        + sum((ROOT / object_path).stat().st_size for object_path in object_paths)
        + aggregate_path.stat().st_size
        + sum((ROOT / entry["path"]).stat().st_size for entry in scenario_reports)
    )
    manifest["summary"] = summary
    summary["compact_total_size_bytes"] += len(_canonical_bytes(manifest)) + 1
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact ConnectorHub connector-contract-adapter scenario reports."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-date", default=DEFAULT_REPORT_DATE)
    parser.add_argument("--expected-focused-count", type=int, default=40)
    parser.add_argument("--delete-sources", action="store_true")
    parser.add_argument(
        "--rewrite-existing-compact-layout",
        action="store_true",
        help="Rewrite an existing compact layout without regenerating source reports.",
    )
    args = parser.parse_args()
    if args.rewrite_existing_compact_layout:
        manifest = rewrite_existing_compact_layout(
            output_dir=Path(args.output_dir),
            report_date=args.report_date,
        )
    else:
        manifest = compact_reports(
            source_dir=Path(args.source_dir),
            output_dir=Path(args.output_dir),
            report_date=args.report_date,
            expected_focused_count=args.expected_focused_count,
            delete_sources=args.delete_sources,
        )
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
