from __future__ import annotations

import argparse
import textwrap
import urllib.parse

from ..config import resolve_base_url
from ..support import _http_json, _print_json


def command_context(args: argparse.Namespace) -> int:
    base_url = resolve_base_url(args.base_url).rstrip("/")
    query = urllib.parse.urlencode({"q": args.query})
    payload = _http_json(f"{base_url}/v1/context/query?{query}", timeout=args.timeout)
    if args.json:
        _print_json(payload)
        return 0
    print("Answer:")
    print(textwrap.fill(str(payload.get("answer", "")), width=100))
    print()
    print(f"Trust: {payload.get('trustLabel')}")
    freshness = payload.get("freshness", {})
    print(f"Freshness: {freshness.get('state', 'unknown')}")
    print(f"Official answer available: {payload.get('officialAnswerAvailable')}")
    evidence = payload.get("evidence", [])
    print()
    print("Citations:")
    if evidence:
        for index, citation in enumerate(evidence, start=1):
            validity = "valid" if citation.get("isValid") else "invalid"
            print(f"{index}. Evidence {citation.get('evidenceFragmentId')} — {validity}")
    else:
        print("None")
    limitations = payload.get("limitations", [])
    print()
    print("Limitations:")
    if limitations:
        for item in limitations:
            print(f"- {item}")
    else:
        print("None")
    return 0
