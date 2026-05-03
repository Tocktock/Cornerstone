from __future__ import annotations

import argparse
import sys

from ..config import config_path, config_payload, set_config_value, unset_config_value
from ..support import _print_json


def command_config(args: argparse.Namespace) -> int:
    if args.action == "path":
        print(config_path())
        return 0
    if args.action == "get":
        payload = config_payload()
        if args.json:
            _print_json(payload)
        else:
            print(f"Config path: {payload['path']}")
            for key, value in payload["config"].items():
                print(f"{key}: {value}")
        return 0
    if args.action == "set":
        try:
            path = set_config_value(args.key, args.value)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Updated {path}")
        return 0
    if args.action == "unset":
        try:
            path = unset_config_value(args.key)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Updated {path}")
        return 0
    print(f"ERROR: unknown config action {args.action}", file=sys.stderr)
    return 1
