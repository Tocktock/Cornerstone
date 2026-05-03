from __future__ import annotations

import sys
from typing import Sequence

from .parser import build_parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "Could not reach" in str(exc):
            print("Next action: start the API with `cornerstone api --reload` or set --base-url.", file=sys.stderr)
        return 1
