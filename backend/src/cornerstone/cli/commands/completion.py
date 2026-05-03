from __future__ import annotations

import argparse

from ..completion import completion_script


def command_completion(args: argparse.Namespace) -> int:
    print(completion_script(args.shell))
    return 0
