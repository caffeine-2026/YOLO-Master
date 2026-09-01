#!/usr/bin/env python3
"""Run an unchanged validator while redirecting its fixed legacy output path."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("validator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = (REPO_ROOT / args.source).resolve()
    destination = (REPO_ROOT / args.destination).resolve()
    destination.relative_to(REPO_ROOT)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite redirected validator output: {args.destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(args.module)
    original_write_text = Path.write_text

    def redirected_write_text(path: Path, data: str, *write_args, **write_kwargs) -> int:
        target = destination if path.resolve() == source else path
        return original_write_text(target, data, *write_args, **write_kwargs)

    Path.write_text = redirected_write_text
    forwarded = args.validator_args[1:] if args.validator_args[:1] == ["--"] else args.validator_args
    previous_argv = sys.argv
    sys.argv = [args.module, *forwarded]
    try:
        return int(module.main())
    finally:
        Path.write_text = original_write_text
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
