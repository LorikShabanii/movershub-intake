"""Run the pipeline on one request from the command line.

    python -m intake.cli "Hi, I need to move from ... on 2026-11-20 ..." --provider mock
    python -m intake.cli --file request.txt --provider groq --today 2026-11-02
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

from .pipeline import run
from .providers import get_provider


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MoversHub intake pipeline")
    ap.add_argument("text", nargs="?", help="request text (or use --file)")
    ap.add_argument("--file", help="read the request from a file")
    ap.add_argument("--provider", default=os.getenv("INTAKE_PROVIDER", "groq"), choices=["groq", "mistral", "anthropic", "mock"])
    ap.add_argument("--today", type=date.fromisoformat, default=None, help="YYYY-MM-DD, for resolving relative dates")
    args = ap.parse_args(argv)

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    result = run(text, get_provider(args.provider), today=args.today)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
