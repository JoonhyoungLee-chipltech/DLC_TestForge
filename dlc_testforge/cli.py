from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dlc_testforge.paths import discover_environment


def _add_llvm_root(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--llvm-root",
    required=True,
    type=Path,
    help="Path to the LLVM checkout to inspect.",
  )
  parser.add_argument(
    "--archer-reference",
    type=Path,
    default=Path("/root/references/Archer"),
    help="Optional read-only Archer reference path.",
  )


def _print_json(data: object) -> None:
  print(json.dumps(data, indent=2, sort_keys=True))


def cmd_env(args: argparse.Namespace) -> int:
  report = discover_environment(
    args.llvm_root,
    archer_reference=args.archer_reference,
    check_versions=False,
  )
  _print_json(report.to_dict())
  return 0 if report.llvm_root.ok else 2


def cmd_check_tools(args: argparse.Namespace) -> int:
  report = discover_environment(
    args.llvm_root,
    archer_reference=args.archer_reference,
    check_versions=True,
  )
  _print_json(report.to_dict())
  return 0 if report.ok else 2


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="dlc-testforge",
    description="DLC TestForge environment discovery CLI.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  env_parser = subparsers.add_parser(
    "env", help="Print discovered LLVM/DLC environment as JSON."
  )
  _add_llvm_root(env_parser)
  env_parser.set_defaults(func=cmd_env)

  check_parser = subparsers.add_parser(
    "check-tools", help="Check required LLVM/DLC tools and paths."
  )
  _add_llvm_root(check_parser)
  check_parser.set_defaults(func=cmd_check_tools)

  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  sys.exit(main())
