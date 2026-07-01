from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dlc_testforge.extract_td import build_td_index, write_td_index
from dlc_testforge.extract_spec import (
  build_spec_index,
  load_spec_index,
  lookup_dlc_spec,
  write_spec_index,
)
from dlc_testforge.index import build_index, write_index
from dlc_testforge.paths import discover_environment
from dlc_testforge.profiles import load_profiles, profile_summary


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


def cmd_index(args: argparse.Namespace) -> int:
  try:
    index = build_index(args.llvm_root)
    write_index(index, args.out)
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  return 0


def cmd_extract_spec(args: argparse.Namespace) -> int:
  try:
    index = build_spec_index(args.llvm_root)
    write_spec_index(index, args.out)
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  return 0


def cmd_extract_td(args: argparse.Namespace) -> int:
  try:
    index = build_td_index(args.llvm_root)
    write_td_index(index, args.out)
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  return 0


def cmd_lookup_spec(args: argparse.Namespace) -> int:
  try:
    index = load_spec_index(args.index)
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  matches = lookup_dlc_spec(index, args.topic)
  _print_json(
    {
      "topic": args.topic,
      "match_count": len(matches),
      "records": [record.to_dict() for record in matches],
    }
  )
  return 0


def cmd_list_profiles(args: argparse.Namespace) -> int:
  try:
    profiles = load_profiles(args.profiles_dir)
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  _print_json(
    {
      "profile_count": len(profiles),
      "profiles": [profile_summary(profile) for profile in profiles],
    }
  )
  return 0


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

  index_parser = subparsers.add_parser(
    "index", help="Index existing DLC CodeGen .ll and .mir tests."
  )
  _add_llvm_root(index_parser)
  index_parser.add_argument(
    "--out",
    required=True,
    type=Path,
    help="Path to write the test index JSON.",
  )
  index_parser.set_defaults(func=cmd_index)

  extract_spec_parser = subparsers.add_parser(
    "extract-spec", help="Extract searchable records from DLC Markdown specs."
  )
  _add_llvm_root(extract_spec_parser)
  extract_spec_parser.add_argument(
    "--out",
    required=True,
    type=Path,
    help="Path to write the DLC spec index JSON.",
  )
  extract_spec_parser.set_defaults(func=cmd_extract_spec)

  extract_td_parser = subparsers.add_parser(
    "extract-td", help="Extract DLC TableGen, intrinsic, builtin, and pass evidence."
  )
  _add_llvm_root(extract_td_parser)
  extract_td_parser.add_argument(
    "--out",
    required=True,
    type=Path,
    help="Path to write the DLC TableGen/source evidence index JSON.",
  )
  extract_td_parser.set_defaults(func=cmd_extract_td)

  lookup_spec_parser = subparsers.add_parser(
    "lookup-spec", help="Search a DLC spec index JSON by topic."
  )
  lookup_spec_parser.add_argument(
    "--index",
    required=True,
    type=Path,
    help="Path to a DLC spec index JSON produced by extract-spec.",
  )
  lookup_spec_parser.add_argument(
    "--topic",
    required=True,
    help="Case-insensitive topic substring to search for.",
  )
  lookup_spec_parser.set_defaults(func=cmd_lookup_spec)

  list_profiles_parser = subparsers.add_parser(
    "list-profiles", help="List available DLC mutation profiles."
  )
  _add_llvm_root(list_profiles_parser)
  list_profiles_parser.add_argument(
    "--profiles-dir",
    type=Path,
    default=None,
    help="Optional directory of profile YAML files to load.",
  )
  list_profiles_parser.set_defaults(func=cmd_list_profiles)

  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  sys.exit(main())
