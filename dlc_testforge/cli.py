from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dlc_testforge.classify import (
  classification_summary,
  classify_validation,
  write_classification,
)
from dlc_testforge.extract_td import build_td_index, write_td_index
from dlc_testforge.extract_spec import (
  build_spec_index,
  load_spec_index,
  lookup_dlc_spec,
  write_spec_index,
)
from dlc_testforge.index import build_index, write_index
from dlc_testforge.generate import create_workspace, generation_summary
from dlc_testforge.paths import discover_environment
from dlc_testforge.profiles import load_profiles, profile_summary
from dlc_testforge.reduce import reduce_bug_bundle, reduction_summary
from dlc_testforge.report import write_report_bundle, write_report_summary
from dlc_testforge.validate import validate_candidate, validation_summary


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


def cmd_generate(args: argparse.Namespace) -> int:
  try:
    manifest = create_workspace(
      args.llvm_root,
      args.profile,
      args.seed,
      args.out_dir,
      dry_run=args.dry_run,
      profiles_dir=args.profiles_dir,
      max_candidates=args.max_candidates,
      mode=args.mode,
      agent_proposal=args.agent_proposal,
      agent_model=args.agent_model,
      agent_endpoint=args.agent_endpoint,
    )
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  _print_json(generation_summary(manifest))
  return 0


def cmd_validate(args: argparse.Namespace) -> int:
  try:
    report = validate_candidate(
      args.llvm_root,
      args.candidate,
      args.profile,
      args.out_dir,
      profiles_dir=args.profiles_dir,
      timeout=args.timeout,
    )
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  _print_json(validation_summary(report))
  return 0 if report.overall_status != "fail" else 1


def cmd_classify(args: argparse.Namespace) -> int:
  try:
    report = classify_validation(args.validation, profiles_dir=args.profiles_dir)
    if args.out is not None:
      write_classification(report, args.out)
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  _print_json(classification_summary(report))
  return 0


def cmd_report(args: argparse.Namespace) -> int:
  try:
    summary = write_report_bundle(args.run_dir)
    if args.out is not None:
      write_report_summary(summary, args.out)
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  _print_json(summary.to_dict())
  return 0


def cmd_reduce(args: argparse.Namespace) -> int:
  try:
    report = reduce_bug_bundle(
      args.bundle_dir,
      args.llvm_root,
      args.profile,
      args.out_dir,
      profiles_dir=args.profiles_dir,
      timeout=args.timeout,
    )
  except OSError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  _print_json(reduction_summary(report))
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

  generate_parser = subparsers.add_parser(
    "generate", help="Create a DLC mutation workspace."
  )
  _add_llvm_root(generate_parser)
  generate_parser.add_argument(
    "--profile",
    required=True,
    help="Profile name to use.",
  )
  generate_parser.add_argument(
    "--seed",
    required=True,
    help="LLVM-root-relative seed test path.",
  )
  generate_parser.add_argument(
    "--out-dir",
    required=True,
    type=Path,
    help="Workspace directory to create.",
  )
  generate_parser.add_argument(
    "--profiles-dir",
    type=Path,
    default=None,
    help="Optional directory of profile YAML files to load.",
  )
  generate_parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Create workspace inputs but do not write candidate files.",
  )
  generate_parser.add_argument(
    "--max-candidates",
    type=int,
    default=10,
    help="Maximum number of mutation candidates to write.",
  )
  generate_parser.add_argument(
    "--mode",
    choices=["manual", "agent"],
    default="manual",
    help="Generation mode. Agent mode asks for structured mutation proposals before writing candidates.",
  )
  generate_parser.add_argument(
    "--agent-proposal",
    type=Path,
    default=None,
    help="Optional JSON proposal file for agent mode. Skips the LLM request.",
  )
  generate_parser.add_argument(
    "--agent-model",
    default=None,
    help="LLM model for agent mode when --agent-proposal is not provided.",
  )
  generate_parser.add_argument(
    "--agent-endpoint",
    default=None,
    help="OpenAI-compatible API endpoint for agent mode.",
  )
  generate_parser.set_defaults(func=cmd_generate)

  validate_parser = subparsers.add_parser(
    "validate", help="Validate one DLC mutation candidate."
  )
  _add_llvm_root(validate_parser)
  validate_parser.add_argument(
    "--candidate",
    required=True,
    type=Path,
    help="Candidate .ll or .mir file to validate.",
  )
  validate_parser.add_argument(
    "--profile",
    required=True,
    help="Profile name to use.",
  )
  validate_parser.add_argument(
    "--out-dir",
    required=True,
    type=Path,
    help="Directory for validation status and logs.",
  )
  validate_parser.add_argument(
    "--profiles-dir",
    type=Path,
    default=None,
    help="Optional directory of profile YAML files to load.",
  )
  validate_parser.add_argument(
    "--timeout",
    type=int,
    default=30,
    help="Per-command timeout in seconds.",
  )
  validate_parser.set_defaults(func=cmd_validate)

  classify_parser = subparsers.add_parser(
    "classify", help="Classify one validation status JSON file."
  )
  classify_parser.add_argument(
    "--validation",
    required=True,
    type=Path,
    help="Path to a validation status.json file.",
  )
  classify_parser.add_argument(
    "--profiles-dir",
    type=Path,
    default=None,
    help="Optional directory of profile YAML files to load.",
  )
  classify_parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Optional path to write full classification JSON.",
  )
  classify_parser.set_defaults(func=cmd_classify)

  report_parser = subparsers.add_parser(
    "report", help="Write review bundles from classification JSON files."
  )
  report_parser.add_argument(
    "--run-dir",
    required=True,
    type=Path,
    help="Mutation workspace containing manifest.json and results/classifications.",
  )
  report_parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Optional path to write report summary JSON.",
  )
  report_parser.set_defaults(func=cmd_report)

  reduce_parser = subparsers.add_parser(
    "reduce", help="Reduce one bug-scout report bundle."
  )
  _add_llvm_root(reduce_parser)
  reduce_parser.add_argument(
    "--bundle-dir",
    required=True,
    type=Path,
    help="Path to a reports/bug-scout/<candidate-id> bundle.",
  )
  reduce_parser.add_argument(
    "--profile",
    required=True,
    help="Profile name to use for validation.",
  )
  reduce_parser.add_argument(
    "--out-dir",
    required=True,
    type=Path,
    help="Directory for reduction attempts and result files.",
  )
  reduce_parser.add_argument(
    "--profiles-dir",
    type=Path,
    default=None,
    help="Optional directory of profile YAML files to load.",
  )
  reduce_parser.add_argument(
    "--timeout",
    type=int,
    default=30,
    help="Per-command timeout in seconds.",
  )
  reduce_parser.set_defaults(func=cmd_reduce)

  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  sys.exit(main())
