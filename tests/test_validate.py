from __future__ import annotations

import json

from dlc_testforge.cli import main
from dlc_testforge.validate import validate_candidate


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _script(path, text):
  _write(path, text)
  path.chmod(0o755)


def _make_llvm_root(tmp_path, *, fail_llvm_as=False, fail_llc=False, fail_lit=False):
  llvm_root = tmp_path / "LLVM"
  build_bin = llvm_root / "build" / "bin"
  _script(
    build_bin / "llvm-as",
    f"""#!/bin/sh
echo llvm-as stdout
echo llvm-as stderr >&2
exit {1 if fail_llvm_as else 0}
""",
  )
  _script(
    build_bin / "llc",
    f"""#!/bin/sh
echo "$@" > "{tmp_path / 'llc-argv.txt'}"
echo llc stdout
echo llc stderr >&2
exit {1 if fail_llc else 0}
""",
  )
  _script(
    build_bin / "llvm-lit",
    f"""#!/bin/sh
echo llvm-lit stdout
echo llvm-lit stderr >&2
exit {1 if fail_lit else 0}
""",
  )
  _script(build_bin / "FileCheck", "#!/bin/sh\nexit 0\n")
  _script(build_bin / "clang", "#!/bin/sh\nexit 0\n")
  (llvm_root / "llvm" / "test" / "CodeGen" / "DLC").mkdir(parents=True)
  (llvm_root / "docs" / "dlc_spec").mkdir(parents=True)
  (llvm_root / "llvm" / "lib" / "Target" / "DLC").mkdir(parents=True)
  _write(llvm_root / "llvm" / "include" / "llvm" / "IR" / "IntrinsicsDLC.td", "")
  return llvm_root


def _make_profiles_dir(tmp_path, command="{llc} -mtriple=dlc {input} -o -"):
  profiles_dir = tmp_path / "profiles"
  _write(
    profiles_dir / "example.yaml",
    f"""name: example
description: Example validation profile.
seed_selectors:
  paths:
    - llvm/test/CodeGen/DLC/example.ll
  features:
    - llc
commands:
  base:
    - "{command}"
validation:
  allow_verify_machineinstrs: false
  required_levels:
    - syntax
    - command
    - filecheck
mutation_axes:
  immediates:
    enabled: true
    values: [0, 1]
bug_scout:
  enabled: true
  classify_crash: true
""",
  )
  return profiles_dir


def _candidate(path, *, checks=True):
  check_text = "; CHECK: example\n" if checks else ""
  _write(
    path,
    f"""; RUN: llc -mtriple=dlc < %s | FileCheck %s
define i32 @example(i32 %x) {{
  ret i32 %x
}}
{check_text}""",
  )
  return path


def test_validate_passes_and_writes_status_and_logs(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll")

  report = validate_candidate(
    llvm_root, candidate, "example", tmp_path / "validation", profiles_dir=profiles_dir
  )

  assert report.overall_status == "pass"
  status = json.loads((tmp_path / "validation" / "status.json").read_text())
  assert status["overall_status"] == "pass"
  assert [step["level"] for step in status["steps"]] == [
    "syntax",
    "command",
    "spec",
    "filecheck",
    "suite",
  ]
  assert (tmp_path / "validation" / "logs" / "syntax.stdout").is_file()
  assert (tmp_path / "validation" / "logs" / "command.stderr").is_file()
  assert (tmp_path / "validation" / "logs" / "filecheck.stdout").is_file()
  assert status["suggested_suite_command"].endswith("check-llvm-codegen-dlc")


def test_syntax_failure_is_structured(tmp_path):
  llvm_root = _make_llvm_root(tmp_path, fail_llvm_as=True)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll")

  report = validate_candidate(
    llvm_root, candidate, "example", tmp_path / "validation", profiles_dir=profiles_dir
  )

  assert report.overall_status == "fail"
  syntax = report.steps[0]
  assert syntax.level == "syntax"
  assert syntax.status == "fail"
  assert syntax.exit_code == 1
  assert syntax.stderr_path is not None
  assert syntax.stderr_path.read_text(encoding="utf-8").strip() == "llvm-as stderr"


def test_command_failure_is_structured(tmp_path):
  llvm_root = _make_llvm_root(tmp_path, fail_llc=True)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll")

  report = validate_candidate(
    llvm_root, candidate, "example", tmp_path / "validation", profiles_dir=profiles_dir
  )

  assert report.overall_status == "fail"
  command = [step for step in report.steps if step.level == "command"][0]
  assert command.status == "fail"
  assert command.exit_code == 1


def test_candidate_without_checks_needs_checks(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll", checks=False
  )

  report = validate_candidate(
    llvm_root, candidate, "example", tmp_path / "validation", profiles_dir=profiles_dir
  )

  assert report.overall_status == "needs-checks"
  filecheck = [step for step in report.steps if step.level == "filecheck"][0]
  assert filecheck.status == "needs-checks"


def test_candidate_outside_test_tree_skips_lit(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(tmp_path / "outside.ll")

  report = validate_candidate(
    llvm_root, candidate, "example", tmp_path / "validation", profiles_dir=profiles_dir
  )

  assert report.overall_status == "needs-checks"
  filecheck = [step for step in report.steps if step.level == "filecheck"][0]
  assert filecheck.status == "skipped"
  assert filecheck.reason is not None
  assert "lit_unavailable" in filecheck.reason


def test_candidate_outside_test_tree_can_stage_for_lit(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(tmp_path / "outside.ll")

  report = validate_candidate(
    llvm_root,
    candidate,
    "example",
    tmp_path / "validation",
    profiles_dir=profiles_dir,
    stage_in_tree=True,
  )

  assert report.overall_status == "pass"
  filecheck = [step for step in report.steps if step.level == "filecheck"][0]
  assert filecheck.status == "pass"
  assert filecheck.details is not None
  assert filecheck.details["staged_from"] == str(candidate.resolve(strict=False))
  staged_path = filecheck.details["staged_path"]
  assert ".dlc-testforge-staging" in staged_path
  assert not any(
    (llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / ".dlc-testforge-staging").iterdir()
  )


def test_mir_candidate_skips_syntax_and_runs_command(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path, command="{llc} -run-pass=legalizer {input} -o -")
  candidate = llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.mir"
  _write(candidate, "# RUN: llc -run-pass=legalizer %s -o - | FileCheck %s\n# CHECK: G_ADD\n")

  report = validate_candidate(
    llvm_root, candidate, "example", tmp_path / "validation", profiles_dir=profiles_dir
  )

  syntax = [step for step in report.steps if step.level == "syntax"][0]
  command = [step for step in report.steps if step.level == "command"][0]
  assert syntax.status == "skipped"
  assert command.status == "pass"


def test_missing_candidate_fails_clearly(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)

  try:
    validate_candidate(
      llvm_root,
      tmp_path / "missing.ll",
      "example",
      tmp_path / "validation",
      profiles_dir=profiles_dir,
    )
  except ValueError as exc:
    assert "candidate file not found" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_cli_validate_prints_summary(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll")

  assert (
    main(
      [
        "validate",
        "--llvm-root",
        str(llvm_root),
        "--candidate",
        str(candidate),
        "--profile",
        "example",
        "--out-dir",
        str(tmp_path / "validation"),
        "--profiles-dir",
        str(profiles_dir),
      ]
    )
    == 0
  )

  result = json.loads(capsys.readouterr().out)
  assert result["overall_status"] == "pass"
  assert result["profile"] == "example"
  assert result["failed_steps"] == []
  assert (tmp_path / "validation" / "status.json").is_file()


def test_validator_does_not_add_machine_verifier(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  candidate = _candidate(llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll")

  validate_candidate(
    llvm_root, candidate, "example", tmp_path / "validation", profiles_dir=profiles_dir
  )

  argv = (tmp_path / "llc-argv.txt").read_text(encoding="utf-8")
  assert "-verify-machineinstrs" not in argv
