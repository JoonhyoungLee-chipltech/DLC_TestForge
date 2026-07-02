from __future__ import annotations

import json

from dlc_testforge.classify import classify_validation
from dlc_testforge.cli import main


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _make_profiles_dir(tmp_path, *, required_levels=None):
  if required_levels is None:
    required_levels = ["syntax", "command", "filecheck"]
  required_yaml = "\n".join(f"    - {level}" for level in required_levels)
  profiles_dir = tmp_path / "profiles"
  _write(
    profiles_dir / "example.yaml",
    f"""name: example
description: Example classification profile.
seed_selectors:
  paths:
    - llvm/test/CodeGen/DLC/example.ll
  features:
    - llc
commands:
  base:
    - "{{llc}} -mtriple=dlc {{input}} -o -"
validation:
  allow_verify_machineinstrs: false
  required_levels:
{required_yaml}
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


def _status(tmp_path, steps, *, profile="example"):
  status = {
    "schema_version": 1,
    "candidate": str(tmp_path / "candidate.ll"),
    "profile": profile,
    "overall_status": "pass",
    "steps": steps,
    "suggested_suite_command": "ninja -C /root/LLVM/build check-llvm-codegen-dlc",
    "out_dir": str(tmp_path / "validation"),
  }
  status_path = tmp_path / "validation" / "status.json"
  _write(status_path, json.dumps(status))
  return status_path


def _step(level, status, *, exit_code=None, reason=None, stderr=None, tmp_path=None):
  stderr_path = None
  if stderr is not None:
    if tmp_path is None:
      raise AssertionError("tmp_path required for stderr")
    stderr_path = tmp_path / "logs" / f"{level}.stderr"
    _write(stderr_path, stderr)
  return {
    "level": level,
    "status": status,
    "command": f"{level} command",
    "exit_code": exit_code,
    "stdout_path": None,
    "stderr_path": str(stderr_path) if stderr_path is not None else None,
    "duration_ms": 1,
    "reason": reason,
  }


def _passing_steps():
  return [
    _step("syntax", "pass", exit_code=0),
    _step("command", "pass", exit_code=0),
    _step("spec", "unknown", reason="not implemented"),
    _step("filecheck", "pass", exit_code=0),
    _step("suite", "skipped", reason="manual"),
  ]


def test_passing_required_levels_are_accepted(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  status_path = _status(tmp_path, _passing_steps())

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "accepted-regression-candidate"
  assert report.requires_human_triage is False


def test_syntax_failure_rejects_invalid_ir(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  status_path = _status(
    tmp_path,
    [
      _step("syntax", "fail", exit_code=1),
      _step("command", "skipped"),
    ],
  )

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "rejected-invalid-ir"
  assert [step["level"] for step in report.evidence_steps] == ["syntax"]


def test_spec_failure_rejects_spec_conflict(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  steps = _passing_steps()
  steps[2] = _step("spec", "fail", reason="reserved value")
  status_path = _status(tmp_path, steps)

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "rejected-spec-conflict"


def test_timeout_is_bug_scout_timeout(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  status_path = _status(
    tmp_path,
    [
      _step("syntax", "pass", exit_code=0),
      _step("command", "fail", exit_code=124, reason="timeout"),
    ],
  )

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "bug-scout-timeout"
  assert report.requires_human_triage is True


def test_crash_like_command_failure_is_bug_scout_crash(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  status_path = _status(
    tmp_path,
    [
      _step("syntax", "pass", exit_code=0),
      _step(
        "command",
        "fail",
        exit_code=139,
        stderr="Segmentation fault (core dumped)\n",
        tmp_path=tmp_path,
      ),
    ],
  )

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "bug-scout-crash"
  assert "Segmentation fault" in report.stderr_excerpt


def test_assertion_failure_is_bug_scout_assertion(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  status_path = _status(
    tmp_path,
    [
      _step("syntax", "pass", exit_code=0),
      _step(
        "command",
        "fail",
        exit_code=1,
        stderr="LLVM ERROR: unreachable selected\n",
        tmp_path=tmp_path,
      ),
    ],
  )

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "bug-scout-assertion"


def test_plain_command_failure_is_compile_failure(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  status_path = _status(
    tmp_path,
    [
      _step("syntax", "pass", exit_code=0),
      _step("command", "fail", exit_code=1, stderr="bad operand\n", tmp_path=tmp_path),
    ],
  )

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "bug-scout-compile-failure"


def test_missing_or_skipped_required_filecheck_needs_checks(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  steps = _passing_steps()
  steps[3] = _step("filecheck", "skipped", reason="lit_unavailable")
  status_path = _status(tmp_path, steps)

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "needs-checks"
  assert report.reason == "required FileCheck validation was skipped: lit_unavailable"


def test_failed_required_filecheck_is_rejected_check_failure(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  steps = _passing_steps()
  steps[3] = _step("filecheck", "fail", exit_code=1, stderr="CHECK failed\n", tmp_path=tmp_path)
  status_path = _status(tmp_path, steps)

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "rejected-check-failure"
  assert report.reason == "required FileCheck validation failed"
  assert "CHECK failed" in report.stderr_excerpt


def test_non_required_unknown_and_skipped_do_not_block_acceptance(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path, required_levels=["syntax", "command"])
  steps = _passing_steps()
  steps[3] = _step("filecheck", "skipped", reason="outside test tree")
  status_path = _status(tmp_path, steps)

  report = classify_validation(status_path, profiles_dir=profiles_dir)

  assert report.state == "accepted-regression-candidate"


def test_cli_classify_prints_summary_and_writes_full_report(tmp_path, capsys):
  profiles_dir = _make_profiles_dir(tmp_path)
  status_path = _status(tmp_path, _passing_steps())
  out_path = tmp_path / "classification.json"

  assert (
    main(
      [
        "classify",
        "--validation",
        str(status_path),
        "--profiles-dir",
        str(profiles_dir),
        "--out",
        str(out_path),
      ]
    )
    == 0
  )

  summary = json.loads(capsys.readouterr().out)
  full_report = json.loads(out_path.read_text(encoding="utf-8"))
  assert summary["state"] == "accepted-regression-candidate"
  assert full_report["state"] == "accepted-regression-candidate"
  assert full_report["schema_version"] == 1
