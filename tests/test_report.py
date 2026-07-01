from __future__ import annotations

import json

from dlc_testforge.cli import main
from dlc_testforge.report import write_report_bundle


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _write_json(path, data):
  _write(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _make_run_dir(tmp_path):
  run_dir = tmp_path / "run"
  candidate = run_dir / "candidates" / "candidate-0001.ll"
  validation = run_dir / "results" / "candidate-0001" / "status.json"
  _write(candidate, "; RUN: llc %s -o -\ndefine i32 @f() { ret i32 0 }\n")
  _write_json(
    run_dir / "manifest.json",
    {
      "schema_version": 1,
      "workspace": str(run_dir),
      "profile": "example",
      "seed": "llvm/test/CodeGen/DLC/example.ll",
      "candidate_count": 1,
      "candidates": [
        {
          "path": "candidates/candidate-0001.ll",
          "seed": "llvm/test/CodeGen/DLC/example.ll",
          "profile": "example",
          "mutation_axis": "immediate_boundary",
          "source_value": 1,
          "new_value": 0,
          "line": 2,
          "comment": "; DLC-MUTATION: profile=example",
        }
      ],
    },
  )
  _write_json(
    validation,
    {
      "schema_version": 1,
      "candidate": str(candidate),
      "profile": "example",
      "overall_status": "pass",
      "steps": [],
    },
  )
  (run_dir / "results" / "classifications").mkdir(parents=True)
  return run_dir, candidate, validation


def _classification(candidate, validation, state, *, stdout="", stderr=""):
  return {
    "schema_version": 1,
    "candidate": str(candidate),
    "profile": "example",
    "state": state,
    "reason": "test reason",
    "validation": str(validation),
    "evidence_steps": [
      {
        "level": "command",
        "status": "pass" if state == "accepted-regression-candidate" else "fail",
        "exit_code": 0 if state == "accepted-regression-candidate" else 1,
        "reason": None,
        "command": "llc candidate.ll -o -",
        "stdout_path": None,
        "stderr_path": None,
      }
    ],
    "stdout_excerpt": stdout,
    "stderr_excerpt": stderr,
    "requires_human_triage": state.startswith("bug-scout-"),
  }


def test_accepted_classification_creates_regression_bundle(tmp_path):
  run_dir, candidate, validation = _make_run_dir(tmp_path)
  _write_json(
    run_dir / "results" / "classifications" / "candidate-0001.json",
    _classification(candidate, validation, "accepted-regression-candidate"),
  )

  summary = write_report_bundle(run_dir)

  bundle = run_dir / "reports" / "regression-candidates" / "candidate-0001"
  assert summary.classification_count == 1
  assert summary.regression_bundles[0].path == str(bundle)
  assert (bundle / "candidate.ll").read_text(encoding="utf-8") == candidate.read_text(
    encoding="utf-8"
  )
  assert (bundle / "validation.json").is_file()
  assert (bundle / "classification.json").is_file()
  command_log = (bundle / "command.log").read_text(encoding="utf-8")
  assert "llc candidate.ll -o -" in command_log
  summary_md = (bundle / "summary.md").read_text(encoding="utf-8")
  assert "Regression Candidate" in summary_md
  assert "immediate_boundary: 1 -> 0 at line 2" in summary_md
  assert "llvm/test/CodeGen/DLC/candidate-0001.ll" in summary_md


def test_bug_scout_classification_creates_bug_bundle(tmp_path):
  run_dir, candidate, validation = _make_run_dir(tmp_path)
  _write_json(
    run_dir / "results" / "classifications" / "candidate-0001.json",
    _classification(
      candidate,
      validation,
      "bug-scout-assertion",
      stdout="stdout text",
      stderr="LLVM ERROR: failed",
    ),
  )

  summary = write_report_bundle(run_dir)

  bundle = run_dir / "reports" / "bug-scout" / "candidate-0001"
  assert summary.bug_bundles[0].path == str(bundle)
  assert (bundle / "reproducer.ll").is_file()
  assert (bundle / "stdout.txt").read_text(encoding="utf-8") == "stdout text"
  assert (bundle / "stderr.txt").read_text(encoding="utf-8") == "LLVM ERROR: failed"
  assert (bundle / "command.sh").read_text(encoding="utf-8") == "llc candidate.ll -o -\n"
  summary_md = (bundle / "summary.md").read_text(encoding="utf-8")
  assert "Bug Scout Candidate" in summary_md
  assert "bug-scout-assertion" in summary_md


def test_non_bundle_states_are_skipped(tmp_path):
  run_dir, candidate, validation = _make_run_dir(tmp_path)
  _write_json(
    run_dir / "results" / "classifications" / "candidate-0001.json",
    _classification(candidate, validation, "needs-checks"),
  )

  summary = write_report_bundle(run_dir)

  assert summary.regression_bundles == []
  assert summary.bug_bundles == []
  assert summary.skipped[0].state == "needs-checks"
  assert not (run_dir / "reports" / "regression-candidates").exists()


def test_missing_candidate_is_documented_instead_of_crashing(tmp_path):
  run_dir, candidate, validation = _make_run_dir(tmp_path)
  missing_candidate = run_dir / "candidates" / "missing.ll"
  _write_json(
    run_dir / "results" / "classifications" / "missing.json",
    _classification(missing_candidate, validation, "accepted-regression-candidate"),
  )

  write_report_bundle(run_dir)

  bundle = run_dir / "reports" / "regression-candidates" / "missing"
  assert "missing candidate artifact" in (bundle / "candidate.ll").read_text(
    encoding="utf-8"
  )
  assert "Missing Artifacts" in (bundle / "summary.md").read_text(encoding="utf-8")


def test_missing_classification_directory_fails(tmp_path):
  run_dir = tmp_path / "run"
  _write_json(run_dir / "manifest.json", {"schema_version": 1})

  try:
    write_report_bundle(run_dir)
  except ValueError as exc:
    assert "classifications directory" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_cli_report_prints_summary_and_writes_out(tmp_path, capsys):
  run_dir, candidate, validation = _make_run_dir(tmp_path)
  _write_json(
    run_dir / "results" / "classifications" / "candidate-0001.json",
    _classification(candidate, validation, "accepted-regression-candidate"),
  )
  out_path = tmp_path / "summary.json"

  assert main(["report", "--run-dir", str(run_dir), "--out", str(out_path)]) == 0

  printed = json.loads(capsys.readouterr().out)
  written = json.loads(out_path.read_text(encoding="utf-8"))
  assert printed["regression_bundle_count"] == 1
  assert written["regression_bundle_count"] == 1
  assert written["classification_count"] == 1
