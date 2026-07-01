from __future__ import annotations

import json

from dlc_testforge.cli import main
from dlc_testforge.reduce import reduce_bug_bundle


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _script(path, text):
  _write(path, text)
  path.chmod(0o755)


def _make_llvm_root(tmp_path, *, fail_token="TRIGGER"):
  llvm_root = tmp_path / "LLVM"
  build_bin = llvm_root / "build" / "bin"
  _script(build_bin / "llvm-as", "#!/bin/sh\nexit 0\n")
  _script(
    build_bin / "llc",
    f"""#!/bin/sh
for arg in "$@"; do
  if [ -f "$arg" ] && grep -q "{fail_token}" "$arg"; then
    echo "LLVM ERROR: reducer fixture" >&2
    exit 1
  fi
done
exit 0
""",
  )
  _script(build_bin / "llvm-lit", "#!/bin/sh\nexit 0\n")
  _script(build_bin / "FileCheck", "#!/bin/sh\nexit 0\n")
  _script(build_bin / "clang", "#!/bin/sh\nexit 0\n")
  (llvm_root / "llvm" / "test" / "CodeGen" / "DLC").mkdir(parents=True)
  (llvm_root / "docs" / "dlc_spec").mkdir(parents=True)
  (llvm_root / "llvm" / "lib" / "Target" / "DLC").mkdir(parents=True)
  _write(llvm_root / "llvm" / "include" / "llvm" / "IR" / "IntrinsicsDLC.td", "")
  return llvm_root


def _make_profiles_dir(tmp_path):
  profiles_dir = tmp_path / "profiles"
  _write(
    profiles_dir / "example.yaml",
    """name: example
description: Example reducer profile.
seed_selectors:
  paths:
    - llvm/test/CodeGen/DLC/example.ll
  features:
    - llc
commands:
  base:
    - "{llc} -mtriple=dlc {input} -o -"
validation:
  allow_verify_machineinstrs: false
  required_levels:
    - syntax
    - command
mutation_axes:
  immediates:
    enabled: true
    values: [0, 1]
bug_scout:
  enabled: true
  classify_assertion: true
""",
  )
  return profiles_dir


def _make_bundle(tmp_path, text, *, state="bug-scout-assertion", suffix=".ll"):
  bundle = tmp_path / "bundle"
  reproducer = bundle / f"reproducer{suffix}"
  _write(reproducer, text)
  _write(
    bundle / "classification.json",
    json.dumps(
      {
        "schema_version": 1,
        "candidate": str(reproducer),
        "profile": "example",
        "state": state,
        "reason": "fixture",
        "validation": str(bundle / "validation.json"),
        "evidence_steps": [],
        "stdout_excerpt": "",
        "stderr_excerpt": "LLVM ERROR: reducer fixture",
        "requires_human_triage": state.startswith("bug-scout-"),
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
  )
  _write(bundle / "validation.json", json.dumps({"schema_version": 1}) + "\n")
  return bundle


def _multi_function_text(*, trigger_in_helper=False):
  helper_body = "  ret i32 0\n"
  if trigger_in_helper:
    helper_body = "  ; TRIGGER\n  ret i32 0\n"
  return f"""; RUN: llc -mtriple=dlc %s -o -
define i32 @helper() {{
{helper_body}}}

define i32 @bug() {{
entry:
; DLC-MUTATION: profile=example axis=immediate_boundary source_value=1 new_value=0
  ; TRIGGER
  ret i32 0
}}
"""


def test_reduces_unrelated_function_and_preserves_failure_state(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  bundle = _make_bundle(tmp_path, _multi_function_text())

  report = reduce_bug_bundle(
    bundle,
    llvm_root,
    "example",
    tmp_path / "reduced",
    profiles_dir=profiles_dir,
  )

  reduced = report.reduced_path.read_text(encoding="utf-8")
  assert report.status == "reduced"
  assert report.original_state == "bug-scout-assertion"
  assert report.final_state == "bug-scout-assertion"
  assert report.accepted_reduction_count == 1
  assert "@helper" not in reduced
  assert "@bug" in reduced
  assert "; RUN:" in reduced
  assert "DLC-MUTATION:" in reduced
  assert (tmp_path / "reduced" / "attempts" / "attempt-0001" / "classification.json").is_file()


def test_keeps_original_when_removal_changes_failure_state(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  text = """; RUN: llc -mtriple=dlc %s -o -
define i32 @helper() {
  ; TRIGGER
  ret i32 0
}

define i32 @bug() {
entry:
; DLC-MUTATION: profile=example axis=immediate_boundary source_value=1 new_value=0
  ret i32 0
}
"""
  bundle = _make_bundle(tmp_path, text)

  report = reduce_bug_bundle(
    bundle,
    llvm_root,
    "example",
    tmp_path / "reduced",
    profiles_dir=profiles_dir,
  )

  assert report.status == "unchanged"
  assert report.accepted_reduction_count == 0
  assert report.rejected_attempts[0].state != "bug-scout-assertion"
  assert report.reduced_path.read_text(encoding="utf-8") == text


def test_single_function_is_unchanged(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  text = """; RUN: llc -mtriple=dlc %s -o -
define i32 @bug() {
; DLC-MUTATION: profile=example
  ; TRIGGER
  ret i32 0
}
"""
  bundle = _make_bundle(tmp_path, text)

  report = reduce_bug_bundle(
    bundle,
    llvm_root,
    "example",
    tmp_path / "reduced",
    profiles_dir=profiles_dir,
  )

  assert report.status == "unchanged"
  assert report.attempt_count == 0
  assert report.reduced_path.read_text(encoding="utf-8") == text


def test_non_ll_reproducer_is_copied_unchanged(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  bundle = _make_bundle(tmp_path, "# MIR\n", suffix=".mir")

  report = reduce_bug_bundle(
    bundle,
    llvm_root,
    "example",
    tmp_path / "reduced",
    profiles_dir=profiles_dir,
  )

  assert report.status == "unchanged-non-ll"
  assert report.reduced_path.read_text(encoding="utf-8") == "# MIR\n"


def test_missing_bundle_file_fails(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  bundle = tmp_path / "bundle"
  _write(bundle / "classification.json", "{}\n")

  try:
    reduce_bug_bundle(
      bundle,
      llvm_root,
      "example",
      tmp_path / "reduced",
      profiles_dir=profiles_dir,
    )
  except ValueError as exc:
    assert "required bundle file" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_cli_reduce_prints_summary(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  bundle = _make_bundle(tmp_path, _multi_function_text())
  out_dir = tmp_path / "reduced"

  assert (
    main(
      [
        "reduce",
        "--bundle-dir",
        str(bundle),
        "--llvm-root",
        str(llvm_root),
        "--profile",
        "example",
        "--out-dir",
        str(out_dir),
        "--profiles-dir",
        str(profiles_dir),
      ]
    )
    == 0
  )

  result = json.loads(capsys.readouterr().out)
  assert result["status"] == "reduced"
  assert result["accepted_reduction_count"] == 1
  assert (out_dir / "reduction.json").is_file()
