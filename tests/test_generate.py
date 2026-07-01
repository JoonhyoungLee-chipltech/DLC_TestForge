from __future__ import annotations

import json

from dlc_testforge.cli import main
from dlc_testforge.generate import create_workspace


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _make_llvm_root(tmp_path):
  llvm_root = tmp_path / "LLVM"
  _write(
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll",
    """; RUN: llc -mtriple=dlc < %s | FileCheck %s
define i32 @example(i32 %x) {
  ret i32 %x
}
; CHECK: example
""",
  )
  (llvm_root / "docs" / "dlc_spec").mkdir(parents=True)
  (llvm_root / "llvm" / "lib" / "Target" / "DLC").mkdir(parents=True)
  return llvm_root


def _make_profiles_dir(tmp_path):
  profiles_dir = tmp_path / "profiles"
  _write(
    profiles_dir / "example.yaml",
    """name: example
description: Example generation profile.
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


def test_dry_run_creates_workspace_snapshots_and_no_candidates(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  out_dir = tmp_path / "workspace"
  seed_path = llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll"
  original_seed = seed_path.read_text(encoding="utf-8")

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    out_dir,
    dry_run=True,
    profiles_dir=profiles_dir,
  )

  assert (out_dir / "manifest.json").is_file()
  assert (out_dir / "inputs" / "example.ll").read_text(encoding="utf-8") == original_seed
  assert (out_dir / "inputs" / "profile.yaml").is_file()
  assert (out_dir / "inputs" / "test-index.json").is_file()
  assert (out_dir / "inputs" / "spec-index.json").is_file()
  assert (out_dir / "inputs" / "td-index.json").is_file()
  assert list((out_dir / "candidates").iterdir()) == []
  assert (out_dir / "results").is_dir()
  assert (out_dir / "reports").is_dir()
  assert seed_path.read_text(encoding="utf-8") == original_seed

  manifest_json = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
  assert manifest_json == manifest.to_dict()
  assert manifest_json["schema_version"] == 1
  assert manifest_json["profile"] == "example"
  assert manifest_json["seed"] == "llvm/test/CodeGen/DLC/example.ll"
  assert manifest_json["dry_run"] is True
  assert manifest_json["candidate_count"] == 0
  assert manifest_json["inputs"]["seed"] == "inputs/example.ll"
  assert manifest_json["run_id"].endswith("-example")
  assert manifest_json["created_at"].endswith("Z")


def test_seed_outside_profile_fails(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  _write(
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "other.ll",
    "; RUN: llc -mtriple=dlc < %s | FileCheck %s\n",
  )

  try:
    create_workspace(
      llvm_root,
      "example",
      "llvm/test/CodeGen/DLC/other.ll",
      tmp_path / "workspace",
      dry_run=True,
      profiles_dir=profiles_dir,
    )
  except ValueError as exc:
    assert "not allowed by profile" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_missing_seed_fails(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = tmp_path / "profiles"
  _write(
    profiles_dir / "missing.yaml",
    _make_profile_text("missing", "llvm/test/CodeGen/DLC/missing.ll"),
  )

  try:
    create_workspace(
      llvm_root,
      "missing",
      "llvm/test/CodeGen/DLC/missing.ll",
      tmp_path / "workspace",
      dry_run=True,
      profiles_dir=profiles_dir,
    )
  except ValueError as exc:
    assert "seed file not found" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_seed_must_be_relative(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)

  try:
    create_workspace(
      llvm_root,
      "example",
      "/tmp/example.ll",
      tmp_path / "workspace",
      dry_run=True,
      profiles_dir=profiles_dir,
    )
  except ValueError as exc:
    assert "relative path" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_non_empty_output_directory_fails(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  out_dir = tmp_path / "workspace"
  _write(out_dir / "old.txt", "old\n")

  try:
    create_workspace(
      llvm_root,
      "example",
      "llvm/test/CodeGen/DLC/example.ll",
      out_dir,
      dry_run=True,
      profiles_dir=profiles_dir,
    )
  except ValueError as exc:
    assert "not empty" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_non_dry_run_fails(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)

  try:
    create_workspace(
      llvm_root,
      "example",
      "llvm/test/CodeGen/DLC/example.ll",
      tmp_path / "workspace",
      dry_run=False,
      profiles_dir=profiles_dir,
    )
  except ValueError as exc:
    assert "non-dry-run" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_cli_generate_dry_run_outputs_json(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  out_dir = tmp_path / "workspace"

  assert (
    main(
      [
        "generate",
        "--llvm-root",
        str(llvm_root),
        "--profile",
        "example",
        "--seed",
        "llvm/test/CodeGen/DLC/example.ll",
        "--out-dir",
        str(out_dir),
        "--profiles-dir",
        str(profiles_dir),
        "--dry-run",
      ]
    )
    == 0
  )

  captured = capsys.readouterr()
  result = json.loads(captured.out)
  assert result["status"] == "dry-run"
  assert result["profile"] == "example"
  assert result["seed"] == "llvm/test/CodeGen/DLC/example.ll"
  assert result["candidate_count"] == 0
  assert result["manifest"] == str(out_dir / "manifest.json")


def _make_profile_text(name, seed):
  return f"""name: {name}
description: Example generation profile.
seed_selectors:
  paths:
    - {seed}
  features:
    - llc
commands:
  base:
    - "{{llc}} -mtriple=dlc {{input}} -o -"
validation:
  allow_verify_machineinstrs: false
  required_levels:
    - syntax
mutation_axes:
  immediates:
    enabled: true
    values: [0, 1]
bug_scout:
  enabled: true
  classify_crash: true
"""
