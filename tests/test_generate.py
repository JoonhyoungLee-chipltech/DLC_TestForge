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
  %shl = shl i32 %x, 7
  %add = add i32 %shl, 5
  ret i32 %add
}
; CHECK: S_ADD %{{[0-9]+}}, 5
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
    values: [0, 1, 6]
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


def test_non_dry_run_generates_deterministic_candidates(tmp_path):
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
    dry_run=False,
    profiles_dir=profiles_dir,
    max_candidates=3,
  )

  assert manifest.dry_run is False
  assert manifest.candidate_count == 3
  assert [candidate.path for candidate in manifest.candidates] == [
    "candidates/candidate-0001.ll",
    "candidates/candidate-0002.ll",
    "candidates/candidate-0003.ll",
  ]
  assert [candidate.new_value for candidate in manifest.candidates] == [0, 1, 6]
  assert [candidate.mutation_axis for candidate in manifest.candidates] == [
    "shift_amount_boundary",
    "shift_amount_boundary",
    "shift_amount_boundary",
  ]
  candidate_text = (out_dir / "candidates" / "candidate-0001.ll").read_text(
    encoding="utf-8"
  )
  assert candidate_text.count("DLC-MUTATION:") == 1
  assert "source_value=7 new_value=0" in candidate_text
  assert "%shl = shl i32 %x, 0" in candidate_text
  assert "; RUN: llc -mtriple=dlc < %s | FileCheck %s" in candidate_text
  assert "; CHECK: S_ADD %{{[0-9]+}}, 5" in candidate_text
  assert seed_path.read_text(encoding="utf-8") == original_seed

  manifest_json = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
  assert manifest_json["dry_run"] is False
  assert manifest_json["candidate_count"] == 3
  assert manifest_json["candidates"] == [
    candidate.to_dict() for candidate in manifest.candidates
  ]


def test_non_dry_run_respects_max_candidates(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    tmp_path / "workspace",
    dry_run=False,
    profiles_dir=profiles_dir,
    max_candidates=1,
  )

  assert manifest.candidate_count == 1
  candidate_names = sorted(
    path.name for path in (tmp_path / "workspace" / "candidates").iterdir()
  )
  assert candidate_names == ["candidate-0001.ll"]


def test_invalid_max_candidates_fails(tmp_path):
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
      max_candidates=0,
    )
  except ValueError as exc:
    assert "max-candidates" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_non_dry_run_mir_seed_creates_no_candidates(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = tmp_path / "profiles"
  _write(
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.mir",
    "# RUN: llc -run-pass=legalizer %s -o - | FileCheck %s\n"
    "# CHECK: G_ADD\n"
    "%0:_(s32) = COPY $r12\n",
  )
  _write(
    profiles_dir / "mir.yaml",
    _make_profile_text("mir", "llvm/test/CodeGen/DLC/example.mir"),
  )

  manifest = create_workspace(
    llvm_root,
    "mir",
    "llvm/test/CodeGen/DLC/example.mir",
    tmp_path / "workspace",
    dry_run=False,
    profiles_dir=profiles_dir,
  )

  assert manifest.candidate_count == 0
  assert list((tmp_path / "workspace" / "candidates").iterdir()) == []


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


def test_cli_generate_non_dry_run_outputs_json(tmp_path, capsys):
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
        "--max-candidates",
        "2",
      ]
    )
    == 0
  )

  captured = capsys.readouterr()
  result = json.loads(captured.out)
  assert result["status"] == "generated"
  assert result["candidate_count"] == 2
  assert (out_dir / "candidates" / "candidate-0001.ll").is_file()


def test_agent_mode_uses_proposal_file_and_records_rejections(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  proposal_path = tmp_path / "proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "proposed_mutations": [
          {
            "axis": "shift_amount_boundary",
            "location_hint": "example",
            "old_value": 7,
            "new_value": 6,
            "rationale": "exercise the adjacent lower shift boundary",
          },
          {
            "axis": "shift_amount_boundary",
            "location_hint": "example",
            "old_value": 99,
            "new_value": 1,
            "rationale": "cannot be applied because old value is absent",
          },
          {
            "axis": "address_shapes",
            "location_hint": "example",
            "old_value": 7,
            "new_value": 1,
            "rationale": "unsupported by current agent generator",
          },
        ],
      }
    ),
    encoding="utf-8",
  )

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    tmp_path / "workspace",
    dry_run=False,
    profiles_dir=profiles_dir,
    max_candidates=3,
    mode="agent",
    agent_proposal=proposal_path,
  )

  assert manifest.mode == "agent"
  assert manifest.candidate_count == 1
  assert manifest.inputs["agent_context"] == "inputs/agent-context.json"
  assert manifest.inputs["agent_proposal"] == "inputs/agent-proposal.json"
  candidate_text = (tmp_path / "workspace" / "candidates" / "candidate-0001.ll").read_text(
    encoding="utf-8"
  )
  assert "mode=agent axis=shift_amount_boundary edits=7->6" in candidate_text
  assert "%shl = shl i32 %x, 6" in candidate_text
  assert manifest.candidates[0].rationale == "exercise the adjacent lower shift boundary"

  rejections = json.loads(
    (tmp_path / "workspace" / "results" / "agent-rejections.json").read_text(
      encoding="utf-8"
    )
  )
  assert rejections["rejection_count"] == 2
  assert [entry["reason"] for entry in rejections["rejections"]] == [
    "unsupported mutation axis for current agent generator: address_shapes",
    "one or more old_value entries were not found on mutable seed lines for the requested axis",
  ]


def test_agent_mode_applies_grouped_edits_to_one_candidate(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  proposal_path = tmp_path / "proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "proposed_mutations": [
          {
            "axis": "immediates",
            "location_hint": "@example",
            "edits": [
              {"old_value": 7, "new_value": 6},
              {"old_value": 5, "new_value": 1},
            ],
            "rationale": "mutate related immediates together",
          }
        ],
      }
    ),
    encoding="utf-8",
  )

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    tmp_path / "workspace",
    dry_run=False,
    profiles_dir=profiles_dir,
    mode="agent",
    agent_proposal=proposal_path,
  )

  assert manifest.candidate_count == 1
  candidate_text = (tmp_path / "workspace" / "candidates" / "candidate-0001.ll").read_text(
    encoding="utf-8"
  )
  assert "edits=7->6,5->1" in candidate_text
  assert "%shl = shl i32 %x, 6" in candidate_text
  assert "%add = add i32 %shl, 1" in candidate_text
  assert manifest.candidates[0].edits == [
    {"old_value": 7, "new_value": 6, "line": 3},
    {"old_value": 5, "new_value": 1, "line": 4},
  ]


def test_cli_generate_agent_mode_outputs_json(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  out_dir = tmp_path / "workspace"
  proposal_path = tmp_path / "proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "proposed_mutations": [
          {
            "axis": "immediates",
            "location_hint": "example",
            "old_value": 5,
            "new_value": 1,
            "rationale": "exercise small add immediate",
          }
        ],
      }
    ),
    encoding="utf-8",
  )

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
        "--mode",
        "agent",
        "--agent-proposal",
        str(proposal_path),
      ]
    )
    == 0
  )

  captured = capsys.readouterr()
  result = json.loads(captured.out)
  assert result["status"] == "generated"
  assert result["candidate_count"] == 1
  assert (out_dir / "inputs" / "agent-context.json").is_file()
  assert (out_dir / "results" / "agent-rejections.json").is_file()


def test_agent_mode_without_proposal_requires_model(tmp_path, monkeypatch):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  monkeypatch.setattr("dlc_testforge.agent.CODEX_AUTH_PATH", tmp_path / "missing-auth.json")
  monkeypatch.setattr(
    "dlc_testforge.agent.CODEX_CONFIG_PATH", tmp_path / "missing-config.toml"
  )

  try:
    create_workspace(
      llvm_root,
      "example",
      "llvm/test/CodeGen/DLC/example.ll",
      tmp_path / "workspace",
      dry_run=False,
      profiles_dir=profiles_dir,
      mode="agent",
    )
  except ValueError as exc:
    assert "requires --agent-model" in str(exc)
  else:
    raise AssertionError("expected ValueError")


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
    values: [0, 1, 6]
bug_scout:
  enabled: true
  classify_crash: true
"""
