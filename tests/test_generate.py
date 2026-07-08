from __future__ import annotations

import json

from dlc_testforge.agent import AgentFullFileCandidate, AgentFullFileProposal
from dlc_testforge.cli import main
from dlc_testforge.generate import (
  RejectedFullFileCandidate,
  _full_file_rejection_reason,
  create_workspace,
)


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


def _write_kernel_usage_index(tmp_path):
  path = tmp_path / "kernel-usage-index.json"
  path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "summary": {"kernel_count": 1},
        "kernels": [
          {
            "name": "custom_edge",
            "source": "dlc_kernels/custom_edge.c",
            "category": "root",
            "dtype_hints": ["f32"],
            "features": [],
            "usage": {
              "dma_calls": [{"line": 3}],
              "memory_spaces": ["HBM", "VMEM"],
              "vector_types": ["float8_128"],
              "intrinsics": ["v_f32_ld_tnsr_st_msk"],
              "constants": [7, 128],
              "relations": [
                {
                  "kind": "address_exponent",
                  "line": 3,
                  "evidence": "addr_exp=7",
                  "reason": "test",
                }
              ],
            },
            "edge_hints": [
              {
                "kind": "addr_exp_boundary",
                "base": 7,
                "values": [6, 7, 8],
                "source": "dlc_kernels/custom_edge.c:3",
                "reason": "test",
              }
            ],
          }
        ],
        "global_edge_hints": [],
      }
    ),
    encoding="utf-8",
  )
  return path


def _full_file_candidate(text):
  return AgentFullFileCandidate(
    filename="candidate.ll",
    text=text,
    rationale="test candidate",
    intended_stress="test stress",
  )


def test_full_file_quality_gate_accepts_candidate():
  candidate = _full_file_candidate(
    """; RUN: llc -mtriple=dlc < %s | FileCheck %s
define void @candidate() {
  ret void
}
"""
  )

  assert (
    _full_file_rejection_reason(
      candidate,
      seed_text="; RUN: llc < %s\n",
      seen_texts=set(),
    )
    is None
  )


def test_full_file_quality_gate_rejects_empty_text():
  assert _full_file_rejection_reason(
    _full_file_candidate("  \n"),
    seed_text="; RUN: llc < %s\n",
    seen_texts=set(),
  ) == "empty candidate text"


def test_full_file_quality_gate_rejects_markdown_fences():
  assert _full_file_rejection_reason(
    _full_file_candidate(
      """```llvm
; RUN: llc < %s
define void @candidate() { ret void }
```"""
    ),
    seed_text="; RUN: llc < %s\n",
    seen_texts=set(),
  ) == "candidate text contains Markdown fences"


def test_full_file_quality_gate_rejects_seed_duplicate():
  seed_text = """; RUN: llc < %s
define void @seed() {
  ret void
}
"""

  assert _full_file_rejection_reason(
    _full_file_candidate(seed_text),
    seed_text=seed_text,
    seen_texts=set(),
  ) == "candidate is identical to seed"


def test_full_file_quality_gate_rejects_duplicate_candidate_text():
  text = """; RUN: llc < %s
define void @candidate() {
  ret void
}
"""

  assert _full_file_rejection_reason(
    _full_file_candidate(text),
    seed_text="; RUN: llc < %s\ndefine void @seed() { ret void }\n",
    seen_texts={text.strip()},
  ) == "duplicate candidate text"


def test_full_file_quality_gate_does_not_mutate_seen_texts():
  text = """; RUN: llc < %s
define void @candidate() {
  ret void
}
"""
  seen_texts = set()

  assert _full_file_rejection_reason(
    _full_file_candidate(text),
    seed_text="; RUN: llc < %s\ndefine void @seed() { ret void }\n",
    seen_texts=seen_texts,
  ) is None
  assert seen_texts == set()


def test_full_file_quality_gate_rejects_missing_run_line():
  assert _full_file_rejection_reason(
    _full_file_candidate("define void @candidate() { ret void }\n"),
    seed_text="; RUN: llc < %s\n",
    seen_texts=set(),
  ) == "missing RUN line"


def test_full_file_quality_gate_rejects_oversized_candidate():
  candidate_text = (
    "; RUN: llc < %s\n"
    "define void @candidate() { ret void }\n"
    + ("; filler\n" * 3000)
  )

  assert _full_file_rejection_reason(
    _full_file_candidate(candidate_text),
    seed_text="; RUN: llc < %s\ndefine void @seed() { ret void }\n",
    seen_texts=set(),
  ) == "candidate is too large"


def test_full_file_quality_gate_rejects_missing_define_line():
  assert _full_file_rejection_reason(
    _full_file_candidate("; RUN: llc < %s\n; CHECK: nothing\n"),
    seed_text="; RUN: llc < %s\n",
    seen_texts=set(),
  ) == "missing define line"


def test_full_file_quality_gate_rejects_removed_target_triple():
  seed_text = """target triple = "dlc"
; RUN: llc < %s
define void @seed() {
  ret void
}
"""

  assert _full_file_rejection_reason(
    _full_file_candidate(
      """; RUN: llc < %s
define void @candidate() {
  ret void
}
"""
    ),
    seed_text=seed_text,
    seen_texts=set(),
  ) == "candidate removed seed target triple"


def test_full_file_quality_gate_allows_missing_target_triple_when_seed_has_none():
  candidate = _full_file_candidate(
    """; RUN: llc < %s
define void @candidate() {
  ret void
}
"""
  )

  assert _full_file_rejection_reason(
    candidate,
    seed_text="; RUN: llc < %s\ndefine void @seed() { ret void }\n",
    seen_texts=set(),
  ) is None


def test_rejected_full_file_candidate_to_dict_preserves_fields():
  candidate = _full_file_candidate("bad\n").to_dict()
  rejection = RejectedFullFileCandidate(
    index=2,
    candidate=candidate,
    reason="missing RUN line",
  )

  assert rejection.to_dict() == {
    "index": 2,
    "candidate": candidate,
    "reason": "missing RUN line",
  }


def test_agent_full_file_dry_run_writes_context_only(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  out_dir = tmp_path / "workspace"

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    out_dir,
    dry_run=True,
    profiles_dir=profiles_dir,
    mode="agent-full-file",
  )

  assert manifest.mode == "agent-full-file"
  assert manifest.candidate_count == 0
  assert manifest.inputs["full_file_agent_context"] == (
    "inputs/full-file-agent-context.json"
  )
  assert "agent_context" not in manifest.inputs
  assert "agent_proposal" not in manifest.inputs
  assert "full_file_agent_proposal" not in manifest.inputs
  assert (out_dir / "inputs" / "full-file-agent-context.json").is_file()
  assert not (out_dir / "inputs" / "full-file-agent-proposal.json").exists()
  assert not (out_dir / "results" / "full-file-agent-rejections.json").exists()
  assert list((out_dir / "candidates").iterdir()) == []


def test_agent_full_file_offline_proposal_writes_candidate_and_metadata(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  proposal_path = tmp_path / "full-file-proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "candidates": [
          {
            "filename": "suggested-name.ll",
            "text": (
              "; RUN: llc -mtriple=dlc < %s | FileCheck %s\n"
              "define void @candidate() {\n"
              "  ret void\n"
              "}\n\n"
            ),
            "rationale": "write a complete candidate",
            "intended_stress": "full-file workspace integration",
            "evidence_tags": ["addr_exp_boundary"],
            "source_evidence": "kernel evidence",
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
    mode="agent-full-file",
    agent_full_file_proposal=proposal_path,
  )

  assert manifest.candidate_count == 1
  assert manifest.inputs["full_file_agent_context"] == (
    "inputs/full-file-agent-context.json"
  )
  assert manifest.inputs["full_file_agent_proposal"] == (
    "inputs/full-file-agent-proposal.json"
  )
  candidate = manifest.candidates[0]
  assert candidate.path == "candidates/candidate-0001.ll"
  assert candidate.mutation_axis == "agent_full_file"
  assert candidate.source_value == 0
  assert candidate.new_value == 0
  assert candidate.line == 1
  assert candidate.comment == "; DLC-MUTATION: profile=example mode=agent-full-file"
  assert candidate.suggested_filename == "suggested-name.ll"
  assert candidate.rationale == "write a complete candidate"
  assert candidate.intended_stress == "full-file workspace integration"
  assert candidate.evidence_tags == ["addr_exp_boundary"]
  assert candidate.source_evidence == "kernel evidence"
  candidate_text = (
    tmp_path / "workspace" / "candidates" / "candidate-0001.ll"
  ).read_text(encoding="utf-8")
  assert candidate_text.endswith("\n")
  assert not candidate_text.endswith("\n\n")
  manifest_json = json.loads(
    (tmp_path / "workspace" / "manifest.json").read_text(encoding="utf-8")
  )
  assert manifest_json["candidates"][0]["suggested_filename"] == "suggested-name.ll"
  assert manifest_json["candidates"][0]["intended_stress"] == (
    "full-file workspace integration"
  )


def test_agent_full_file_records_rejections(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  proposal_path = tmp_path / "full-file-proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "candidates": [
          {
            "filename": "missing-run.ll",
            "text": "define void @candidate() { ret void }\n",
            "rationale": "missing run",
            "intended_stress": "quality gate",
          },
          {
            "filename": "valid.ll",
            "text": (
              "; RUN: llc -mtriple=dlc < %s | FileCheck %s\n"
              "define void @candidate() { ret void }\n"
            ),
            "rationale": "valid",
            "intended_stress": "accepted candidate",
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
    mode="agent-full-file",
    agent_full_file_proposal=proposal_path,
  )

  assert manifest.candidate_count == 1
  rejections = json.loads(
    (
      tmp_path / "workspace" / "results" / "full-file-agent-rejections.json"
    ).read_text(encoding="utf-8")
  )
  assert rejections["rejection_count"] == 1
  assert rejections["rejections"][0]["index"] == 0
  assert rejections["rejections"][0]["reason"] == "missing RUN line"


def test_agent_full_file_rejects_seed_duplicate(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  seed_text = (
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "example.ll"
  ).read_text(encoding="utf-8")
  proposal_path = tmp_path / "full-file-proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "candidates": [
          {
            "filename": "same-as-seed.ll",
            "text": seed_text,
            "rationale": "duplicate seed",
            "intended_stress": "quality gate",
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
    mode="agent-full-file",
    agent_full_file_proposal=proposal_path,
  )

  assert manifest.candidate_count == 0
  rejections = json.loads(
    (
      tmp_path / "workspace" / "results" / "full-file-agent-rejections.json"
    ).read_text(encoding="utf-8")
  )
  assert rejections["rejections"][0]["reason"] == "candidate is identical to seed"


def test_agent_full_file_proposal_seed_profile_mismatch_fails(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  proposal_path = tmp_path / "full-file-proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/other.ll",
        "profile": "example",
        "candidates": [],
      }
    ),
    encoding="utf-8",
  )

  try:
    create_workspace(
      llvm_root,
      "example",
      "llvm/test/CodeGen/DLC/example.ll",
      tmp_path / "workspace",
      dry_run=False,
      profiles_dir=profiles_dir,
      mode="agent-full-file",
      agent_full_file_proposal=proposal_path,
    )
  except ValueError as exc:
    assert "does not match requested seed" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_agent_full_file_live_request_path_writes_artifacts(tmp_path, monkeypatch):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)

  def fake_request(context, *, model, endpoint, max_candidates):
    assert context["seed"]["path"] == "llvm/test/CodeGen/DLC/example.ll"
    assert model == "test-model"
    assert endpoint == "https://example.test/v1"
    assert max_candidates == 2
    return AgentFullFileProposal(
      seed="llvm/test/CodeGen/DLC/example.ll",
      profile="example",
      candidates=[
        AgentFullFileCandidate(
          filename="live.ll",
          text=(
            "; RUN: llc -mtriple=dlc < %s | FileCheck %s\n"
            "define void @candidate() { ret void }\n"
          ),
          rationale="live request candidate",
          intended_stress="live request",
        )
      ],
    )

  monkeypatch.setattr(
    "dlc_testforge.generate.request_agent_full_file_proposal",
    fake_request,
  )

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    tmp_path / "workspace",
    dry_run=False,
    profiles_dir=profiles_dir,
    mode="agent-full-file",
    agent_model="test-model",
    agent_endpoint="https://example.test/v1",
    max_candidates=2,
  )

  assert manifest.candidate_count == 1
  assert (tmp_path / "workspace" / "inputs" / "full-file-agent-proposal.json").is_file()
  assert (tmp_path / "workspace" / "candidates" / "candidate-0001.ll").is_file()


def test_agent_full_file_context_includes_kernel_evidence_when_provided(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  kernel_usage_index = _write_kernel_usage_index(tmp_path)

  without_evidence = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    tmp_path / "workspace-no-evidence",
    dry_run=True,
    profiles_dir=profiles_dir,
    mode="agent-full-file",
  )
  with_evidence = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    tmp_path / "workspace-with-evidence",
    dry_run=True,
    profiles_dir=profiles_dir,
    mode="agent-full-file",
    kernel_usage_index=kernel_usage_index,
  )

  assert "kernel_usage_index" not in without_evidence.inputs
  assert with_evidence.inputs["kernel_usage_index"] == "inputs/kernel-usage-index.json"
  base_context = json.loads(
    (
      tmp_path
      / "workspace-no-evidence"
      / "inputs"
      / "full-file-agent-context.json"
    ).read_text(encoding="utf-8")
  )
  context = json.loads(
    (
      tmp_path
      / "workspace-with-evidence"
      / "inputs"
      / "full-file-agent-context.json"
    ).read_text(encoding="utf-8")
  )
  assert "kernel_usage_evidence" not in base_context
  assert context["kernel_usage_evidence"]["selection"]["selected_count"] == 1


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
            "evidence_tags": ["addr_exp_boundary", "dma_length_boundary"],
            "source_evidence": (
              "selected kernel evidence points at address boundary behavior"
            ),
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
  agent_proposal = json.loads(
    (tmp_path / "workspace" / "inputs" / "agent-proposal.json").read_text(
      encoding="utf-8"
    )
  )
  saved_mutation = agent_proposal["proposed_mutations"][0]
  assert saved_mutation["evidence_tags"] == [
    "addr_exp_boundary",
    "dma_length_boundary",
  ]
  assert (
    saved_mutation["source_evidence"]
    == "selected kernel evidence points at address boundary behavior"
  )
  context = json.loads(
    (tmp_path / "workspace" / "inputs" / "agent-context.json").read_text(
      encoding="utf-8"
    )
  )
  assert "kernel_usage_evidence" not in context

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


def test_manual_mode_prioritizes_kernel_informed_candidates(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  kernel_usage_index = _write_kernel_usage_index(tmp_path)
  out_dir = tmp_path / "workspace"

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    out_dir,
    dry_run=False,
    profiles_dir=profiles_dir,
    max_candidates=1,
    kernel_usage_index=kernel_usage_index,
  )

  assert manifest.candidate_count == 1
  candidate = manifest.candidates[0]
  assert candidate.mutation_axis == "addr_exp_boundary"
  assert candidate.source_value == 7
  assert candidate.new_value == 6
  assert candidate.evidence_tags == ["addr_exp_boundary"]
  assert candidate.source_evidence == "dlc_kernels/custom_edge.c:3: test"
  candidate_text = (out_dir / "candidates" / "candidate-0001.ll").read_text(
    encoding="utf-8"
  )
  assert "mode=kernel-informed axis=addr_exp_boundary" in candidate_text
  assert "%shl = shl i32 %x, 6" in candidate_text
  manifest_json = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
  assert manifest_json["candidates"][0]["evidence_tags"] == ["addr_exp_boundary"]
  assert (
    manifest_json["candidates"][0]["source_evidence"]
    == "dlc_kernels/custom_edge.c:3: test"
  )


def test_kernel_informed_generation_skips_malformed_and_duplicate_hints(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  out_dir = tmp_path / "workspace"
  kernel_usage_index = tmp_path / "kernel-usage-index.json"
  kernel_usage_index.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "summary": {"kernel_count": 1},
        "kernels": [
          {
            "name": "custom_edge",
            "source": "dlc_kernels/custom_edge.c",
            "usage": {},
            "edge_hints": [
              {"kind": "unsupported", "base": 7, "values": [6]},
              {"kind": "addr_exp_boundary", "base": "7", "values": [6]},
              {"kind": "addr_exp_boundary", "base": 7, "values": "bad"},
              {
                "kind": "addr_exp_boundary",
                "base": 7,
                "values": [8],
                "source": "dlc_kernels/custom_edge.c:1",
              },
              {
                "kind": "addr_exp_boundary",
                "base": 7,
                "values": [6],
                "source": "dlc_kernels/custom_edge.c:2",
              },
            ],
          }
        ],
        "global_edge_hints": [
          {
            "kind": "addr_exp_boundary",
            "base": 7,
            "values": [6],
            "source": "dlc_kernels/custom_edge.c:2",
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
    out_dir,
    dry_run=False,
    profiles_dir=profiles_dir,
    max_candidates=3,
    kernel_usage_index=kernel_usage_index,
  )

  kernel_candidates = [
    candidate for candidate in manifest.candidates
    if candidate.mutation_axis == "addr_exp_boundary"
  ]
  assert len(kernel_candidates) == 1
  assert kernel_candidates[0].new_value == 6
  assert all(candidate.new_value != 8 for candidate in manifest.candidates)


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


def test_agent_mode_applies_kernel_informed_axis(tmp_path):
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
            "axis": "addr_exp_boundary",
            "location_hint": "@example",
            "old_value": 7,
            "new_value": 6,
            "rationale": "apply kernel-informed address exponent edge",
            "evidence_tags": ["addr_exp_boundary"],
            "source_evidence": "dlc_kernels/custom_edge.c:3",
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
  assert manifest.candidates[0].mutation_axis == "addr_exp_boundary"
  candidate_text = (tmp_path / "workspace" / "candidates" / "candidate-0001.ll").read_text(
    encoding="utf-8"
  )
  assert "mode=agent axis=addr_exp_boundary edits=7->6" in candidate_text
  assert "%shl = shl i32 %x, 6" in candidate_text


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


def test_cli_generate_help_includes_agent_full_file(capsys):
  try:
    main(["generate", "--help"])
  except SystemExit as exc:
    assert exc.code == 0
  else:
    raise AssertionError("expected SystemExit")

  captured = capsys.readouterr()
  assert "agent-full-file" in captured.out
  assert "--agent-full-file-proposal" in captured.out


def test_cli_generate_agent_full_file_outputs_json(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  out_dir = tmp_path / "workspace"
  proposal_path = tmp_path / "full-file-proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "candidates": [
          {
            "filename": "suggested.ll",
            "text": (
              "; RUN: llc -mtriple=dlc < %s | FileCheck %s\n"
              "define void @candidate() { ret void }\n"
            ),
            "rationale": "CLI full-file proposal",
            "intended_stress": "CLI integration",
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
        "agent-full-file",
        "--agent-full-file-proposal",
        str(proposal_path),
      ]
    )
    == 0
  )

  captured = capsys.readouterr()
  result = json.loads(captured.out)
  assert result["status"] == "generated"
  assert result["candidate_count"] == 1
  assert (out_dir / "candidates" / "candidate-0001.ll").is_file()
  assert (out_dir / "inputs" / "full-file-agent-context.json").is_file()
  assert (out_dir / "inputs" / "full-file-agent-proposal.json").is_file()
  assert (out_dir / "results" / "full-file-agent-rejections.json").is_file()


def test_cli_generate_agent_full_file_invalid_proposal_exits_nonzero(
  tmp_path, capsys
):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  proposal_path = tmp_path / "full-file-proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/other.ll",
        "profile": "example",
        "candidates": [],
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
        str(tmp_path / "workspace"),
        "--profiles-dir",
        str(profiles_dir),
        "--mode",
        "agent-full-file",
        "--agent-full-file-proposal",
        str(proposal_path),
      ]
    )
    == 2
  )

  assert "error:" in capsys.readouterr().err


def test_agent_mode_includes_kernel_usage_evidence_when_provided(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  kernel_usage_index = _write_kernel_usage_index(tmp_path)
  out_dir = tmp_path / "workspace"

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    out_dir,
    dry_run=True,
    profiles_dir=profiles_dir,
    mode="agent",
    kernel_usage_index=kernel_usage_index,
  )

  assert manifest.inputs["kernel_usage_index"] == "inputs/kernel-usage-index.json"
  assert (out_dir / "inputs" / "kernel-usage-index.json").is_file()
  context = json.loads(
    (out_dir / "inputs" / "agent-context.json").read_text(encoding="utf-8")
  )
  assert context["kernel_usage_evidence"]["selection"]["selected_count"] == 1
  assert context["kernel_usage_evidence"]["kernels"][0]["name"] == "custom_edge"


def test_manual_mode_copies_kernel_usage_index_without_agent_context(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  kernel_usage_index = _write_kernel_usage_index(tmp_path)
  out_dir = tmp_path / "workspace"

  manifest = create_workspace(
    llvm_root,
    "example",
    "llvm/test/CodeGen/DLC/example.ll",
    out_dir,
    dry_run=True,
    profiles_dir=profiles_dir,
    kernel_usage_index=kernel_usage_index,
  )

  assert manifest.inputs["kernel_usage_index"] == "inputs/kernel-usage-index.json"
  assert (out_dir / "inputs" / "kernel-usage-index.json").is_file()
  assert not (out_dir / "inputs" / "agent-context.json").exists()


def test_cli_generate_accepts_kernel_usage_index(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  kernel_usage_index = _write_kernel_usage_index(tmp_path)
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
        "--kernel-usage-index",
        str(kernel_usage_index),
      ]
    )
    == 0
  )

  json.loads(capsys.readouterr().out)
  manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
  assert manifest["inputs"]["kernel_usage_index"] == "inputs/kernel-usage-index.json"


def test_invalid_kernel_usage_index_json_fails(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  bad_index = tmp_path / "bad-kernel-index.json"
  bad_index.write_text("{not json\n", encoding="utf-8")

  try:
    create_workspace(
      llvm_root,
      "example",
      "llvm/test/CodeGen/DLC/example.ll",
      tmp_path / "workspace",
      dry_run=True,
      profiles_dir=profiles_dir,
      kernel_usage_index=bad_index,
    )
  except ValueError as exc:
    assert "Expecting property name" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_cli_generate_invalid_kernel_usage_index_json_exits_nonzero(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path)
  profiles_dir = _make_profiles_dir(tmp_path)
  bad_index = tmp_path / "bad-kernel-index.json"
  bad_index.write_text("{not json\n", encoding="utf-8")

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
        str(tmp_path / "workspace"),
        "--profiles-dir",
        str(profiles_dir),
        "--dry-run",
        "--kernel-usage-index",
        str(bad_index),
      ]
    )
    == 2
  )

  assert "error:" in capsys.readouterr().err


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
