from __future__ import annotations

import json

from dlc_testforge.agent import (
  build_agent_context,
  filter_agent_mutations,
  load_agent_full_file_proposal,
  parse_agent_full_file_proposal,
  parse_agent_proposal,
  request_agent_proposal,
  select_kernel_evidence,
)
from dlc_testforge.profiles import get_profile
from tests.test_generate import _make_profiles_dir


def _kernel_record(
  name,
  *,
  source,
  edge_kinds=(),
  relation_kinds=(),
  dma=False,
  vector_types=(),
  intrinsics=(),
  dtype_hints=(),
  constants=(),
  memory_spaces=(),
  extra=None,
):
  record = {
    "name": name,
    "source": source,
    "category": "test",
    "dtype_hints": list(dtype_hints),
    "features": [],
    "usage": {
      "dma_calls": [{"line": 1}] if dma else [],
      "sync_calls": [],
      "memory_spaces": list(memory_spaces),
      "vector_types": list(vector_types),
      "intrinsics": list(intrinsics),
      "constants": list(constants),
      "relations": [
        {"kind": kind, "line": index + 1, "evidence": kind, "reason": "test"}
        for index, kind in enumerate(relation_kinds)
      ],
    },
    "edge_hints": [
      {
        "kind": kind,
        "base": index + 1,
        "values": [index, index + 1, index + 2],
        "source": f"{source}:{index + 1}",
        "reason": "test",
      }
      for index, kind in enumerate(edge_kinds)
    ],
  }
  if extra:
    record.update(extra)
  return record


def test_parse_agent_full_file_proposal_preserves_valid_candidate(tmp_path):
  proposal_path = tmp_path / "proposal.json"
  proposal_path.write_text(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "candidates": [
          {
            "filename": "example_agent_0.ll",
            "text": "define void @example() {\n  ret void\n}\n",
            "rationale": "keep the IR minimal while exercising the path",
            "intended_stress": "baseline parser acceptance",
            "evidence_tags": ["addr_exp_boundary"],
            "source_evidence": "kernel evidence mentions address boundaries",
          },
          {
            "filename": "example_agent_1.ll",
            "text": "define void @example2() {\n  ret void\n}\n",
            "rationale": "empty metadata should be omitted",
            "intended_stress": "metadata serialization",
            "evidence_tags": [],
            "source_evidence": "",
          },
        ],
      }
    ),
    encoding="utf-8",
  )

  proposal = load_agent_full_file_proposal(
    proposal_path,
    seed_relative="llvm/test/CodeGen/DLC/example.ll",
    profile_name="example",
  )

  assert proposal.seed == "llvm/test/CodeGen/DLC/example.ll"
  assert proposal.profile == "example"
  assert proposal.candidates[0].filename == "example_agent_0.ll"
  assert proposal.candidates[0].evidence_tags == ["addr_exp_boundary"]
  assert proposal.to_dict() == {
    "seed": "llvm/test/CodeGen/DLC/example.ll",
    "profile": "example",
    "candidates": [
      {
        "filename": "example_agent_0.ll",
        "text": "define void @example() {\n  ret void\n}\n",
        "rationale": "keep the IR minimal while exercising the path",
        "intended_stress": "baseline parser acceptance",
        "evidence_tags": ["addr_exp_boundary"],
        "source_evidence": "kernel evidence mentions address boundaries",
      },
      {
        "filename": "example_agent_1.ll",
        "text": "define void @example2() {\n  ret void\n}\n",
        "rationale": "empty metadata should be omitted",
        "intended_stress": "metadata serialization",
      },
    ],
  }


def test_parse_agent_full_file_proposal_accepts_json_fence():
  proposal = parse_agent_full_file_proposal(
    """```json
{
  "seed": "llvm/test/CodeGen/DLC/example.ll",
  "profile": "example",
  "candidates": [
    {
      "filename": "example_agent_0.ll",
      "text": "define void @example() { ret void }\\n",
      "rationale": "exercise full-file proposal parsing",
      "intended_stress": "JSON fence handling"
    }
  ]
}
```"""
  )

  assert proposal.candidates[0].filename == "example_agent_0.ll"
  assert proposal.candidates[0].text == "define void @example() { ret void }\n"


def test_parse_agent_full_file_proposal_rejects_seed_and_profile_mismatch():
  base = {
    "seed": "llvm/test/CodeGen/DLC/example.ll",
    "profile": "example",
    "candidates": [
      {
        "filename": "example_agent_0.ll",
        "text": "define void @example() { ret void }\n",
        "rationale": "valid candidate",
        "intended_stress": "valid candidate",
      }
    ],
  }

  try:
    parse_agent_full_file_proposal(
      json.dumps(base),
      seed_relative="llvm/test/CodeGen/DLC/other.ll",
      profile_name="example",
    )
  except ValueError as exc:
    assert "does not match requested seed" in str(exc)
  else:
    raise AssertionError("expected ValueError")

  try:
    parse_agent_full_file_proposal(
      json.dumps(base),
      seed_relative="llvm/test/CodeGen/DLC/example.ll",
      profile_name="other",
    )
  except ValueError as exc:
    assert "does not match requested profile" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_parse_agent_full_file_proposal_rejects_unsafe_filename():
  invalid_filenames = [
    "../bad.ll",
    "bad/seed.ll",
    "bad\\seed.ll",
    "/tmp/bad.ll",
    "bad.txt",
    "bad..ll",
  ]

  for filename in invalid_filenames:
    try:
      parse_agent_full_file_proposal(
        json.dumps(
          {
            "seed": "llvm/test/CodeGen/DLC/example.ll",
            "profile": "example",
            "candidates": [
              {
                "filename": filename,
                "text": "define void @example() { ret void }\n",
                "rationale": "valid candidate",
                "intended_stress": "valid candidate",
              }
            ],
          }
        )
      )
    except ValueError as exc:
      assert "filename" in str(exc)
    else:
      raise AssertionError("expected ValueError")


def test_parse_agent_full_file_proposal_rejects_required_candidate_fields():
  base_candidate = {
    "filename": "example_agent_0.ll",
    "text": "define void @example() { ret void }\n",
    "rationale": "valid candidate",
    "intended_stress": "valid candidate",
  }
  invalid_fields = [
    ("filename", ""),
    ("filename", 7),
    ("text", ""),
    ("text", 7),
    ("text", "```llvm\ndefine void @example() { ret void }\n```"),
    ("rationale", ""),
    ("rationale", 7),
    ("intended_stress", ""),
    ("intended_stress", 7),
  ]

  for field, value in invalid_fields:
    candidate = {**base_candidate, field: value}
    try:
      parse_agent_full_file_proposal(
        json.dumps(
          {
            "seed": "llvm/test/CodeGen/DLC/example.ll",
            "profile": "example",
            "candidates": [candidate],
          }
        )
      )
    except ValueError as exc:
      assert field in str(exc)
    else:
      raise AssertionError("expected ValueError")


def test_parse_agent_full_file_proposal_rejects_malformed_metadata():
  base_candidate = {
    "filename": "example_agent_0.ll",
    "text": "define void @example() { ret void }\n",
    "rationale": "valid candidate",
    "intended_stress": "valid candidate",
  }
  invalid_metadata = [
    {"evidence_tags": "addr_exp_boundary"},
    {"evidence_tags": ["addr_exp_boundary", 7]},
    {"evidence_tags": {"kind": "addr_exp_boundary"}},
    {"source_evidence": ["not", "a", "string"]},
  ]

  for metadata in invalid_metadata:
    candidate = {**base_candidate, **metadata}
    try:
      parse_agent_full_file_proposal(
        json.dumps(
          {
            "seed": "llvm/test/CodeGen/DLC/example.ll",
            "profile": "example",
            "candidates": [candidate],
          }
        )
      )
    except ValueError as exc:
      assert "evidence" in str(exc)
    else:
      raise AssertionError("expected ValueError")


def test_parse_agent_full_file_proposal_rejects_duplicate_text():
  try:
    parse_agent_full_file_proposal(
      json.dumps(
        {
          "seed": "llvm/test/CodeGen/DLC/example.ll",
          "profile": "example",
          "candidates": [
            {
              "filename": "example_agent_0.ll",
              "text": "define void @example() { ret void }\n",
              "rationale": "first candidate",
              "intended_stress": "first candidate",
            },
            {
              "filename": "example_agent_1.ll",
              "text": "define void @example() { ret void }\n",
              "rationale": "duplicate candidate",
              "intended_stress": "duplicate candidate",
            },
          ],
        }
      )
    )
  except ValueError as exc:
    assert "duplicates" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_parse_agent_full_file_proposal_caps_after_validation():
  proposal = parse_agent_full_file_proposal(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "candidates": [
          {
            "filename": "example_agent_0.ll",
            "text": "define void @example0() { ret void }\n",
            "rationale": "first candidate",
            "intended_stress": "first candidate",
          },
          {
            "filename": "example_agent_1.ll",
            "text": "define void @example1() { ret void }\n",
            "rationale": "second candidate",
            "intended_stress": "second candidate",
          },
        ],
      }
    ),
    max_candidates=1,
  )

  assert [candidate.filename for candidate in proposal.candidates] == [
    "example_agent_0.ll"
  ]

  try:
    parse_agent_full_file_proposal(
      json.dumps(
        {
          "seed": "llvm/test/CodeGen/DLC/example.ll",
          "profile": "example",
          "candidates": [
            {
              "filename": "example_agent_0.ll",
              "text": "define void @example0() { ret void }\n",
              "rationale": "first candidate",
              "intended_stress": "first candidate",
            },
            {
              "filename": "../bad.ll",
              "text": "define void @example1() { ret void }\n",
              "rationale": "invalid candidate after cap",
              "intended_stress": "invalid candidate after cap",
            },
          ],
        }
      ),
      max_candidates=1,
    )
  except ValueError as exc:
    assert "filename" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_parse_agent_proposal_accepts_json_fence():
  proposal = parse_agent_proposal(
    """```json
{
  "seed": "llvm/test/CodeGen/DLC/example.ll",
  "profile": "example",
  "proposed_mutations": [
    {
      "axis": "shift_amount_boundary",
      "location_hint": "example",
      "old_value": 7,
      "new_value": 6,
      "rationale": "exercise the lower adjacent shift boundary"
    }
  ]
}
```"""
  )

  assert proposal.seed == "llvm/test/CodeGen/DLC/example.ll"
  assert proposal.profile == "example"
  assert proposal.proposed_mutations[0].new_value == 6


def test_parse_agent_proposal_accepts_grouped_edits():
  proposal = parse_agent_proposal(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "proposed_mutations": [
          {
            "axis": "shift_amount_boundary",
            "location_hint": "@example",
            "edits": [
              {"old_value": 7, "new_value": 6, "occurrence": 1},
              {"old_value": 7, "new_value": 6, "occurrence": 2},
            ],
            "rationale": "keep paired shifts aligned",
          }
        ],
      }
    )
  )

  mutation = proposal.proposed_mutations[0]
  assert mutation.old_value == 7
  assert mutation.new_value == 6
  assert [edit.occurrence for edit in mutation.edits] == [1, 2]
  assert mutation.to_dict()["edits"] == [
    {"old_value": 7, "new_value": 6, "occurrence": 1},
    {"old_value": 7, "new_value": 6, "occurrence": 2},
  ]


def test_parse_agent_proposal_preserves_optional_evidence_metadata():
  proposal = parse_agent_proposal(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/example.ll",
        "profile": "example",
        "proposed_mutations": [
          {
            "axis": "shift_amount_boundary",
            "location_hint": "@example",
            "old_value": 7,
            "new_value": 6,
            "rationale": "exercise a kernel-informed boundary",
            "evidence_tags": ["addr_exp_boundary", "dma_length_boundary"],
            "source_evidence": (
              "selected kernel evidence has address and DMA boundary hints"
            ),
          }
        ],
      }
    )
  )

  mutation = proposal.proposed_mutations[0]
  assert mutation.evidence_tags == ["addr_exp_boundary", "dma_length_boundary"]
  assert (
    mutation.source_evidence
    == "selected kernel evidence has address and DMA boundary hints"
  )
  assert mutation.to_dict()["evidence_tags"] == [
    "addr_exp_boundary",
    "dma_length_boundary",
  ]
  assert (
    mutation.to_dict()["source_evidence"]
    == "selected kernel evidence has address and DMA boundary hints"
  )


def test_parse_agent_proposal_rejects_malformed_evidence_metadata():
  base_mutation = {
    "axis": "shift_amount_boundary",
    "location_hint": "@example",
    "old_value": 7,
    "new_value": 6,
    "rationale": "exercise a kernel-informed boundary",
  }
  invalid_metadata = [
    {"evidence_tags": "addr_exp_boundary"},
    {"evidence_tags": ["addr_exp_boundary", 7]},
    {"evidence_tags": {"kind": "addr_exp_boundary"}},
    {"source_evidence": ["not", "a", "string"]},
  ]

  for metadata in invalid_metadata:
    mutation = {**base_mutation, **metadata}
    try:
      parse_agent_proposal(
        json.dumps(
          {
            "seed": "llvm/test/CodeGen/DLC/example.ll",
            "profile": "example",
            "proposed_mutations": [mutation],
          }
        )
      )
    except ValueError as exc:
      assert "evidence" in str(exc)
    else:
      raise AssertionError("expected ValueError")


def test_filter_agent_mutations_rejects_unsupported_axis_and_value(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  profile = get_profile("example", profiles_dir)
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
            "rationale": "valid immediate mutation",
          },
          {
            "axis": "address_shapes",
            "location_hint": "example",
            "old_value": 7,
            "new_value": 6,
            "rationale": "unsupported in phase 11",
            "evidence_tags": ["addr_exp_boundary"],
            "source_evidence": "metadata must not make this axis supported",
          },
          {
            "axis": "shift_amount_boundary",
            "location_hint": "example",
            "old_value": 7,
            "new_value": 99,
            "rationale": "not in profile",
          },
        ],
      }
    ),
    encoding="utf-8",
  )
  proposal = parse_agent_proposal(proposal_path.read_text(encoding="utf-8"))

  accepted, rejected = filter_agent_mutations(
    proposal,
    profile,
    "llvm/test/CodeGen/DLC/example.ll",
    max_candidates=5,
  )

  assert [mutation.new_value for mutation in accepted] == [6]
  assert [entry.reason for entry in rejected] == [
    "unsupported mutation axis for current agent generator: address_shapes",
    "new_value 99 is not allowed by profile immediates.values",
  ]


def test_filter_agent_mutations_rejects_seed_mismatch(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  profile = get_profile("example", profiles_dir)
  proposal = parse_agent_proposal(
    json.dumps(
      {
        "seed": "llvm/test/CodeGen/DLC/other.ll",
        "profile": "example",
        "proposed_mutations": [],
      }
    )
  )

  try:
    filter_agent_mutations(
      proposal,
      profile,
      "llvm/test/CodeGen/DLC/example.ll",
      max_candidates=5,
    )
  except ValueError as exc:
    assert "does not match requested seed" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_request_agent_proposal_accepts_llvm_harness_env(monkeypatch):
  captured = {}

  class FakeResponse:
    def __enter__(self):
      return self

    def __exit__(self, *_):
      return None

    def read(self):
      return json.dumps(
        {
          "choices": [
            {
              "message": {
                "content": json.dumps(
                  {
                    "seed": "llvm/test/CodeGen/DLC/example.ll",
                    "profile": "example",
                    "proposed_mutations": [],
                  }
                )
              }
            }
          ]
        }
      ).encode("utf-8")

  def fake_urlopen(request, timeout):
    captured["url"] = request.full_url
    captured["headers"] = dict(request.header_items())
    captured["payload"] = json.loads(request.data.decode("utf-8"))
    captured["timeout"] = timeout
    return FakeResponse()

  monkeypatch.delenv("DLC_TESTFORGE_LM_API_KEY", raising=False)
  monkeypatch.delenv("DLC_TESTFORGE_LM_API_ENDPOINT", raising=False)
  monkeypatch.delenv("DLC_TESTFORGE_LM_MODEL", raising=False)
  monkeypatch.setenv("LLVM_HARNESS_LM_API_KEY", "sk-test")
  monkeypatch.setenv("LLVM_HARNESS_LM_API_ENDPOINT", "https://example.test/v1")
  monkeypatch.setenv("LLVM_HARNESS_LM_MODEL", "test-model")
  monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

  proposal = request_agent_proposal({"task": "unit-test"})

  assert proposal.seed == "llvm/test/CodeGen/DLC/example.ll"
  assert captured["url"] == "https://example.test/v1/chat/completions"
  assert captured["payload"]["model"] == "test-model"
  assert captured["timeout"] == 120
  assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_request_agent_proposal_accepts_codex_auth_and_config(tmp_path, monkeypatch):
  captured = {}
  auth_path = tmp_path / "auth.json"
  config_path = tmp_path / "config.toml"
  auth_path.write_text('{"OPENAI_API_KEY": "sk-codex-test"}\n', encoding="utf-8")
  config_path.write_text(
    """
model_provider = "openrouter"
model = "gpt-5.5"

[model_providers.openrouter]
base_url = "https://openrouter.example.test/v1"

[notice.model_migrations]
"gpt-5.5" = "openai/gpt-5.5"
""",
    encoding="utf-8",
  )

  class FakeResponse:
    def __enter__(self):
      return self

    def __exit__(self, *_):
      return None

    def read(self):
      return json.dumps(
        {
          "choices": [
            {
              "message": {
                "content": json.dumps(
                  {
                    "seed": "llvm/test/CodeGen/DLC/example.ll",
                    "profile": "example",
                    "proposed_mutations": [],
                  }
                )
              }
            }
          ]
        }
      ).encode("utf-8")

  def fake_urlopen(request, timeout):
    captured["url"] = request.full_url
    captured["headers"] = dict(request.header_items())
    captured["payload"] = json.loads(request.data.decode("utf-8"))
    return FakeResponse()

  for name in [
    "DLC_TESTFORGE_LM_API_KEY",
    "DLC_TESTFORGE_LM_API_ENDPOINT",
    "DLC_TESTFORGE_LM_MODEL",
    "LLVM_HARNESS_LM_API_KEY",
    "LLVM_HARNESS_LM_API_ENDPOINT",
    "LLVM_HARNESS_LM_MODEL",
  ]:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setattr("dlc_testforge.agent.CODEX_AUTH_PATH", auth_path)
  monkeypatch.setattr("dlc_testforge.agent.CODEX_CONFIG_PATH", config_path)
  monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

  request_agent_proposal({"task": "unit-test"})

  assert captured["url"] == "https://openrouter.example.test/v1/chat/completions"
  assert captured["payload"]["model"] == "openai/gpt-5.5"
  assert captured["headers"]["Authorization"] == "Bearer sk-codex-test"


def test_select_kernel_evidence_prefers_machine_addropt_address_records():
  kernel_index = {
    "summary": {"kernel_count": 3},
    "global_edge_hints": [],
    "kernels": [
      _kernel_record(
        "vector_only",
        source="dlc_kernels/vector.c",
        edge_kinds=["vector_lane_boundary"],
        vector_types=["float8_128"],
      ),
      _kernel_record(
        "address_rich",
        source="dlc_kernels/address.c",
        edge_kinds=["addr_exp_boundary", "dma_length_boundary", "stride_boundary"],
        relation_kinds=["address_exponent", "memory_space_pair"],
        dma=True,
        constants=[7, 128],
        memory_spaces=["HBM", "VMEM"],
      ),
    ],
  }

  selected = select_kernel_evidence(
    kernel_index,
    "machine_addropt",
    "llvm/test/CodeGen/DLC/machine-addropt-prera.ll",
    max_records=1,
  )

  assert selected["summary"] == {"kernel_count": 3}
  assert selected["selection"]["selected_count"] == 1
  assert selected["kernels"][0]["name"] == "address_rich"
  assert selected["kernels"][0]["selection_score"] > 0


def test_select_kernel_evidence_prefers_globalisel_vector_records():
  kernel_index = {
    "kernels": [
      _kernel_record(
        "dma_only",
        source="dlc_kernels/dma.c",
        edge_kinds=["addr_exp_boundary"],
        relation_kinds=["address_exponent"],
        dma=True,
      ),
      _kernel_record(
        "vector_mask",
        source="dlc_kernels/vector.c",
        edge_kinds=["vector_lane_boundary", "mask_boundary"],
        relation_kinds=["mask_boundary"],
        vector_types=["float8_128"],
        intrinsics=["v_f32_ld_tnsr_st_msk"],
        dtype_hints=["f32"],
        constants=[8, 128],
      ),
    ],
  }

  selected = select_kernel_evidence(
    kernel_index,
    "globalisel",
    "llvm/test/CodeGen/DLC/GIISel/basic.ll",
    max_records=1,
  )

  assert selected["kernels"][0]["name"] == "vector_mask"


def test_select_kernel_evidence_is_deterministic_and_respects_max_records():
  kernel_index = {
    "kernels": [
      _kernel_record("b", source="dlc_kernels/b.c", dma=True),
      _kernel_record("a", source="dlc_kernels/a.c", dma=True),
      _kernel_record("c", source="dlc_kernels/c.c", dma=True),
    ],
  }

  selected = select_kernel_evidence(
    kernel_index,
    "unknown_profile",
    "seed.ll",
    max_records=2,
  )

  assert [record["name"] for record in selected["kernels"]] == ["a", "b"]
  assert selected["selection"]["selected_count"] == 2


def test_select_kernel_evidence_handles_malformed_input():
  selected = select_kernel_evidence(None, "machine_addropt", "seed.ll")

  assert selected == {
    "summary": {},
    "selection": {
      "profile": "machine_addropt",
      "seed": "seed.ll",
      "max_records": 12,
      "selected_count": 0,
    },
    "kernels": [],
    "global_edge_hints": [],
  }
  malformed = select_kernel_evidence({"kernels": "bad"}, "globalisel", "seed.ll")
  assert malformed["kernels"] == []


def test_select_kernel_evidence_prunes_records_and_caps_global_edges():
  kernel_index = {
    "summary": {"kernel_count": 1},
    "global_edge_hints": [
      {"kind": "tile_boundary", "base": index}
      for index in range(25)
    ],
    "kernels": [
      _kernel_record(
        "wide",
        source="dlc_kernels/wide.c",
        edge_kinds=["mask_boundary", "vector_lane_boundary"],
        relation_kinds=["mask_boundary"] * 10,
        vector_types=[f"type{index}_128" for index in range(10)],
        intrinsics=[f"v_op_{index}" for index in range(14)],
        constants=list(range(20)),
        extra={"raw_source": "do not include"},
      )
    ],
  }

  selected = select_kernel_evidence(
    kernel_index,
    "globalisel",
    "seed.ll",
    max_records=1,
  )
  record = selected["kernels"][0]

  assert "raw_source" not in record
  assert len(record["usage"]["vector_types"]) == 8
  assert len(record["usage"]["intrinsics"]) == 12
  assert len(record["usage"]["constants"]) == 16
  assert len(record["usage"]["relations"]) == 8
  assert len(record["edge_hints"]) == 2
  assert len(selected["global_edge_hints"]) == 20


def test_build_agent_context_optionally_includes_kernel_usage_evidence(tmp_path):
  profiles_dir = _make_profiles_dir(tmp_path)
  profile = get_profile("example", profiles_dir)
  evidence = {
    "selection": {"selected_count": 1},
    "kernels": [{"name": "custom_kernel"}],
  }

  base_context = build_agent_context(
    profile,
    "llvm/test/CodeGen/DLC/example.ll",
    "define void @example() { ret void }\n",
    3,
  )
  context = build_agent_context(
    profile,
    "llvm/test/CodeGen/DLC/example.ll",
    "define void @example() { ret void }\n",
    3,
    kernel_usage_evidence=evidence,
  )

  assert "kernel_usage_evidence" not in base_context
  assert context["kernel_usage_evidence"] == evidence
  assert context["contract"]["optional_mutation_fields"] == [
    "evidence_tags",
    "source_evidence",
  ]
  assert any("kernel usage evidence" in guardrail for guardrail in context["guardrails"])
