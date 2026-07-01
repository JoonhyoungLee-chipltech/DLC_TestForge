from __future__ import annotations

import json

from dlc_testforge.agent import (
  filter_agent_mutations,
  parse_agent_proposal,
  request_agent_proposal,
)
from dlc_testforge.profiles import get_profile
from tests.test_generate import _make_profiles_dir


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
