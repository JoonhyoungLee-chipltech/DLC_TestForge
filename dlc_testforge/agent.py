from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlc_testforge.profiles import MutationProfile


DEFAULT_ENDPOINT = "https://api.openai.com/v1"
PLACEHOLDER_VALUES = {"", "sk-", "Please fill out"}
CODEX_AUTH_PATH = Path("/root/.codex/auth.json")
CODEX_CONFIG_PATH = Path("/root/.codex/config.toml")
KERNEL_INFORMED_AGENT_AXES = {
  "addr_exp_boundary",
  "dma_length_boundary",
  "stride_boundary",
  "tile_boundary",
  "mask_boundary",
  "vector_lane_boundary",
}


@dataclass(frozen=True)
class AgentMutationEdit:
  old_value: int
  new_value: int
  occurrence: int | None = None

  def to_dict(self) -> dict[str, Any]:
    data = {
      "old_value": self.old_value,
      "new_value": self.new_value,
    }
    if self.occurrence is not None:
      data["occurrence"] = self.occurrence
    return data


@dataclass(frozen=True)
class AgentMutationProposal:
  axis: str
  location_hint: str
  edits: list[AgentMutationEdit]
  rationale: str
  evidence_tags: list[str] | None = None
  source_evidence: str | None = None

  @property
  def old_value(self) -> int:
    return self.edits[0].old_value

  @property
  def new_value(self) -> int:
    return self.edits[0].new_value

  def to_dict(self) -> dict[str, Any]:
    data = {
      "axis": self.axis,
      "location_hint": self.location_hint,
      "old_value": self.old_value,
      "new_value": self.new_value,
      "edits": [edit.to_dict() for edit in self.edits],
      "rationale": self.rationale,
    }
    if self.evidence_tags:
      data["evidence_tags"] = self.evidence_tags
    if self.source_evidence:
      data["source_evidence"] = self.source_evidence
    return data


@dataclass(frozen=True)
class AgentProposal:
  seed: str
  profile: str
  proposed_mutations: list[AgentMutationProposal]

  def to_dict(self) -> dict[str, Any]:
    return {
      "seed": self.seed,
      "profile": self.profile,
      "proposed_mutations": [
        mutation.to_dict() for mutation in self.proposed_mutations
      ],
    }


@dataclass(frozen=True)
class AgentFullFileCandidate:
  filename: str
  text: str
  rationale: str
  intended_stress: str
  evidence_tags: list[str] | None = None
  source_evidence: str | None = None

  def to_dict(self) -> dict[str, Any]:
    data = {
      "filename": self.filename,
      "text": self.text,
      "rationale": self.rationale,
      "intended_stress": self.intended_stress,
    }
    if self.evidence_tags:
      data["evidence_tags"] = self.evidence_tags
    if self.source_evidence:
      data["source_evidence"] = self.source_evidence
    return data


@dataclass(frozen=True)
class AgentFullFileProposal:
  seed: str
  profile: str
  candidates: list[AgentFullFileCandidate]

  def to_dict(self) -> dict[str, Any]:
    return {
      "seed": self.seed,
      "profile": self.profile,
      "candidates": [candidate.to_dict() for candidate in self.candidates],
    }


@dataclass(frozen=True)
class RejectedAgentMutation:
  index: int
  mutation: dict[str, Any]
  reason: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "index": self.index,
      "mutation": self.mutation,
      "reason": self.reason,
    }


def build_agent_context(
  profile: MutationProfile,
  seed_relative: str,
  seed_text: str,
  max_candidates: int,
  *,
  test_index: dict[str, Any] | None = None,
  spec_index: dict[str, Any] | None = None,
  td_index: dict[str, Any] | None = None,
  kernel_usage_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
  seed_record = _find_seed_record(test_index, seed_relative)
  guardrails = [
    "Use the exact seed and profile from this context.",
    "Only propose immediate-value mutations supported by profile.mutation_axes.immediates.values.",
    "Use edits when one candidate must change multiple related immediate occurrences together.",
    "Do not change RUN lines.",
    "Do not invent DLC instructions or intrinsics.",
    "Each edit must change one old_value to one new_value.",
  ]
  if kernel_usage_evidence is not None:
    guardrails.append(
      "Treat kernel usage evidence as workload evidence, not permission to invent unsupported IR mutations."
    )
  context = {
    "task": "propose_dlc_test_mutations",
    "contract": {
      "role": "propose mutation ideas only",
      "output_format": "json",
      "required_top_level_fields": [
        "seed",
        "profile",
        "proposed_mutations",
      ],
      "required_mutation_fields": [
        "axis",
        "location_hint",
        "edits",
        "rationale",
      ],
      "optional_mutation_fields": [
        "evidence_tags",
        "source_evidence",
      ],
    },
    "seed": {
      "path": seed_relative,
      "text": seed_text,
      "index_record": seed_record,
    },
    "profile": profile.to_dict(),
    "spec_records": _select_spec_records(spec_index, profile.spec_sources),
    "source_records": _select_source_records(td_index, profile.source_files),
    "td_summary": (td_index or {}).get("summary", {}),
    "mutation_budget": max_candidates,
    "guardrails": guardrails,
  }
  if kernel_usage_evidence is not None:
    context["kernel_usage_evidence"] = kernel_usage_evidence
  return context


def build_full_file_agent_context(
  profile: MutationProfile,
  seed_relative: str,
  seed_text: str,
  max_candidates: int,
  *,
  llvm_root: Path,
  test_index: dict[str, Any] | None = None,
  spec_index: dict[str, Any] | None = None,
  td_index: dict[str, Any] | None = None,
  kernel_usage_evidence: dict[str, Any] | None = None,
  nearby_test_limit: int = 4,
) -> dict[str, Any]:
  seed_path = Path(seed_relative)
  seed_directory = seed_path.parent.as_posix()
  if seed_directory == ".":
    seed_directory = ""
  guardrails = [
    "Preserve the seed test's pass workflow unless the rationale explains a narrow variation.",
    "Keep RUN lines valid for the active profile.",
    "Write complete .ll files, not patches.",
    "Do not invent DLC intrinsics, instructions, address spaces, or datalayouts.",
    "Prefer patterns supported by the seed, specs, backend source, or kernel evidence.",
    "Each candidate must target one stress point.",
    "Do not output Markdown fences.",
    "Validation and classification are authoritative.",
  ]
  context = {
    "task": "write_dlc_full_file_test_mutations",
    "contract": {
      "role": "write complete candidate .ll files only",
      "output_format": "json",
      "required_top_level_fields": [
        "seed",
        "profile",
        "candidates",
      ],
      "required_candidate_fields": [
        "filename",
        "text",
        "rationale",
        "intended_stress",
      ],
      "optional_candidate_fields": [
        "evidence_tags",
        "source_evidence",
      ],
    },
    "seed": {
      "path": seed_relative,
      "directory": seed_directory,
      "text": seed_text,
      "run_lines": _extract_run_lines(seed_text),
      "check_prefixes": _extract_check_prefixes(seed_text),
      "index_record": _find_seed_record(test_index, seed_relative),
    },
    "nearby_tests": _select_nearby_tests(
      llvm_root,
      seed_relative,
      nearby_test_limit,
    ),
    "profile": profile.to_dict(),
    "spec_records": _select_spec_records(spec_index, profile.spec_sources),
    "source_records": _select_source_records(td_index, profile.source_files),
    "td_summary": (td_index or {}).get("summary", {}),
    "mutation_budget": max_candidates,
    "guardrails": guardrails,
  }
  if kernel_usage_evidence is not None:
    context["kernel_usage_evidence"] = kernel_usage_evidence
  return context


def select_kernel_evidence(
  kernel_index: dict[str, Any] | None,
  profile_name: str,
  seed_relative: str,
  max_records: int = 12,
) -> dict[str, Any]:
  empty = {
    "summary": {},
    "selection": {
      "profile": profile_name,
      "seed": seed_relative,
      "max_records": max_records,
      "selected_count": 0,
    },
    "kernels": [],
    "global_edge_hints": [],
  }
  if not isinstance(kernel_index, dict):
    return empty
  kernels = kernel_index.get("kernels")
  if not isinstance(kernels, list):
    return empty

  scored = []
  for record in kernels:
    if not isinstance(record, dict):
      continue
    score = _kernel_evidence_score(record, profile_name)
    if score <= 0:
      continue
    scored.append((score, record))

  scored.sort(
    key=lambda item: (
      -item[0],
      str(item[1].get("source") or ""),
      str(item[1].get("name") or ""),
    )
  )
  limit = max(max_records, 0)
  selected = [
    _prune_kernel_evidence(record, score)
    for score, record in scored[:limit]
  ]
  return {
    "summary": kernel_index.get("summary")
    if isinstance(kernel_index.get("summary"), dict)
    else {},
    "selection": {
      "profile": profile_name,
      "seed": seed_relative,
      "max_records": max_records,
      "selected_count": len(selected),
    },
    "kernels": selected,
    "global_edge_hints": _limit_dict_list(kernel_index.get("global_edge_hints"), 20),
  }


def load_agent_proposal(path: Path) -> AgentProposal:
  text = path.expanduser().read_text(encoding="utf-8")
  return parse_agent_proposal(text)


def load_agent_full_file_proposal(
  path: Path,
  *,
  seed_relative: str | None = None,
  profile_name: str | None = None,
  max_candidates: int | None = None,
) -> AgentFullFileProposal:
  text = path.expanduser().read_text(encoding="utf-8")
  return parse_agent_full_file_proposal(
    text,
    seed_relative=seed_relative,
    profile_name=profile_name,
    max_candidates=max_candidates,
  )


def request_agent_proposal(
  context: dict[str, Any],
  *,
  model: str | None = None,
  endpoint: str | None = None,
  api_key: str | None = None,
) -> AgentProposal:
  resolved_model = _first_config_value(
    model,
    os.environ.get("DLC_TESTFORGE_LM_MODEL"),
    os.environ.get("LLVM_HARNESS_LM_MODEL"),
    _load_codex_config().get("model"),
  )
  if not resolved_model:
    raise ValueError(
      "agent mode requires --agent-model, DLC_TESTFORGE_LM_MODEL, or "
      "LLVM_HARNESS_LM_MODEL when no --agent-proposal is provided"
    )

  resolved_api_key = _first_config_value(
    api_key,
    os.environ.get("DLC_TESTFORGE_LM_API_KEY"),
    os.environ.get("LLVM_HARNESS_LM_API_KEY"),
    _load_codex_auth().get("OPENAI_API_KEY"),
  )
  if not resolved_api_key:
    raise ValueError(
      "agent mode requires DLC_TESTFORGE_LM_API_KEY or LLVM_HARNESS_LM_API_KEY "
      "when no --agent-proposal is provided"
    )

  resolved_endpoint = _first_config_value(
    endpoint,
    os.environ.get("DLC_TESTFORGE_LM_API_ENDPOINT"),
    os.environ.get("LLVM_HARNESS_LM_API_ENDPOINT"),
    _load_codex_config().get("provider.base_url"),
    DEFAULT_ENDPOINT,
  ).rstrip("/")
  resolved_model = _migrate_codex_model(resolved_model)
  payload = {
    "model": resolved_model,
    "temperature": 0,
    "messages": [
      {
        "role": "system",
        "content": (
          "You propose DLC LLVM test mutations. Return only one JSON object. "
          "Do not include Markdown fences, prose, comments, or extra top-level keys. "
          "The object must have exactly these top-level fields: seed, profile, "
          "proposed_mutations. The seed and profile fields must be strings. "
          "Mutation evidence_tags and source_evidence are optional metadata only; "
          "they do not permit unsupported mutation axes, values, IR, or intrinsics."
        ),
      },
      {
        "role": "user",
        "content": "\n".join(
          [
            "Use this context to propose mutations:",
            json.dumps(context, indent=2, sort_keys=True),
            "",
            "Return this exact JSON shape:",
            json.dumps(
              {
                "seed": context.get("seed", {}).get("path", ""),
                "profile": context.get("profile", {}).get("name", ""),
                "proposed_mutations": [
                  {
                    "axis": "shift_amount_boundary",
                    "location_hint": "function or line hint",
                    "edits": [
                      {
                        "old_value": 7,
                        "new_value": 6,
                        "occurrence": 1,
                      }
                    ],
                    "rationale": "why this edge case is worth trying",
                    "evidence_tags": ["addr_exp_boundary"],
                    "source_evidence": (
                      "kernel usage evidence includes address exponent boundary hints"
                    ),
                  }
                ],
              },
              indent=2,
              sort_keys=True,
            ),
            "Use an empty proposed_mutations list if no valid mutation is supported.",
          ]
        ),
      },
    ],
  }
  request = urllib.request.Request(
    f"{resolved_endpoint}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
      "Authorization": f"Bearer {resolved_api_key}",
      "Content-Type": "application/json",
    },
    method="POST",
  )
  try:
    with urllib.request.urlopen(request, timeout=120) as response:
      response_data = json.loads(response.read().decode("utf-8"))
  except urllib.error.URLError as exc:
    raise ValueError(f"agent request failed: {exc}") from exc

  try:
    content = response_data["choices"][0]["message"]["content"]
  except (KeyError, IndexError, TypeError) as exc:
    raise ValueError("agent response did not contain choices[0].message.content") from exc
  return parse_agent_proposal(content)


def request_agent_full_file_proposal(
  context: dict[str, Any],
  *,
  model: str | None = None,
  endpoint: str | None = None,
  api_key: str | None = None,
  max_candidates: int | None = None,
) -> AgentFullFileProposal:
  resolved_model = _first_config_value(
    model,
    os.environ.get("DLC_TESTFORGE_LM_MODEL"),
    os.environ.get("LLVM_HARNESS_LM_MODEL"),
    _load_codex_config().get("model"),
  )
  if not resolved_model:
    raise ValueError(
      "agent mode requires --agent-model, DLC_TESTFORGE_LM_MODEL, or "
      "LLVM_HARNESS_LM_MODEL when no --agent-proposal is provided"
    )

  resolved_api_key = _first_config_value(
    api_key,
    os.environ.get("DLC_TESTFORGE_LM_API_KEY"),
    os.environ.get("LLVM_HARNESS_LM_API_KEY"),
    _load_codex_auth().get("OPENAI_API_KEY"),
  )
  if not resolved_api_key:
    raise ValueError(
      "agent mode requires DLC_TESTFORGE_LM_API_KEY or LLVM_HARNESS_LM_API_KEY "
      "when no --agent-proposal is provided"
    )

  resolved_endpoint = _first_config_value(
    endpoint,
    os.environ.get("DLC_TESTFORGE_LM_API_ENDPOINT"),
    os.environ.get("LLVM_HARNESS_LM_API_ENDPOINT"),
    _load_codex_config().get("provider.base_url"),
    DEFAULT_ENDPOINT,
  ).rstrip("/")
  resolved_model = _migrate_codex_model(resolved_model)
  seed_relative = context.get("seed", {}).get("path")
  profile_name = context.get("profile", {}).get("name")
  payload = {
    "model": resolved_model,
    "temperature": 0,
    "messages": [
      {
        "role": "system",
        "content": (
          "You write complete DLC LLVM .ll test mutation candidates. "
          "Return only one JSON object. Do not include Markdown fences, prose, "
          "comments outside JSON, or extra top-level fields. Each candidate text "
          "must be a complete .ll file. Do not invent DLC intrinsics, "
          "instructions, address spaces, or datalayouts. Validation and "
          "classification will decide whether candidates are useful."
        ),
      },
      {
        "role": "user",
        "content": "\n".join(
          [
            "Use this context to write full-file candidate tests:",
            json.dumps(context, indent=2, sort_keys=True),
            "",
            "Return this exact JSON shape:",
            json.dumps(
              {
                "seed": seed_relative or "",
                "profile": profile_name or "",
                "candidates": [
                  {
                    "filename": "seed-mutated-stress.ll",
                    "text": "; RUN: ...\n\ndefine void @candidate() {\n  ret void\n}\n",
                    "rationale": "why this complete candidate is worth trying",
                    "intended_stress": "one clear stress point",
                    "evidence_tags": ["addr_exp_boundary"],
                    "source_evidence": (
                      "kernel usage evidence includes address exponent boundary hints"
                    ),
                  }
                ],
              },
              indent=2,
              sort_keys=True,
            ),
            "Use an empty candidates list only if no valid candidate can be written.",
          ]
        ),
      },
    ],
  }
  request = urllib.request.Request(
    f"{resolved_endpoint}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
      "Authorization": f"Bearer {resolved_api_key}",
      "Content-Type": "application/json",
    },
    method="POST",
  )
  try:
    with urllib.request.urlopen(request, timeout=120) as response:
      response_data = json.loads(response.read().decode("utf-8"))
  except urllib.error.URLError as exc:
    raise ValueError(f"agent request failed: {exc}") from exc

  try:
    content = response_data["choices"][0]["message"]["content"]
  except (KeyError, IndexError, TypeError) as exc:
    raise ValueError(
      "agent full-file response did not contain choices[0].message.content"
    ) from exc
  return parse_agent_full_file_proposal(
    content,
    seed_relative=seed_relative,
    profile_name=profile_name,
    max_candidates=max_candidates,
  )


def parse_agent_proposal(text: str) -> AgentProposal:
  data = _loads_json_object(text)
  if not isinstance(data, dict):
    raise ValueError("agent proposal must be a JSON object")

  seed = data.get("seed")
  profile = data.get("profile")
  proposed = data.get("proposed_mutations")
  if not isinstance(seed, str) or not seed:
    raise ValueError("agent proposal field seed must be a non-empty string")
  if not isinstance(profile, str) or not profile:
    raise ValueError("agent proposal field profile must be a non-empty string")
  if not isinstance(proposed, list):
    raise ValueError("agent proposal field proposed_mutations must be a list")

  mutations = []
  for index, item in enumerate(proposed):
    mutations.append(_parse_mutation(index, item))
  return AgentProposal(seed=seed, profile=profile, proposed_mutations=mutations)


def parse_agent_full_file_proposal(
  text: str,
  *,
  seed_relative: str | None = None,
  profile_name: str | None = None,
  max_candidates: int | None = None,
) -> AgentFullFileProposal:
  data = _loads_json_object(text)
  if not isinstance(data, dict):
    raise ValueError("agent full-file proposal must be a JSON object")

  seed = data.get("seed")
  profile = data.get("profile")
  candidates_data = data.get("candidates")
  if not isinstance(seed, str) or not seed.strip():
    raise ValueError(
      "agent full-file proposal field seed must be a non-empty string"
    )
  if not isinstance(profile, str) or not profile.strip():
    raise ValueError(
      "agent full-file proposal field profile must be a non-empty string"
    )
  if seed_relative is not None and seed != seed_relative:
    raise ValueError(
      f"agent full-file proposal seed {seed} does not match requested seed {seed_relative}"
    )
  if profile_name is not None and profile != profile_name:
    raise ValueError(
      f"agent full-file proposal profile {profile} does not match requested profile {profile_name}"
    )
  if not isinstance(candidates_data, list):
    raise ValueError("agent full-file proposal field candidates must be a list")
  if max_candidates is not None and (
    not isinstance(max_candidates, int)
    or isinstance(max_candidates, bool)
    or max_candidates < 0
  ):
    raise ValueError("max_candidates must be a non-negative integer")

  candidates = []
  seen_texts = set()
  for index, item in enumerate(candidates_data):
    candidate = _parse_full_file_candidate(index, item)
    if candidate.text in seen_texts:
      raise ValueError(f"candidates[{index}].text duplicates an earlier candidate")
    seen_texts.add(candidate.text)
    candidates.append(candidate)

  if max_candidates is not None:
    candidates = candidates[:max_candidates]
  return AgentFullFileProposal(seed=seed, profile=profile, candidates=candidates)


def filter_agent_mutations(
  proposal: AgentProposal,
  profile: MutationProfile,
  seed_relative: str,
  max_candidates: int,
) -> tuple[list[AgentMutationProposal], list[RejectedAgentMutation]]:
  if proposal.seed != seed_relative:
    raise ValueError(
      f"agent proposal seed {proposal.seed} does not match requested seed {seed_relative}"
    )
  if proposal.profile != profile.name:
    raise ValueError(
      f"agent proposal profile {proposal.profile} does not match requested profile {profile.name}"
    )

  accepted: list[AgentMutationProposal] = []
  rejected: list[RejectedAgentMutation] = []
  allowed_values = _allowed_immediate_values(profile)
  if not allowed_values:
    for index, mutation in enumerate(proposal.proposed_mutations):
      rejected.append(
        RejectedAgentMutation(
          index=index,
          mutation=mutation.to_dict(),
          reason="profile does not enable integer immediate mutation values",
        )
      )
    return accepted, rejected

  for index, mutation in enumerate(proposal.proposed_mutations):
    reason = _rejection_reason(mutation, allowed_values)
    if reason is not None:
      rejected.append(
        RejectedAgentMutation(
          index=index,
          mutation=mutation.to_dict(),
          reason=reason,
        )
      )
      continue
    if len(accepted) >= max_candidates:
      rejected.append(
        RejectedAgentMutation(
          index=index,
          mutation=mutation.to_dict(),
          reason="mutation budget exhausted",
        )
      )
      continue
    accepted.append(mutation)
  return accepted, rejected


def write_agent_json(path: Path, data: Any) -> None:
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_config_value(*values: str | None) -> str | None:
  for value in values:
    if value is None:
      continue
    stripped = value.strip()
    if stripped in PLACEHOLDER_VALUES:
      continue
    return stripped
  return None


def _load_codex_auth() -> dict[str, str]:
  if not CODEX_AUTH_PATH.is_file():
    return {}
  try:
    data = json.loads(CODEX_AUTH_PATH.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return {}
  return data if isinstance(data, dict) else {}


def _load_codex_config() -> dict[str, str]:
  if not CODEX_CONFIG_PATH.is_file():
    return {}
  try:
    text = CODEX_CONFIG_PATH.read_text(encoding="utf-8")
  except OSError:
    return {}

  root: dict[str, str] = {}
  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
      continue
    key, value = _parse_toml_assignment(line)
    if key:
      root[key] = value

  provider_name = root.get("model_provider", "")
  provider_cfg = _parse_toml_table(text, f"model_providers.{provider_name}")
  migrations = _parse_toml_table(text, "notice.model_migrations")

  merged = dict(root)
  for key, value in provider_cfg.items():
    merged[f"provider.{key}"] = value
  for key, value in migrations.items():
    merged[f"migration.{key}"] = value
  return merged


def _parse_toml_table(text: str, table_name: str) -> dict[str, str]:
  current = None
  data: dict[str, str] = {}
  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("[") and line.endswith("]"):
      current = line[1:-1].strip()
      continue
    if current != table_name or "=" not in line:
      continue
    key, value = _parse_toml_assignment(line)
    if key:
      data[key] = value
  return data


def _parse_toml_assignment(line: str) -> tuple[str, str]:
  key, value = line.split("=", 1)
  key = key.strip()
  value = value.strip()
  if key.startswith('"') and key.endswith('"'):
    key = key[1:-1]
  if value.startswith('"') and value.endswith('"'):
    value = value[1:-1]
  return key, value


def _migrate_codex_model(model: str) -> str:
  config = _load_codex_config()
  migrated = config.get(f"migration.{model}")
  if migrated:
    return migrated
  if config.get("model_provider") == "openrouter" and "/" not in model:
    if model.startswith("gpt-"):
      return f"openai/{model}"
  return model


def _loads_json_object(text: str) -> Any:
  stripped = text.strip()
  if stripped.startswith("```"):
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
      stripped = "\n".join(lines[1:-1]).strip()
      if stripped.startswith("json\n"):
        stripped = stripped[len("json\n") :].strip()
  try:
    return json.loads(stripped)
  except json.JSONDecodeError:
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
      raise
    return json.loads(stripped[start : end + 1])


def _parse_mutation(index: int, item: Any) -> AgentMutationProposal:
  if not isinstance(item, dict):
    raise ValueError(f"proposed_mutations[{index}] must be a JSON object")
  axis = item.get("axis")
  location_hint = item.get("location_hint")
  rationale = item.get("rationale")
  if not isinstance(axis, str) or not axis:
    raise ValueError(f"proposed_mutations[{index}].axis must be a non-empty string")
  if not isinstance(location_hint, str):
    raise ValueError(f"proposed_mutations[{index}].location_hint must be a string")
  if not isinstance(rationale, str) or not rationale:
    raise ValueError(f"proposed_mutations[{index}].rationale must be a non-empty string")
  edits = _parse_mutation_edits(index, item)
  evidence_tags = _parse_evidence_tags(index, item.get("evidence_tags"))
  source_evidence = _parse_source_evidence(index, item.get("source_evidence"))
  return AgentMutationProposal(
    axis=axis,
    location_hint=location_hint,
    edits=edits,
    rationale=rationale,
    evidence_tags=evidence_tags,
    source_evidence=source_evidence,
  )


def _parse_full_file_candidate(index: int, item: Any) -> AgentFullFileCandidate:
  if not isinstance(item, dict):
    raise ValueError(f"candidates[{index}] must be a JSON object")
  filename = item.get("filename")
  text = item.get("text")
  rationale = item.get("rationale")
  intended_stress = item.get("intended_stress")
  if not isinstance(filename, str) or not _is_safe_ll_filename(filename):
    raise ValueError(
      f"candidates[{index}].filename must be a safe .ll basename"
    )
  if not isinstance(text, str) or not text.strip():
    raise ValueError(f"candidates[{index}].text must be a non-empty string")
  if "```" in text:
    raise ValueError(f"candidates[{index}].text must not contain Markdown fences")
  if not isinstance(rationale, str) or not rationale.strip():
    raise ValueError(f"candidates[{index}].rationale must be a non-empty string")
  if not isinstance(intended_stress, str) or not intended_stress.strip():
    raise ValueError(
      f"candidates[{index}].intended_stress must be a non-empty string"
    )
  evidence_tags = _parse_candidate_evidence_tags(index, item.get("evidence_tags"))
  source_evidence = _parse_candidate_source_evidence(
    index, item.get("source_evidence")
  )
  return AgentFullFileCandidate(
    filename=filename,
    text=text,
    rationale=rationale,
    intended_stress=intended_stress,
    evidence_tags=evidence_tags,
    source_evidence=source_evidence,
  )


def _is_safe_ll_filename(filename: str) -> bool:
  return (
    bool(filename)
    and filename.endswith(".ll")
    and "/" not in filename
    and "\\" not in filename
    and ".." not in filename
    and not os.path.isabs(filename)
  )


def _parse_candidate_evidence_tags(index: int, value: Any) -> list[str] | None:
  if value is None:
    return None
  if not isinstance(value, list):
    raise ValueError(
      f"candidates[{index}].evidence_tags must be a list of strings"
    )
  for tag_index, tag in enumerate(value):
    if not isinstance(tag, str):
      raise ValueError(
        f"candidates[{index}].evidence_tags[{tag_index}] must be a string"
      )
  return value or None


def _parse_candidate_source_evidence(index: int, value: Any) -> str | None:
  if value is None:
    return None
  if not isinstance(value, str):
    raise ValueError(f"candidates[{index}].source_evidence must be a string")
  return value or None


def _parse_evidence_tags(index: int, value: Any) -> list[str] | None:
  if value is None:
    return None
  if not isinstance(value, list):
    raise ValueError(
      f"proposed_mutations[{index}].evidence_tags must be a list of strings"
    )
  for tag_index, tag in enumerate(value):
    if not isinstance(tag, str):
      raise ValueError(
        f"proposed_mutations[{index}].evidence_tags[{tag_index}] must be a string"
      )
  return value or None


def _parse_source_evidence(index: int, value: Any) -> str | None:
  if value is None:
    return None
  if not isinstance(value, str):
    raise ValueError(f"proposed_mutations[{index}].source_evidence must be a string")
  return value or None


def _parse_mutation_edits(index: int, item: dict[str, Any]) -> list[AgentMutationEdit]:
  raw_edits = item.get("edits")
  if raw_edits is None:
    raw_edits = [
      {
        "old_value": item.get("old_value"),
        "new_value": item.get("new_value"),
      }
    ]
  if not isinstance(raw_edits, list) or not raw_edits:
    raise ValueError(f"proposed_mutations[{index}].edits must be a non-empty list")

  edits = []
  for edit_index, edit in enumerate(raw_edits):
    if not isinstance(edit, dict):
      raise ValueError(
        f"proposed_mutations[{index}].edits[{edit_index}] must be a JSON object"
      )
    old_value = edit.get("old_value")
    new_value = edit.get("new_value")
    occurrence = edit.get("occurrence")
    if not isinstance(old_value, int) or isinstance(old_value, bool):
      raise ValueError(
        f"proposed_mutations[{index}].edits[{edit_index}].old_value must be an integer"
      )
    if not isinstance(new_value, int) or isinstance(new_value, bool):
      raise ValueError(
        f"proposed_mutations[{index}].edits[{edit_index}].new_value must be an integer"
      )
    if occurrence is not None and (
      not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1
    ):
      raise ValueError(
        f"proposed_mutations[{index}].edits[{edit_index}].occurrence must be a positive integer"
      )
    edits.append(
      AgentMutationEdit(
        old_value=old_value,
        new_value=new_value,
        occurrence=occurrence,
      )
    )
  return edits


def _allowed_immediate_values(profile: MutationProfile) -> set[int]:
  axis = profile.mutation_axes.get("immediates", {})
  if not isinstance(axis, dict) or not axis.get("enabled", False):
    return set()
  values = axis.get("values", [])
  if not isinstance(values, list):
    return set()
  return {value for value in values if isinstance(value, int) and not isinstance(value, bool)}


def _rejection_reason(
  mutation: AgentMutationProposal, allowed_values: set[int]
) -> str | None:
  if mutation.axis not in {
    "immediates",
    "immediate_boundary",
    "shift_amount_boundary",
  } | KERNEL_INFORMED_AGENT_AXES:
    return f"unsupported mutation axis for current agent generator: {mutation.axis}"
  for edit in mutation.edits:
    if edit.old_value == edit.new_value:
      return "old_value and new_value are identical"
    if edit.new_value not in allowed_values:
      return f"new_value {edit.new_value} is not allowed by profile immediates.values"
  return None


def _find_seed_record(
  test_index: dict[str, Any] | None, seed_relative: str
) -> dict[str, Any] | None:
  for record in (test_index or {}).get("tests", []):
    if isinstance(record, dict) and record.get("path") == seed_relative:
      return record
  return None


def _extract_run_lines(text: str) -> list[str]:
  return [
    line.strip()
    for line in text.splitlines()
    if re.match(r"^\s*[;#]\s*RUN:", line)
  ]


def _extract_check_prefixes(text: str) -> list[str]:
  prefixes = []
  seen = set()
  for line in text.splitlines():
    match = re.match(r"^\s*[;#]\s*([A-Za-z0-9_.$-]*CHECK[A-Za-z0-9_.$-]*):", line)
    if match is None:
      continue
    prefix = match.group(1)
    if prefix in seen:
      continue
    seen.add(prefix)
    prefixes.append(prefix)
  return prefixes


def _select_nearby_tests(
  llvm_root: Path,
  seed_relative: str,
  limit: int,
) -> list[dict[str, Any]]:
  limit = max(limit, 0)
  if limit == 0:
    return []
  seed_path = Path(seed_relative)
  seed_directory = llvm_root / seed_path.parent
  if not seed_directory.is_dir():
    return []

  records = []
  for path in sorted(seed_directory.glob("*.ll")):
    relative_path = path.relative_to(llvm_root).as_posix()
    if relative_path == seed_relative:
      continue
    text = path.read_text(encoding="utf-8", errors="replace")
    records.append(
      {
        "path": relative_path,
        "run_lines": _extract_run_lines(text),
        "text": _nearby_test_preview(text),
      }
    )
    if len(records) >= limit:
      break
  return records


def _nearby_test_preview(text: str) -> str:
  preview = "".join(text.splitlines(keepends=True)[:80])
  return preview[:12000]


def _kernel_evidence_score(record: dict[str, Any], profile_name: str) -> int:
  if profile_name == "machine_addropt":
    return _machine_addropt_kernel_score(record)
  if profile_name == "globalisel":
    return _globalisel_kernel_score(record)
  return _fallback_kernel_score(record)


def _machine_addropt_kernel_score(record: dict[str, Any]) -> int:
  usage = _record_usage(record)
  score = 0
  score += 5 * len(_matching_edge_kinds(
    record,
    {"addr_exp_boundary", "dma_length_boundary", "stride_boundary", "tile_boundary"},
  ))
  if usage.get("dma_calls"):
    score += 3
  score += 2 * len(_matching_relation_kinds(
    usage,
    {"address_exponent", "dma_length_boundary", "stride_boundary", "memory_space_pair"},
  ))
  if set(_list_or_empty(usage.get("memory_spaces"))) & {"HBM", "VMEM", "CMEM", "SMEM"}:
    score += 2
  if set(_list_or_empty(usage.get("constants"))) & {7, 128, 256, 1024}:
    score += 1
  return score


def _globalisel_kernel_score(record: dict[str, Any]) -> int:
  usage = _record_usage(record)
  score = 0
  if usage.get("vector_types") or usage.get("intrinsics"):
    score += 5
  score += 4 * len(_matching_edge_kinds(
    record,
    {"vector_lane_boundary", "mask_boundary"},
  ))
  if record.get("dtype_hints"):
    score += 3
  score += 2 * len(_matching_relation_kinds(
    usage,
    {"mask_boundary", "tile_boundary"},
  ))
  if set(_list_or_empty(usage.get("constants"))) & {8, 32, 128, 1024}:
    score += 1
  return score


def _fallback_kernel_score(record: dict[str, Any]) -> int:
  usage = _record_usage(record)
  score = len(_list_or_empty(record.get("edge_hints")))
  score += len(_list_or_empty(usage.get("relations")))
  if usage.get("dma_calls") or usage.get("vector_types") or usage.get("intrinsics"):
    score += 1
  return score


def _matching_edge_kinds(record: dict[str, Any], kinds: set[str]) -> set[str]:
  result = set()
  for hint in _list_or_empty(record.get("edge_hints")):
    if isinstance(hint, dict) and hint.get("kind") in kinds:
      result.add(str(hint["kind"]))
  return result


def _matching_relation_kinds(usage: dict[str, Any], kinds: set[str]) -> set[str]:
  result = set()
  for relation in _list_or_empty(usage.get("relations")):
    if isinstance(relation, dict) and relation.get("kind") in kinds:
      result.add(str(relation["kind"]))
  return result


def _prune_kernel_evidence(record: dict[str, Any], score: int) -> dict[str, Any]:
  usage = _record_usage(record)
  return {
    "name": record.get("name"),
    "source": record.get("source"),
    "category": record.get("category"),
    "dtype_hints": _list_or_empty(record.get("dtype_hints")),
    "features": _list_or_empty(record.get("features")),
    "usage": {
      "dma_calls": _limit_dict_list(usage.get("dma_calls"), 4),
      "memory_spaces": _list_or_empty(usage.get("memory_spaces")),
      "vector_types": _list_or_empty(usage.get("vector_types"))[:8],
      "intrinsics": _list_or_empty(usage.get("intrinsics"))[:12],
      "constants": _list_or_empty(usage.get("constants"))[:16],
      "relations": _limit_dict_list(usage.get("relations"), 8),
    },
    "edge_hints": _sorted_edge_hints(record.get("edge_hints"))[:12],
    "selection_score": score,
  }


def _record_usage(record: dict[str, Any]) -> dict[str, Any]:
  usage = record.get("usage")
  return usage if isinstance(usage, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
  return value if isinstance(value, list) else []


def _limit_dict_list(value: Any, limit: int) -> list[dict[str, Any]]:
  return [item for item in _list_or_empty(value) if isinstance(item, dict)][:limit]


def _sorted_edge_hints(value: Any) -> list[dict[str, Any]]:
  hints = [item for item in _list_or_empty(value) if isinstance(item, dict)]
  return sorted(
    hints,
    key=lambda hint: (
      str(hint.get("kind") or ""),
      str(hint.get("source") or ""),
      hint.get("base") if isinstance(hint.get("base"), int) else 0,
    ),
  )


def _select_spec_records(
  spec_index: dict[str, Any] | None, spec_sources: list[str]
) -> list[dict[str, Any]]:
  records = []
  wanted_sources = set(spec_sources)
  for record in (spec_index or {}).get("records", []):
    if not isinstance(record, dict):
      continue
    if wanted_sources and record.get("source") not in wanted_sources:
      continue
    records.append(record)
    if len(records) >= 20:
      break
  return records


def _select_source_records(
  td_index: dict[str, Any] | None, source_files: list[str]
) -> dict[str, list[dict[str, Any]]]:
  wanted_sources = set(source_files)
  result: dict[str, list[dict[str, Any]]] = {
    "intrinsics": [],
    "builtins": [],
    "instructions": [],
    "references": [],
  }
  for key in result:
    for record in (td_index or {}).get(key, []):
      if not isinstance(record, dict):
        continue
      source = record.get("source")
      if wanted_sources and source not in wanted_sources:
        continue
      result[key].append(record)
      if len(result[key]) >= 20:
        break
  return result
