from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dlc_testforge.agent import (
  AgentMutationProposal,
  build_agent_context,
  filter_agent_mutations,
  load_agent_proposal,
  request_agent_proposal,
  select_kernel_evidence,
  write_agent_json,
)
from dlc_testforge.extract_spec import build_spec_index, write_spec_index
from dlc_testforge.extract_td import build_td_index, write_td_index
from dlc_testforge.index import build_index, write_index
from dlc_testforge.profiles import get_profile


TOOL_VERSION = "0.1"
INTEGER_RE = re.compile(r"(?<![%@A-Za-z0-9_.])-?\d+\b")


@dataclass(frozen=True)
class CandidateRecord:
  path: str
  seed: str
  profile: str
  mutation_axis: str
  source_value: int
  new_value: int
  line: int
  comment: str
  rationale: str | None = None
  edits: list[dict[str, Any]] | None = None

  def to_dict(self) -> dict[str, Any]:
    data = {
      "path": self.path,
      "seed": self.seed,
      "profile": self.profile,
      "mutation_axis": self.mutation_axis,
      "source_value": self.source_value,
      "new_value": self.new_value,
      "line": self.line,
      "comment": self.comment,
    }
    if self.rationale is not None:
      data["rationale"] = self.rationale
    if self.edits is not None:
      data["edits"] = self.edits
    return data


@dataclass(frozen=True)
class WorkspaceManifest:
  schema_version: int
  run_id: str
  llvm_root: Path
  profile: str
  seed: str
  mode: str
  dry_run: bool
  created_at: str
  tool_version: str
  workspace: Path
  inputs: dict[str, str]
  candidate_count: int
  candidates: list[CandidateRecord]

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": self.schema_version,
      "run_id": self.run_id,
      "llvm_root": str(self.llvm_root),
      "profile": self.profile,
      "seed": self.seed,
      "mode": self.mode,
      "dry_run": self.dry_run,
      "created_at": self.created_at,
      "tool_version": self.tool_version,
      "workspace": str(self.workspace),
      "inputs": self.inputs,
      "candidate_count": self.candidate_count,
      "candidates": [candidate.to_dict() for candidate in self.candidates],
    }


def create_workspace(
  llvm_root: Path,
  profile_name: str,
  seed: str,
  out_dir: Path,
  *,
  dry_run: bool,
  profiles_dir: Path | None = None,
  max_candidates: int = 10,
  mode: str = "manual",
  agent_proposal: Path | None = None,
  agent_model: str | None = None,
  agent_endpoint: str | None = None,
  kernel_usage_index: Path | None = None,
) -> WorkspaceManifest:
  if max_candidates < 1:
    raise ValueError("max-candidates must be at least 1")
  if mode not in {"manual", "agent"}:
    raise ValueError("mode must be one of: manual, agent")

  llvm_root = llvm_root.expanduser().resolve(strict=False)
  out_dir = out_dir.expanduser().resolve(strict=False)
  profile = get_profile(profile_name, profiles_dir)
  seed_relative = _validate_seed_relative(seed)
  seed_path = llvm_root / seed_relative

  if not seed_path.is_file():
    raise ValueError(f"seed file not found: {seed_relative}")
  if seed_relative not in profile.seed_selectors.get("paths", []):
    raise ValueError(
      f"seed {seed_relative} is not allowed by profile {profile.name}"
    )
  if out_dir.exists() and any(out_dir.iterdir()):
    raise ValueError(f"output directory is not empty: {out_dir}")

  inputs_dir = out_dir / "inputs"
  candidates_dir = out_dir / "candidates"
  results_dir = out_dir / "results"
  reports_dir = out_dir / "reports"
  for directory in [inputs_dir, candidates_dir, results_dir, reports_dir]:
    directory.mkdir(parents=True, exist_ok=True)

  seed_input_name = seed_path.name
  seed_input_path = inputs_dir / seed_input_name
  profile_input_path = inputs_dir / "profile.yaml"
  shutil.copyfile(seed_path, seed_input_path)
  if profile.path is None:
    raise ValueError(f"profile {profile.name} has no source path")
  shutil.copyfile(profile.path, profile_input_path)

  kernel_usage_input_path = None
  selected_kernel_evidence = None
  if kernel_usage_index is not None:
    kernel_usage_index_path = kernel_usage_index.expanduser().resolve(strict=False)
    kernel_usage_data = json.loads(
      kernel_usage_index_path.read_text(encoding="utf-8")
    )
    kernel_usage_input_path = inputs_dir / "kernel-usage-index.json"
    shutil.copyfile(kernel_usage_index_path, kernel_usage_input_path)
    selected_kernel_evidence = select_kernel_evidence(
      kernel_usage_data,
      profile.name,
      seed_relative,
    )

  test_index = build_index(llvm_root)
  spec_index = build_spec_index(llvm_root)
  td_index = build_td_index(llvm_root)
  write_index(test_index, inputs_dir / "test-index.json")
  write_spec_index(spec_index, inputs_dir / "spec-index.json")
  write_td_index(td_index, inputs_dir / "td-index.json")

  candidates = []
  seed_text = seed_path.read_text(encoding="utf-8")
  if mode == "agent":
    context = build_agent_context(
      profile,
      seed_relative,
      seed_text,
      max_candidates,
      test_index=test_index.to_dict(),
      spec_index=spec_index.to_dict(),
      td_index=td_index.to_dict(),
      kernel_usage_evidence=selected_kernel_evidence,
    )
    write_agent_json(inputs_dir / "agent-context.json", context)

  if not dry_run:
    if mode == "manual":
      candidates = _generate_candidates(
        seed_text,
        seed_relative,
        profile.name,
        profile.mutation_axes,
        candidates_dir,
        max_candidates,
      )
    else:
      if agent_proposal is not None:
        proposal = load_agent_proposal(agent_proposal)
      else:
        proposal = request_agent_proposal(
          context,
          model=agent_model,
          endpoint=agent_endpoint,
        )
      write_agent_json(inputs_dir / "agent-proposal.json", proposal.to_dict())
      accepted, rejected = filter_agent_mutations(
        proposal, profile, seed_relative, max_candidates
      )
      candidates, application_rejections = _generate_agent_candidates(
        seed_text,
        seed_relative,
        profile.name,
        accepted,
        candidates_dir,
      )
      write_agent_json(
        results_dir / "agent-rejections.json",
        {
          "rejection_count": len(rejected) + len(application_rejections),
          "rejections": [
            entry.to_dict() for entry in rejected
          ] + application_rejections,
        },
      )

  created_at = datetime.now(timezone.utc).replace(microsecond=0)
  manifest = WorkspaceManifest(
    schema_version=1,
    run_id=f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{profile.name}",
    llvm_root=llvm_root,
    profile=profile.name,
    seed=seed_relative,
    mode=mode,
    dry_run=dry_run,
    created_at=created_at.isoformat().replace("+00:00", "Z"),
    tool_version=TOOL_VERSION,
    workspace=out_dir,
    inputs={
      "seed": _relative_to_workspace(seed_input_path, out_dir),
      "profile": _relative_to_workspace(profile_input_path, out_dir),
      "test_index": "inputs/test-index.json",
      "spec_index": "inputs/spec-index.json",
      "td_index": "inputs/td-index.json",
      **(
        {"kernel_usage_index": "inputs/kernel-usage-index.json"}
        if kernel_usage_input_path is not None
        else {}
      ),
      **({"agent_context": "inputs/agent-context.json"} if mode == "agent" else {}),
      **(
        {"agent_proposal": "inputs/agent-proposal.json"}
        if mode == "agent" and not dry_run
        else {}
      ),
    },
    candidate_count=len(candidates),
    candidates=candidates,
  )
  _write_manifest(manifest, out_dir / "manifest.json")
  return manifest


def generation_summary(manifest: WorkspaceManifest) -> dict[str, Any]:
  return {
    "status": "dry-run" if manifest.dry_run else "generated",
    "workspace": str(manifest.workspace),
    "profile": manifest.profile,
    "seed": manifest.seed,
    "candidate_count": manifest.candidate_count,
    "manifest": str(manifest.workspace / "manifest.json"),
  }


def _validate_seed_relative(seed: str) -> str:
  path = Path(seed)
  if path.is_absolute() or not seed or any(part == ".." for part in path.parts):
    raise ValueError(f"seed must be a relative path inside llvm-root: {seed}")
  return path.as_posix()


def _generate_candidates(
  seed_text: str,
  seed_relative: str,
  profile_name: str,
  mutation_axes: dict[str, Any],
  candidates_dir: Path,
  max_candidates: int,
) -> list[CandidateRecord]:
  if not seed_relative.endswith(".ll"):
    return []

  immediate_axis = mutation_axes.get("immediates", {})
  if not isinstance(immediate_axis, dict) or not immediate_axis.get("enabled", False):
    return []
  values = immediate_axis.get("values", [])
  if not isinstance(values, list):
    return []
  mutation_values = [value for value in values if isinstance(value, int)]
  if not mutation_values:
    return []

  lines = seed_text.splitlines(keepends=True)
  candidates: list[CandidateRecord] = []

  for line_index, line in enumerate(lines):
    if not _is_mutable_ir_line(line):
      continue
    for match in INTEGER_RE.finditer(line):
      source_value = int(match.group(0))
      axis = _axis_for_line(line)
      for new_value in mutation_values:
        if new_value == source_value:
          continue
        candidate_number = len(candidates) + 1
        filename = f"candidate-{candidate_number:04d}.ll"
        comment = (
          f"; DLC-MUTATION: profile={profile_name} axis={axis} "
          f"source_value={source_value} new_value={new_value}"
        )
        candidate_lines = list(lines)
        candidate_lines[line_index] = (
          f"{comment}\n"
          f"{line[:match.start()]}{new_value}{line[match.end():]}"
        )
        candidate_path = candidates_dir / filename
        candidate_path.write_text("".join(candidate_lines), encoding="utf-8")
        candidates.append(
          CandidateRecord(
            path=f"candidates/{filename}",
            seed=seed_relative,
            profile=profile_name,
            mutation_axis=axis,
            source_value=source_value,
            new_value=new_value,
            line=line_index + 1,
            comment=comment,
          )
        )
        if len(candidates) >= max_candidates:
          return candidates

  return candidates


def _generate_agent_candidates(
  seed_text: str,
  seed_relative: str,
  profile_name: str,
  proposals: list[AgentMutationProposal],
  candidates_dir: Path,
) -> tuple[list[CandidateRecord], list[dict[str, Any]]]:
  if not seed_relative.endswith(".ll"):
    return [], [
      {
        "index": index,
        "mutation": proposal.to_dict(),
        "reason": "current agent generator only supports .ll seeds",
      }
      for index, proposal in enumerate(proposals)
    ]

  lines = seed_text.splitlines(keepends=True)
  candidates: list[CandidateRecord] = []
  rejections: list[dict[str, Any]] = []
  used_locations: set[tuple[int, int, int]] = set()
  for index, proposal in enumerate(proposals):
    matches = _find_agent_mutation_locations(lines, proposal, used_locations)
    if matches is None:
      rejections.append(
        {
          "index": index,
          "mutation": proposal.to_dict(),
          "reason": "one or more old_value entries were not found on mutable seed lines for the requested axis",
        }
      )
      continue
    for line_index, start, end, _ in matches:
      used_locations.add((line_index, start, end))
    candidate_number = len(candidates) + 1
    filename = f"candidate-{candidate_number:04d}.ll"
    edit_summary = ",".join(
      f"{edit.old_value}->{edit.new_value}" for edit in proposal.edits
    )
    comment = (
      f"; DLC-MUTATION: profile={profile_name} mode=agent axis={proposal.axis} "
      f"edits={edit_summary}"
    )
    candidate_lines = list(lines)
    for edit_index, (line_index, start, end, edit) in enumerate(
      sorted(matches, key=lambda item: (item[0], item[1]), reverse=True)
    ):
      line = candidate_lines[line_index]
      candidate_lines[line_index] = f"{line[:start]}{edit.new_value}{line[end:]}"
      if edit_index == len(matches) - 1:
        candidate_lines[line_index] = f"{comment}\n{candidate_lines[line_index]}"
    candidate_path = candidates_dir / filename
    candidate_path.write_text("".join(candidate_lines), encoding="utf-8")
    first_line = min(line_index for line_index, _, _, _ in matches)
    candidates.append(
      CandidateRecord(
        path=f"candidates/{filename}",
        seed=seed_relative,
        profile=profile_name,
        mutation_axis=proposal.axis,
        source_value=proposal.old_value,
        new_value=proposal.new_value,
        line=first_line + 1,
        comment=comment,
        rationale=proposal.rationale,
        edits=[
          {
            "old_value": edit.old_value,
            "new_value": edit.new_value,
            "line": line_index + 1,
          }
          for line_index, _, _, edit in sorted(matches, key=lambda item: item[0])
        ],
      )
    )
  return candidates, rejections


def _find_agent_mutation_locations(
  lines: list[str],
  proposal: AgentMutationProposal,
  used_locations: set[tuple[int, int, int]],
) -> list[tuple[int, int, int, Any]] | None:
  matches: list[tuple[int, int, int, Any]] = []
  scoped_indices = set(_line_indices_for_hint(lines, proposal.location_hint))
  for edit in proposal.edits:
    match = _find_agent_edit_location(
      lines,
      proposal.axis,
      edit.old_value,
      edit.occurrence,
      scoped_indices,
      used_locations | {(line, start, end) for line, start, end, _ in matches},
    )
    if match is None:
      return None
    line_index, start, end = match
    matches.append((line_index, start, end, edit))
  return matches


def _find_agent_edit_location(
  lines: list[str],
  axis: str,
  old_value: int,
  occurrence: int | None,
  scoped_indices: set[int],
  used_locations: set[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
  seen = 0
  for line_index, line in enumerate(lines):
    if scoped_indices and line_index not in scoped_indices:
      continue
    if not _is_mutable_ir_line(line):
      continue
    if not _agent_axis_matches_line(axis, line):
      continue
    for match in INTEGER_RE.finditer(line):
      if int(match.group(0)) != old_value:
        continue
      seen += 1
      if occurrence is not None and seen != occurrence:
        continue
      location = (line_index, match.start(), match.end())
      if location in used_locations:
        continue
      return location
  return None


def _line_indices_for_hint(lines: list[str], location_hint: str) -> list[int]:
  match = re.search(r"@([A-Za-z_][A-Za-z0-9_.$-]*)", location_hint)
  if match is None:
    return []
  function_name = match.group(1)
  start = None
  brace_depth = 0
  for index, line in enumerate(lines):
    if start is None:
      if f"@{function_name}" in line and line.lstrip().startswith("define "):
        start = index
        brace_depth += line.count("{") - line.count("}")
      continue
    brace_depth += line.count("{") - line.count("}")
    if brace_depth <= 0:
      return list(range(start, index + 1))
  return list(range(start, len(lines))) if start is not None else []


def _agent_axis_matches_line(axis: str, line: str) -> bool:
  line_axis = _axis_for_line(line)
  if axis == "immediates":
    return True
  if axis == "immediate_boundary":
    return line_axis == "immediate_boundary"
  return axis == line_axis


def _is_mutable_ir_line(line: str) -> bool:
  stripped = line.strip()
  if not stripped or stripped.startswith(";"):
    return False
  if " RUN:" in line or " CHECK" in line or "DLC-MUTATION:" in line:
    return False
  if (
    stripped.endswith(":")
    or stripped.startswith("define ")
    or stripped.startswith("declare ")
  ):
    return False
  return "=" in line and INTEGER_RE.search(line) is not None


def _axis_for_line(line: str) -> str:
  if any(opcode in line for opcode in [" shl ", " lshr ", " ashr "]):
    return "shift_amount_boundary"
  return "immediate_boundary"


def _relative_to_workspace(path: Path, workspace: Path) -> str:
  return path.relative_to(workspace).as_posix()


def _write_manifest(manifest: WorkspaceManifest, path: Path) -> None:
  path.write_text(
    json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
