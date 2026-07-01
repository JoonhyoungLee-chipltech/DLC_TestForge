from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

  def to_dict(self) -> dict[str, Any]:
    return {
      "path": self.path,
      "seed": self.seed,
      "profile": self.profile,
      "mutation_axis": self.mutation_axis,
      "source_value": self.source_value,
      "new_value": self.new_value,
      "line": self.line,
      "comment": self.comment,
    }


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
) -> WorkspaceManifest:
  if max_candidates < 1:
    raise ValueError("max-candidates must be at least 1")

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

  write_index(build_index(llvm_root), inputs_dir / "test-index.json")
  write_spec_index(build_spec_index(llvm_root), inputs_dir / "spec-index.json")
  write_td_index(build_td_index(llvm_root), inputs_dir / "td-index.json")

  candidates = []
  if not dry_run:
    seed_text = seed_path.read_text(encoding="utf-8")
    candidates = _generate_candidates(
      seed_text,
      seed_relative,
      profile.name,
      profile.mutation_axes,
      candidates_dir,
      max_candidates,
    )

  created_at = datetime.now(timezone.utc).replace(microsecond=0)
  manifest = WorkspaceManifest(
    schema_version=1,
    run_id=f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{profile.name}",
    llvm_root=llvm_root,
    profile=profile.name,
    seed=seed_relative,
    mode="manual",
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
