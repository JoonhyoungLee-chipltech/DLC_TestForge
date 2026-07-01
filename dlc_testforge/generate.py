from __future__ import annotations

import json
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
    }


def create_workspace(
  llvm_root: Path,
  profile_name: str,
  seed: str,
  out_dir: Path,
  *,
  dry_run: bool,
  profiles_dir: Path | None = None,
) -> WorkspaceManifest:
  if not dry_run:
    raise ValueError("non-dry-run generation is not implemented yet")

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

  created_at = datetime.now(timezone.utc).replace(microsecond=0)
  manifest = WorkspaceManifest(
    schema_version=1,
    run_id=f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{profile.name}",
    llvm_root=llvm_root,
    profile=profile.name,
    seed=seed_relative,
    mode="manual",
    dry_run=True,
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
    candidate_count=0,
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


def _relative_to_workspace(path: Path, workspace: Path) -> str:
  return path.relative_to(workspace).as_posix()


def _write_manifest(manifest: WorkspaceManifest, path: Path) -> None:
  path.write_text(
    json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
