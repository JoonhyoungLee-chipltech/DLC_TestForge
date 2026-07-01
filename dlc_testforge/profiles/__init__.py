from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = [
  "name",
  "description",
  "seed_selectors",
  "commands",
  "validation",
  "mutation_axes",
  "bug_scout",
]


@dataclass(frozen=True)
class MutationProfile:
  name: str
  description: str
  seed_selectors: dict[str, Any]
  commands: dict[str, Any]
  validation: dict[str, Any]
  mutation_axes: dict[str, Any]
  bug_scout: dict[str, Any]
  source_files: list[str] = field(default_factory=list)
  spec_sources: list[str] = field(default_factory=list)
  notes: list[str] = field(default_factory=list)
  path: Path | None = None

  def to_dict(self) -> dict[str, Any]:
    data = {
      "name": self.name,
      "description": self.description,
      "seed_selectors": self.seed_selectors,
      "commands": self.commands,
      "validation": self.validation,
      "mutation_axes": self.mutation_axes,
      "bug_scout": self.bug_scout,
      "source_files": self.source_files,
      "spec_sources": self.spec_sources,
      "notes": self.notes,
    }
    if self.path is not None:
      data["path"] = str(self.path)
    return data


def load_profiles(profiles_dir: Path | None = None) -> list[MutationProfile]:
  root = _profiles_dir(profiles_dir)
  if not root.is_dir():
    raise ValueError(f"profile directory not found: {root}")

  profiles = []
  for path in sorted(root.glob("*.yaml")):
    profiles.append(_load_profile(path))

  names = [profile.name for profile in profiles]
  duplicates = sorted({name for name in names if names.count(name) > 1})
  if duplicates:
    raise ValueError(f"duplicate profile name(s): {', '.join(duplicates)}")

  return sorted(profiles, key=lambda profile: profile.name)


def get_profile(
  name: str, profiles_dir: Path | None = None
) -> MutationProfile:
  for profile in load_profiles(profiles_dir):
    if profile.name == name:
      return profile
  raise ValueError(f"profile not found: {name}")


def profile_summary(profile: MutationProfile) -> dict[str, Any]:
  return {
    "name": profile.name,
    "description": profile.description,
    "seed_count": len(profile.seed_selectors.get("paths", [])),
    "features": list(profile.seed_selectors.get("features", [])),
    "source_count": len(profile.source_files),
    "spec_source_count": len(profile.spec_sources),
  }


def _profiles_dir(profiles_dir: Path | None) -> Path:
  if profiles_dir is not None:
    return profiles_dir.expanduser().resolve(strict=False)
  return Path(__file__).resolve().parent


def _load_profile(path: Path) -> MutationProfile:
  try:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
  except yaml.YAMLError as exc:
    raise ValueError(f"invalid YAML in {path}: {exc}") from exc

  if not isinstance(data, dict):
    raise ValueError(f"profile must be a YAML mapping: {path}")

  missing = [field for field in REQUIRED_FIELDS if field not in data]
  if missing:
    raise ValueError(f"profile {path} missing required field(s): {', '.join(missing)}")

  _require_string(path, data, "name")
  _require_string(path, data, "description")
  for field_name in [
    "seed_selectors",
    "commands",
    "validation",
    "mutation_axes",
    "bug_scout",
  ]:
    _require_mapping(path, data, field_name)

  return MutationProfile(
    name=data["name"],
    description=data["description"],
    seed_selectors=dict(data["seed_selectors"]),
    commands=dict(data["commands"]),
    validation=dict(data["validation"]),
    mutation_axes=dict(data["mutation_axes"]),
    bug_scout=dict(data["bug_scout"]),
    source_files=_optional_string_list(path, data, "source_files"),
    spec_sources=_optional_string_list(path, data, "spec_sources"),
    notes=_optional_string_list(path, data, "notes"),
    path=path,
  )


def _require_string(path: Path, data: dict[str, Any], field_name: str) -> None:
  if not isinstance(data[field_name], str) or not data[field_name].strip():
    raise ValueError(f"profile {path} field {field_name} must be a non-empty string")


def _require_mapping(path: Path, data: dict[str, Any], field_name: str) -> None:
  if not isinstance(data[field_name], dict):
    raise ValueError(f"profile {path} field {field_name} must be a mapping")


def _optional_string_list(
  path: Path, data: dict[str, Any], field_name: str
) -> list[str]:
  value = data.get(field_name, [])
  if value is None:
    return []
  if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    raise ValueError(f"profile {path} field {field_name} must be a list of strings")
  return list(value)
