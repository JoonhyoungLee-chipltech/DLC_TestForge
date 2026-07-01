from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlc_testforge.profiles import get_profile


DEFAULT_REQUIRED_LEVELS = ["syntax", "command", "filecheck"]
EXCERPT_LIMIT = 2000

ASSERTION_RE = re.compile(
  r"Assertion|assert|PLEASE submit a bug report|Stack dump|UNREACHABLE|LLVM ERROR",
  re.IGNORECASE,
)
CRASH_RE = re.compile(
  r"segmentation fault|segfault|core dumped|\bsignal\b|abort",
  re.IGNORECASE,
)


@dataclass(frozen=True)
class ClassificationReport:
  candidate: str
  profile: str
  state: str
  reason: str
  validation: Path
  evidence_steps: list[dict[str, Any]]
  stdout_excerpt: str
  stderr_excerpt: str
  requires_human_triage: bool

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "candidate": self.candidate,
      "profile": self.profile,
      "state": self.state,
      "reason": self.reason,
      "validation": str(self.validation),
      "evidence_steps": self.evidence_steps,
      "stdout_excerpt": self.stdout_excerpt,
      "stderr_excerpt": self.stderr_excerpt,
      "requires_human_triage": self.requires_human_triage,
    }


def classify_validation(
  status_path: Path, *, profiles_dir: Path | None = None
) -> ClassificationReport:
  status_path = status_path.expanduser().resolve(strict=False)
  data = _load_status(status_path)
  steps = _steps_by_level(data)
  required_levels = _required_levels(str(data.get("profile", "")), profiles_dir)

  state, reason, evidence_levels = _classify_steps(steps, required_levels)
  evidence_steps = [
    _step_evidence(steps[level]) for level in evidence_levels if level in steps
  ]
  stdout_excerpt, stderr_excerpt = _collect_excerpts(evidence_steps)

  return ClassificationReport(
    candidate=str(data.get("candidate", "")),
    profile=str(data.get("profile", "")),
    state=state,
    reason=reason,
    validation=status_path,
    evidence_steps=evidence_steps,
    stdout_excerpt=stdout_excerpt,
    stderr_excerpt=stderr_excerpt,
    requires_human_triage=state.startswith("bug-scout-"),
  )


def write_classification(report: ClassificationReport, out_path: Path) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(
    json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def classification_summary(report: ClassificationReport) -> dict[str, Any]:
  return {
    "candidate": report.candidate,
    "profile": report.profile,
    "state": report.state,
    "reason": report.reason,
    "validation": str(report.validation),
    "requires_human_triage": report.requires_human_triage,
  }


def _load_status(status_path: Path) -> dict[str, Any]:
  try:
    data = json.loads(status_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise ValueError(f"invalid validation JSON: {status_path}: {exc}") from exc
  if not isinstance(data, dict):
    raise ValueError(f"validation JSON must be an object: {status_path}")
  steps = data.get("steps")
  if not isinstance(steps, list):
    raise ValueError(f"validation JSON missing steps list: {status_path}")
  return data


def _required_levels(profile_name: str, profiles_dir: Path | None) -> list[str]:
  if not profile_name:
    return list(DEFAULT_REQUIRED_LEVELS)
  try:
    profile = get_profile(profile_name, profiles_dir)
  except ValueError:
    return list(DEFAULT_REQUIRED_LEVELS)
  levels = profile.validation.get("required_levels", DEFAULT_REQUIRED_LEVELS)
  if not isinstance(levels, list) or not all(isinstance(level, str) for level in levels):
    return list(DEFAULT_REQUIRED_LEVELS)
  return list(levels)


def _steps_by_level(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
  steps: dict[str, dict[str, Any]] = {}
  for step in data.get("steps", []):
    if isinstance(step, dict) and isinstance(step.get("level"), str):
      steps[step["level"]] = step
  return steps


def _classify_steps(
  steps: dict[str, dict[str, Any]], required_levels: list[str]
) -> tuple[str, str, list[str]]:
  syntax = steps.get("syntax")
  if syntax is not None and syntax.get("status") == "fail":
    return "rejected-invalid-ir", "syntax validation failed", ["syntax"]

  spec = steps.get("spec")
  if spec is not None and spec.get("status") == "fail":
    return "rejected-spec-conflict", "spec consistency check failed", ["spec"]

  command = steps.get("command")
  if command is not None:
    stderr = _read_step_text(command, "stderr_path")
    exit_code = _optional_int(command.get("exit_code"))
    if command.get("reason") == "timeout":
      return "bug-scout-timeout", "compiler command timed out", ["command"]
    if command.get("status") == "fail" and (
      _looks_like_signal(exit_code) or CRASH_RE.search(stderr)
    ):
      return "bug-scout-crash", "compiler command looks like a crash", ["command"]
    if command.get("status") == "fail" and ASSERTION_RE.search(stderr):
      return "bug-scout-assertion", "compiler command reported an assertion", ["command"]
    if command.get("status") == "fail" and exit_code is not None and exit_code != 0:
      return (
        "bug-scout-compile-failure",
        "compiler command exited nonzero",
        ["command"],
      )

  for level, step in steps.items():
    if step.get("status") == "needs-checks":
      return "needs-checks", f"{level} requires FileCheck coverage", [level]

  filecheck = steps.get("filecheck")
  if "filecheck" in required_levels and (
    filecheck is None or filecheck.get("status") in {"skipped", "needs-checks"}
  ):
    return "needs-checks", "required FileCheck validation did not pass", ["filecheck"]

  if all(_required_level_passed(steps, level) for level in required_levels):
    return (
      "accepted-regression-candidate",
      "all required validation levels passed",
      list(required_levels),
    )

  return "unknown", "validation result does not match a Phase 8 rule", list(steps)


def _required_level_passed(steps: dict[str, dict[str, Any]], level: str) -> bool:
  step = steps.get(level)
  return step is not None and step.get("status") == "pass"


def _step_evidence(step: dict[str, Any]) -> dict[str, Any]:
  return {
    "level": step.get("level"),
    "status": step.get("status"),
    "exit_code": step.get("exit_code"),
    "reason": step.get("reason"),
    "command": step.get("command"),
    "stdout_path": step.get("stdout_path"),
    "stderr_path": step.get("stderr_path"),
  }


def _collect_excerpts(evidence_steps: list[dict[str, Any]]) -> tuple[str, str]:
  stdout_parts = []
  stderr_parts = []
  for step in evidence_steps:
    stdout_parts.append(_read_step_text(step, "stdout_path"))
    stderr_parts.append(_read_step_text(step, "stderr_path"))
  return _truncate("\n".join(stdout_parts)), _truncate("\n".join(stderr_parts))


def _read_step_text(step: dict[str, Any], path_key: str) -> str:
  path_text = step.get(path_key)
  if not isinstance(path_text, str) or not path_text:
    return ""
  path = Path(path_text)
  if not path.is_file():
    return ""
  return path.read_text(encoding="utf-8", errors="replace")


def _truncate(text: str) -> str:
  if len(text) <= EXCERPT_LIMIT:
    return text
  return text[:EXCERPT_LIMIT]


def _optional_int(value: object) -> int | None:
  if isinstance(value, int):
    return value
  return None


def _looks_like_signal(exit_code: int | None) -> bool:
  if exit_code is None:
    return False
  return exit_code < 0 or exit_code >= 128
