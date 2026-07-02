from __future__ import annotations

import json
import re
import shutil
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlc_testforge.paths import discover_environment
from dlc_testforge.profiles import get_profile
from dlc_testforge.run_command import run_command
from dlc_testforge.schemas import CommandResult


CHECK_RE = re.compile(r"^\s*[;#]\s*[A-Za-z0-9_.-]*CHECK[A-Za-z0-9_.-]*:", re.MULTILINE)
RUN_RE = re.compile(r"^\s*[;#]\s*RUN:", re.MULTILINE)


@dataclass(frozen=True)
class ValidationStep:
  level: str
  status: str
  command: str | None = None
  exit_code: int | None = None
  stdout_path: Path | None = None
  stderr_path: Path | None = None
  duration_ms: int | None = None
  reason: str | None = None
  details: dict[str, Any] | None = None

  def to_dict(self) -> dict[str, Any]:
    data: dict[str, Any] = {
      "level": self.level,
      "status": self.status,
      "command": self.command,
      "exit_code": self.exit_code,
      "stdout_path": str(self.stdout_path) if self.stdout_path is not None else None,
      "stderr_path": str(self.stderr_path) if self.stderr_path is not None else None,
      "duration_ms": self.duration_ms,
      "reason": self.reason,
    }
    if self.details is not None:
      data["details"] = self.details
    return data


@dataclass(frozen=True)
class ValidationReport:
  candidate: Path
  profile: str
  overall_status: str
  steps: list[ValidationStep]
  suggested_suite_command: str
  out_dir: Path

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "candidate": str(self.candidate),
      "profile": self.profile,
      "overall_status": self.overall_status,
      "steps": [step.to_dict() for step in self.steps],
      "suggested_suite_command": self.suggested_suite_command,
      "out_dir": str(self.out_dir),
    }


def validate_candidate(
  llvm_root: Path,
  candidate: Path,
  profile_name: str,
  out_dir: Path,
  *,
  profiles_dir: Path | None = None,
  timeout: int = 30,
  stage_in_tree: bool = False,
) -> ValidationReport:
  llvm_root = llvm_root.expanduser().resolve(strict=False)
  candidate = candidate.expanduser().resolve(strict=False)
  out_dir = out_dir.expanduser().resolve(strict=False)
  if not candidate.is_file():
    raise ValueError(f"candidate file not found: {candidate}")

  env = discover_environment(llvm_root, archer_reference=None, check_versions=False)
  if not env.ok:
    missing = ", ".join(env.missing_required)
    raise ValueError(f"LLVM environment is incomplete: {missing}")

  profile = get_profile(profile_name, profiles_dir)
  required_levels = _required_levels(profile.validation)
  logs_dir = out_dir / "logs"
  logs_dir.mkdir(parents=True, exist_ok=True)

  steps = [
    _syntax_step(env.tools["llvm-as"].path.path, candidate, out_dir, logs_dir, timeout),
    _command_step(env.tools["llc"].path.path, candidate, profile.commands, logs_dir, timeout),
    _spec_step(),
    _filecheck_step(
      env.tools["llvm-lit"].path.path,
      llvm_root,
      candidate,
      logs_dir,
      timeout,
      stage_in_tree,
    ),
    _suite_step(llvm_root),
  ]
  report = ValidationReport(
    candidate=candidate,
    profile=profile.name,
    overall_status=_overall_status(steps, required_levels),
    steps=steps,
    suggested_suite_command=f"ninja -C {llvm_root / 'build'} check-llvm-codegen-dlc",
    out_dir=out_dir,
  )
  _write_status(report, out_dir / "status.json")
  return report


def validation_summary(report: ValidationReport) -> dict[str, Any]:
  return {
    "candidate": str(report.candidate),
    "profile": report.profile,
    "overall_status": report.overall_status,
    "status": str(report.out_dir / "status.json"),
    "step_count": len(report.steps),
    "failed_steps": [
      step.level for step in report.steps if step.status == "fail"
    ],
    "needs_check_steps": [
      step.level
      for step in report.steps
      if step.status in {"needs-checks", "skipped", "unknown"}
    ],
  }


def _syntax_step(
  llvm_as: Path,
  candidate: Path,
  out_dir: Path,
  logs_dir: Path,
  timeout: int,
) -> ValidationStep:
  if candidate.suffix == ".mir":
    return ValidationStep(
      level="syntax",
      status="skipped",
      reason="mir syntax uses profile command in Phase 6 MVP",
    )
  if candidate.suffix != ".ll":
    return ValidationStep(
      level="syntax",
      status="skipped",
      reason=f"unsupported syntax check for suffix {candidate.suffix}",
    )
  command = [str(llvm_as), str(candidate), "-o", str(out_dir / "syntax.bc")]
  return _command_result_step("syntax", command, logs_dir, timeout)


def _command_step(
  llc: Path,
  candidate: Path,
  commands: dict[str, Any],
  logs_dir: Path,
  timeout: int,
) -> ValidationStep:
  base_commands = commands.get("base", [])
  if not base_commands:
    return ValidationStep(
      level="command",
      status="skipped",
      reason="profile has no base command",
    )
  command_text = str(base_commands[0]).format(llc=str(llc), input=str(candidate))
  command = shlex.split(command_text)
  return _command_result_step("command", command, logs_dir, timeout)


def _spec_step() -> ValidationStep:
  return ValidationStep(
    level="spec",
    status="unknown",
    reason="spec consistency check is not implemented in Phase 6 MVP",
  )


def _filecheck_step(
  llvm_lit: Path,
  llvm_root: Path,
  candidate: Path,
  logs_dir: Path,
  timeout: int,
  stage_in_tree: bool,
) -> ValidationStep:
  text = candidate.read_text(encoding="utf-8", errors="replace")
  if not RUN_RE.search(text) or not CHECK_RE.search(text):
    return ValidationStep(
      level="filecheck",
      status="needs-checks",
      reason="candidate has no RUN/CHECK lines",
    )
  if not _is_under(candidate, llvm_root / "llvm" / "test"):
    if stage_in_tree:
      return _staged_filecheck_step(llvm_lit, llvm_root, candidate, logs_dir, timeout)
    return ValidationStep(
      level="filecheck",
      status="skipped",
      reason="lit_unavailable: candidate is outside LLVM test tree",
    )
  return _command_result_step(
    "filecheck", [str(llvm_lit), "-sv", str(candidate)], logs_dir, timeout
  )


def _staged_filecheck_step(
  llvm_lit: Path,
  llvm_root: Path,
  candidate: Path,
  logs_dir: Path,
  timeout: int,
) -> ValidationStep:
  staging_dir = llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / ".dlc-testforge-staging"
  staging_dir.mkdir(parents=True, exist_ok=True)
  staged_candidate = staging_dir / f"{candidate.stem}-{uuid.uuid4().hex[:12]}{candidate.suffix}"
  shutil.copyfile(candidate, staged_candidate)
  try:
    step = _command_result_step(
      "filecheck", [str(llvm_lit), "-sv", str(staged_candidate)], logs_dir, timeout
    )
    return ValidationStep(
      level=step.level,
      status=step.status,
      command=step.command,
      exit_code=step.exit_code,
      stdout_path=step.stdout_path,
      stderr_path=step.stderr_path,
      duration_ms=step.duration_ms,
      reason=step.reason,
      details={
        "staged_from": str(candidate),
        "staged_path": str(staged_candidate),
        "staged_removed": True,
      },
    )
  finally:
    staged_candidate.unlink(missing_ok=True)


def _suite_step(llvm_root: Path) -> ValidationStep:
  return ValidationStep(
    level="suite",
    status="skipped",
    command=f"ninja -C {llvm_root / 'build'} check-llvm-codegen-dlc",
    reason="suite target is not run automatically in Phase 6 MVP",
  )


def _command_result_step(
  level: str, command: list[str], logs_dir: Path, timeout: int
) -> ValidationStep:
  result = run_command(command, timeout=timeout)
  stdout_path = logs_dir / f"{level}.stdout"
  stderr_path = logs_dir / f"{level}.stderr"
  _write_command_logs(result, stdout_path, stderr_path)
  return ValidationStep(
    level=level,
    status="pass" if result.exit_code == 0 and not result.timed_out else "fail",
    command=" ".join(shlex.quote(arg) for arg in command),
    exit_code=result.exit_code,
    stdout_path=stdout_path,
    stderr_path=stderr_path,
    duration_ms=result.duration_ms,
    reason="timeout" if result.timed_out else None,
  )


def _write_command_logs(
  result: CommandResult, stdout_path: Path, stderr_path: Path
) -> None:
  stdout_path.parent.mkdir(parents=True, exist_ok=True)
  stderr_path.parent.mkdir(parents=True, exist_ok=True)
  stdout_path.write_text(result.stdout, encoding="utf-8")
  stderr_path.write_text(result.stderr, encoding="utf-8")


def _overall_status(steps: list[ValidationStep], required_levels: list[str]) -> str:
  statuses = [step.status for step in steps]
  if "fail" in statuses:
    return "fail"
  by_level = {step.level: step for step in steps}
  for level in required_levels:
    step = by_level.get(level)
    if step is None or step.status in {"needs-checks", "skipped", "unknown"}:
      return "needs-checks"
  if "needs-checks" in statuses:
    return "needs-checks"
  return "pass"


def _required_levels(validation: dict[str, Any]) -> list[str]:
  levels = validation.get("required_levels", ["syntax", "command", "filecheck"])
  if not isinstance(levels, list) or not all(isinstance(level, str) for level in levels):
    return ["syntax", "command", "filecheck"]
  return list(levels)


def _write_status(report: ValidationReport, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _is_under(path: Path, directory: Path) -> bool:
  try:
    path.relative_to(directory)
    return True
  except ValueError:
    return False
