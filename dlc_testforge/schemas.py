from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathStatus:
  name: str
  path: Path
  exists: bool
  is_file: bool
  is_dir: bool
  is_executable: bool
  expected: str
  required: bool = True

  @classmethod
  def from_path(
    cls, name: str, path: Path, *, expected: str, required: bool = True
  ) -> "PathStatus":
    return cls(
      name=name,
      path=path,
      exists=path.exists(),
      is_file=path.is_file(),
      is_dir=path.is_dir(),
      is_executable=path.is_file() and path.stat().st_mode & 0o111 != 0
      if path.exists()
      else False,
      expected=expected,
      required=required,
    )

  @property
  def ok(self) -> bool:
    if self.expected == "file":
      shape_ok = self.is_file
    elif self.expected == "dir":
      shape_ok = self.is_dir
    elif self.expected == "executable":
      shape_ok = self.is_file and self.is_executable
    else:
      shape_ok = self.exists
    return self.exists and shape_ok

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "path": str(self.path),
      "exists": self.exists,
      "is_file": self.is_file,
      "is_dir": self.is_dir,
      "is_executable": self.is_executable,
      "expected": self.expected,
      "required": self.required,
      "ok": self.ok,
    }


@dataclass(frozen=True)
class ToolStatus:
  name: str
  path: PathStatus
  version_check: "CommandResult | None" = None

  @property
  def ok(self) -> bool:
    return self.path.ok and (
      self.version_check is None
      or (not self.version_check.timed_out and self.version_check.exit_code == 0)
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "path": self.path.to_dict(),
      "version_check": self.version_check.to_dict()
      if self.version_check is not None
      else None,
      "ok": self.ok,
    }


@dataclass(frozen=True)
class CommandResult:
  argv: list[str]
  exit_code: int
  stdout: str
  stderr: str
  duration_ms: int
  timed_out: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "argv": self.argv,
      "exit_code": self.exit_code,
      "stdout": self.stdout,
      "stderr": self.stderr,
      "duration_ms": self.duration_ms,
      "timed_out": self.timed_out,
    }


@dataclass(frozen=True)
class EnvironmentReport:
  llvm_root: PathStatus
  build_dir: PathStatus
  tools: dict[str, ToolStatus]
  inputs: dict[str, PathStatus]
  missing_required: list[str] = field(default_factory=list)

  @property
  def ok(self) -> bool:
    return not self.missing_required

  def to_dict(self) -> dict[str, Any]:
    return {
      "llvm_root": self.llvm_root.to_dict(),
      "build_dir": self.build_dir.to_dict(),
      "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
      "inputs": {name: status.to_dict() for name, status in self.inputs.items()},
      "summary": {
        "ok": self.ok,
        "missing_required": self.missing_required,
      },
    }
