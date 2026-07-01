from __future__ import annotations

from pathlib import Path

from dlc_testforge.run_command import run_command
from dlc_testforge.schemas import EnvironmentReport, PathStatus, ToolStatus


REQUIRED_TOOLS = {
  "llc": "build/bin/llc",
  "llvm-lit": "build/bin/llvm-lit",
  "FileCheck": "build/bin/FileCheck",
  "llvm-as": "build/bin/llvm-as",
  "clang": "build/bin/clang",
}

REQUIRED_INPUTS = {
  "dlc_codegen_tests": ("llvm/test/CodeGen/DLC", "dir"),
  "dlc_spec_dir": ("docs/dlc_spec", "dir"),
  "dlc_target_src": ("llvm/lib/Target/DLC", "dir"),
  "dlc_intrinsics_td": ("llvm/include/llvm/IR/IntrinsicsDLC.td", "file"),
}


def _normalize(path: Path) -> Path:
  return path.expanduser().resolve(strict=False)


def discover_environment(
  llvm_root: Path,
  *,
  archer_reference: Path | None = Path("/root/references/Archer"),
  check_versions: bool = False,
) -> EnvironmentReport:
  llvm_root = _normalize(llvm_root)
  build_dir = llvm_root / "build"

  llvm_root_status = PathStatus.from_path("llvm_root", llvm_root, expected="dir")
  build_dir_status = PathStatus.from_path("build_dir", build_dir, expected="dir")

  tools: dict[str, ToolStatus] = {}
  for name, relative in REQUIRED_TOOLS.items():
    path_status = PathStatus.from_path(
      name, llvm_root / relative, expected="executable"
    )
    version_check = None
    if check_versions and path_status.ok:
      version_check = run_command([str(path_status.path), "--version"], timeout=10)
    tools[name] = ToolStatus(name=name, path=path_status, version_check=version_check)

  inputs: dict[str, PathStatus] = {}
  for name, (relative, expected) in REQUIRED_INPUTS.items():
    inputs[name] = PathStatus.from_path(name, llvm_root / relative, expected=expected)

  if archer_reference is not None:
    inputs["archer_reference"] = PathStatus.from_path(
      "archer_reference",
      _normalize(archer_reference),
      expected="dir",
      required=False,
    )

  missing_required = []
  for status in [llvm_root_status, build_dir_status, *inputs.values()]:
    if status.required and not status.ok:
      missing_required.append(status.name)
  for tool in tools.values():
    if not tool.ok:
      missing_required.append(tool.name)

  return EnvironmentReport(
    llvm_root=llvm_root_status,
    build_dir=build_dir_status,
    tools=tools,
    inputs=inputs,
    missing_required=missing_required,
  )
