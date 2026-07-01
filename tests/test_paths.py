from __future__ import annotations

import os
import sys

from dlc_testforge.paths import discover_environment
from dlc_testforge.run_command import run_command


def test_discover_environment_reports_missing_tools(tmp_path):
  llvm_root = tmp_path / "LLVM"
  llvm_root.mkdir()

  report = discover_environment(llvm_root, archer_reference=None)

  assert report.llvm_root.ok
  assert not report.ok
  assert "llc" in report.missing_required
  assert "llvm-lit" in report.missing_required
  assert "dlc_codegen_tests" in report.missing_required


def test_discover_environment_accepts_optional_archer_reference(tmp_path):
  llvm_root = tmp_path / "LLVM"
  archer = tmp_path / "Archer"
  llvm_root.mkdir()
  archer.mkdir()

  report = discover_environment(llvm_root, archer_reference=archer)

  assert report.inputs["archer_reference"].ok
  assert "archer_reference" not in report.missing_required


def test_discover_environment_identifies_complete_fake_tree(tmp_path):
  llvm_root = tmp_path / "LLVM"
  bin_dir = llvm_root / "build" / "bin"
  bin_dir.mkdir(parents=True)
  for tool in ["llc", "llvm-lit", "FileCheck", "llvm-as", "clang"]:
    path = bin_dir / tool
    path.write_text("#!/bin/sh\nprintf '%s version\\n' \"$0\"\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)

  (llvm_root / "llvm" / "test" / "CodeGen" / "DLC").mkdir(parents=True)
  (llvm_root / "docs" / "dlc_spec").mkdir(parents=True)
  (llvm_root / "llvm" / "lib" / "Target" / "DLC").mkdir(parents=True)
  intrinsics = llvm_root / "llvm" / "include" / "llvm" / "IR" / "IntrinsicsDLC.td"
  intrinsics.parent.mkdir(parents=True)
  intrinsics.write_text("// fake\n", encoding="utf-8")

  report = discover_environment(llvm_root, archer_reference=None)

  assert report.ok
  assert report.tools["llc"].path.is_executable
  assert report.inputs["dlc_intrinsics_td"].is_file


def test_run_command_captures_stdout_and_exit_code():
  result = run_command(
    [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"]
  )

  assert result.exit_code == 3
  assert result.stdout.strip() == "out"
  assert result.stderr.strip() == "err"
  assert not result.timed_out


def test_run_command_reports_missing_executable():
  result = run_command(["/definitely/missing/dlc-testforge-command"])

  assert result.exit_code == 127
  assert result.stderr


def test_run_command_reports_timeout():
  result = run_command([sys.executable, "-c", "import time; time.sleep(2)"], timeout=1)

  assert result.exit_code == 124
  assert result.timed_out
