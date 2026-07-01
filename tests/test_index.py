from __future__ import annotations

import json

from dlc_testforge.cli import main
from dlc_testforge.index import build_index


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _make_llvm_root(tmp_path):
  llvm_root = tmp_path / "LLVM"
  (llvm_root / "llvm" / "test" / "CodeGen" / "DLC").mkdir(parents=True)
  return llvm_root


def test_indexes_single_line_ll_run(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "add.ll",
    """; RUN: llc -mtriple=dlc < %s | FileCheck %s
define i32 @add(i32 %a, i32 %b) {
  %r = add i32 %a, %b
  ret i32 %r
}
; CHECK-LABEL: add
; CHECK: ret
""",
  )

  index = build_index(llvm_root)

  assert len(index.tests) == 1
  entry = index.tests[0]
  assert entry.path == "llvm/test/CodeGen/DLC/add.ll"
  assert entry.kind == "ll"
  assert entry.category == "root"
  assert entry.commands == ["llc -mtriple=dlc < %s | FileCheck %s"]
  assert entry.check_prefixes == ["CHECK"]
  assert entry.functions == ["add"]
  assert "llc" in entry.features
  assert "add" in entry.opcodes_checked
  assert "ret" in entry.opcodes_checked


def test_joins_continued_run_lines_and_extracts_prefixes(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "continued.ll",
    """; RUN: llc -mtriple=dlc < %s \\
; RUN:   | FileCheck --check-prefixes=DLC,EXTRA %s
define void @continued() {
  ret void
}
; DLC: halt
; EXTRA: ret
""",
  )

  entry = build_index(llvm_root).tests[0]

  assert entry.commands == [
    "llc -mtriple=dlc < %s | FileCheck --check-prefixes=DLC,EXTRA %s"
  ]
  assert entry.check_prefixes == ["DLC", "EXTRA"]
  assert "halt" in entry.opcodes_checked
  assert "ret" in entry.opcodes_checked


def test_indexes_mir_run_and_mir_features(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "llvm" / "test" / "CodeGen" / "DLC" / "GIISel" / "legalizer.mir",
    """# RUN: llc -mtriple=dlc -global-isel -run-pass=legalizer %s -o - | FileCheck -check-prefix=GISEL %s
---
name: legalizer
...
# GISEL: G_ADD
""",
  )

  entry = build_index(llvm_root).tests[0]

  assert entry.kind == "mir"
  assert entry.category == "GIISel"
  assert entry.check_prefixes == ["GISEL"]
  assert entry.uses_global_isel
  assert entry.uses_run_pass
  assert "machine-pass" in entry.features
  assert "mir" in entry.features


def test_detects_categories_features_intrinsics_and_functions(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  root = llvm_root / "llvm" / "test" / "CodeGen" / "DLC"
  _write(
    root / "vector" / "mask.ll",
    """; RUN: llc -mtriple=dlc -mattr=+core1 < %s | FileCheck -check-prefix=VEC %s
define <1024 x i32> @mask(<1024 x i32> %a) {
  %r = call <1024 x i32> @llvm.dlc.shrar(<1024 x i32> %a, <1024 x i32> %a)
  ret <1024 x i32> %r
}
declare <1024 x i32> @llvm.dlc.shrar(<1024 x i32>, <1024 x i32>)
; VEC: shrar
""",
  )
  _write(
    root / "opt" / "pass.ll",
    """; RUN: llc -mtriple=dlc -stop-after=dlc-machine-addropt %s -o - | FileCheck %s
define void @pass() {
  ret void
}
; CHECK: body
""",
  )
  _write(
    root / "hhp.ll",
    """; RUN: llc -mtriple=dlc -mcpu=hhp < %s | FileCheck %s
define void @hhp() {
  ret void
}
; CHECK: ret
""",
  )

  entries = {entry.path: entry for entry in build_index(llvm_root).tests}

  vector = entries["llvm/test/CodeGen/DLC/vector/mask.ll"]
  assert vector.category == "vector"
  assert vector.functions == ["mask"]
  assert vector.intrinsics == ["llvm.dlc.shrar"]
  assert "intrinsic" in vector.features
  assert "vector" in vector.features
  assert vector.uses_mattr

  opt = entries["llvm/test/CodeGen/DLC/opt/pass.ll"]
  assert opt.category == "opt"
  assert opt.uses_stop_after
  assert "machine-pass" in opt.features

  hhp = entries["llvm/test/CodeGen/DLC/hhp.ll"]
  assert hhp.uses_mcpu_hhp


def test_skips_runless_tests_and_records_summary(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  root = llvm_root / "llvm" / "test" / "CodeGen" / "DLC"
  _write(root / "valid.ll", "; RUN: llc < %s | FileCheck %s\n; CHECK: ret\n")
  _write(root / "opt" / "runless.ll", "define void @runless() { ret void }\n")

  index = build_index(llvm_root).to_dict()

  assert index["summary"]["indexed"] == 1
  assert index["summary"]["total_files_seen"] == 2
  assert index["summary"]["skipped"] == [
    {
      "path": "llvm/test/CodeGen/DLC/opt/runless.ll",
      "reason": "missing RUN",
    }
  ]


def test_cli_writes_deterministic_index_json(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  root = llvm_root / "llvm" / "test" / "CodeGen" / "DLC"
  _write(root / "b.ll", "; RUN: llc < %s | FileCheck %s\n; CHECK: b\n")
  _write(root / "a.ll", "; RUN: llc < %s | FileCheck %s\n; CHECK: a\n")
  out_one = tmp_path / "out" / "one.json"
  out_two = tmp_path / "out" / "two.json"

  assert main(["index", "--llvm-root", str(llvm_root), "--out", str(out_one)]) == 0
  assert main(["index", "--llvm-root", str(llvm_root), "--out", str(out_two)]) == 0

  one = json.loads(out_one.read_text(encoding="utf-8"))
  two = json.loads(out_two.read_text(encoding="utf-8"))
  assert one == two
  assert [entry["path"] for entry in one["tests"]] == [
    "llvm/test/CodeGen/DLC/a.ll",
    "llvm/test/CodeGen/DLC/b.ll",
  ]
