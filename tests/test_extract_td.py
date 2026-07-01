from __future__ import annotations

import json

from dlc_testforge.cli import main
from dlc_testforge.extract_td import build_td_index


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _make_llvm_root(tmp_path):
  llvm_root = tmp_path / "LLVM"
  (llvm_root / "llvm" / "lib" / "Target" / "DLC" / "GISel").mkdir(parents=True)
  return llvm_root


def test_extracts_multiline_intrinsic_fields(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "llvm" / "include" / "llvm" / "IR" / "IntrinsicsDLC.td",
    """let TargetPrefix = "dlc" in {
def int_dlc_sync_gte: ClangBuiltin<"dlc_sync_gte">,
  Intrinsic<[],
            [llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<1>>, IntrHasSideEffects]>;
}
""",
  )

  index = build_td_index(llvm_root)
  intrinsic = index.intrinsics[0]

  assert intrinsic.name == "int_dlc_sync_gte"
  assert intrinsic.llvm_name == "llvm.dlc.sync.gte"
  assert intrinsic.return_types == []
  assert intrinsic.operand_types == ["llvm_i32_ty", "llvm_i32_ty"]
  assert intrinsic.attributes == ["ImmArg<ArgIndex<1>>", "IntrHasSideEffects"]
  assert intrinsic.clang_builtin == "dlc_sync_gte"
  assert intrinsic.imm_args == [1]
  assert not index.instructions


def test_extracts_builtin_macros_and_nearby_comments(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "clang" / "include" / "clang" / "Basic" / "BuiltinsDLC.def",
    """// vector load with stride and mask
// parameters are [offset, base, stride, mask]
BUILTIN(v_f32_ld_tnsr_st_msk, "E1024fiv*ii", "n")
TARGET_BUILTIN(v_core_only, "vi", "n", "+core1")
""",
  )

  records = build_td_index(llvm_root).builtins
  by_name = {record.name: record for record in records}

  assert by_name["v_f32_ld_tnsr_st_msk"].macro == "BUILTIN"
  assert by_name["v_f32_ld_tnsr_st_msk"].type_signature == "E1024fiv*ii"
  assert "stride and mask" in by_name["v_f32_ld_tnsr_st_msk"].nearby_comment
  assert by_name["v_core_only"].macro == "TARGET_BUILTIN"
  assert by_name["v_core_only"].feature == '"+core1"'


def test_extracts_tablegen_defs_operands_immediates_and_gisel_flag(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "llvm" / "lib" / "Target" / "DLC" / "DLCInstrInfoVector.td",
    """def V_SUBCORERT : VectorInst<0b0000100, (outs VX:$v_dest),
                  (ins VX:$v_x, i32imm:$v_y),
                  "$v_dest = rotate $v_x, $v_y", []> {
  let Predicates = [HasCore1];
}

defm ADD : VectorAluOpRegOrImm<0b0010100, "add.s32", add>;
class AsmImmRange<int Low, int High> : AsmOperandClass;
""",
  )
  _write(
    llvm_root / "llvm" / "lib" / "Target" / "DLC" / "DLCInstrInfoGISel.td",
    """def G_DMA_LOCAL : DLCGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$s0x, untyped_imm_0:$flag);
}
""",
  )

  records = build_td_index(llvm_root).instructions
  by_name = {record.name: record for record in records}

  assert by_name["V_SUBCORERT"].kind == "def"
  assert by_name["V_SUBCORERT"].base == "VectorInst"
  assert by_name["V_SUBCORERT"].operand_names == ["v_dest", "v_x", "v_y"]
  assert "i32imm" in by_name["V_SUBCORERT"].immediate_operands
  assert by_name["V_SUBCORERT"].predicates == ["HasCore1"]
  assert not by_name["V_SUBCORERT"].is_gisel

  assert by_name["ADD"].kind == "defm"
  assert by_name["AsmImmRange"].kind == "class"
  assert by_name["G_DMA_LOCAL"].is_gisel
  assert "untyped_imm_0" in by_name["G_DMA_LOCAL"].immediate_operands


def test_extracts_source_references(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "llvm" / "lib" / "Target" / "DLC" / "DLCMachineAddrOpt.cpp",
    """#define DEBUG_TYPE "dlc-machine-addropt"
class DLCMachineAddrOpt : public MachineFunctionPass {};
INITIALIZE_PASS(DLCMachineAddrOpt, "dlc-machine-addropt",
                "DLC Machine Address Optimizations", false, false)
""",
  )
  _write(
    llvm_root / "llvm" / "lib" / "Target" / "DLC" / "DLCCombine.td",
    """def fold_insert_subvector : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_INSERT_VECTOR):$root),
  (apply [{ return true; }])>;
""",
  )
  _write(
    llvm_root / "llvm" / "lib" / "Target" / "DLC" / "GISel" / "DLCLegalizerInfo.cpp",
    """#define DEBUG_TYPE "dlc-legalizer"
void buildLegalizer() {
  DLCRegisterBankInfo RBI;
  InstructionSelector *Selector = nullptr;
}
INITIALIZE_PASS(DLCLegalizerPass, "dlc-legalizer", "DLC legalizer", false, false)
""",
  )

  references = build_td_index(llvm_root).references
  topics = {record.topic for record in references}
  names = {record.name for record in references}

  assert {"DEBUG_TYPE", "Legalizer", "RegBank", "InstructionSelector", "INITIALIZE_PASS"} <= topics
  assert "dlc-legalizer" in names
  assert "DLCLegalizerPass" in names
  assert "dlc-machine-addropt" in names
  assert "DLCMachineAddrOpt" in names
  assert "GICombineRule" in names


def test_extracts_clang_builtin_lowering_references(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "clang" / "lib" / "CodeGen" / "CGBuiltinDLC.cpp",
    """Value *EmitBuiltin(unsigned BuiltinID) {
  if (BuiltinID == DLC::BIv_f32_ld_tnsr_b)
    Intrinsic::ID ID = Intrinsic::dlc_ld_tnsr_st_msk;
  if (BuiltinID == DLC::BIv_f32_st_tnsr_b)
    ID = Intrinsic::dlc_ld_tnsr_st_msk;
  return Builder.CreateCall(F, Ops);
}
""",
  )

  references = build_td_index(llvm_root).references
  topics = {record.topic for record in references}
  names = {record.name for record in references}

  assert "BuiltinID" in topics
  assert "IntrinsicID" in topics
  assert "BIv_f32_ld_tnsr_b" in names
  assert "dlc_ld_tnsr_st_msk" in names


def test_missing_optional_sources_are_reported(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)

  index = build_td_index(llvm_root).to_dict()

  assert "llvm/include/llvm/IR/IntrinsicsDLC.td" in index["summary"]["missing_sources"]
  assert "clang/include/clang/Basic/BuiltinsDLC.def" in index["summary"]["missing_sources"]
  assert "llvm/lib/Target/DLC/*.td" in index["summary"]["missing_sources"]


def test_cli_extract_td_writes_deterministic_json(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  _write(
    llvm_root / "llvm" / "include" / "llvm" / "IR" / "IntrinsicsDLC.td",
    'def int_dlc_nop : ClangBuiltin<"dlc_nop">, Intrinsic<[], [llvm_i32_ty], []>;\n',
  )
  out_one = tmp_path / "out" / "one.json"
  out_two = tmp_path / "out" / "two.json"

  assert main(["extract-td", "--llvm-root", str(llvm_root), "--out", str(out_one)]) == 0
  assert main(["extract-td", "--llvm-root", str(llvm_root), "--out", str(out_two)]) == 0

  one = json.loads(out_one.read_text(encoding="utf-8"))
  two = json.loads(out_two.read_text(encoding="utf-8"))
  assert one == two
  assert one["intrinsics"][0]["name"] == "int_dlc_nop"
