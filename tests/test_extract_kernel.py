from __future__ import annotations

import json

from dlc_testforge.extract_kernel import (
  DmaCall,
  EdgeHint,
  KernelRecord,
  KernelUsage,
  KernelUsageIndex,
  KernelUsageSummary,
  RelationHint,
  SyncCall,
  build_kernel_usage_index,
  write_kernel_usage_index,
)


def _write(path, text=""):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def test_empty_kernel_usage_index_shape(tmp_path):
  index = KernelUsageIndex(root=tmp_path / "DLC_Custom_Kernel")

  assert index.to_dict() == {
    "schema_version": 1,
    "root": str(tmp_path / "DLC_Custom_Kernel"),
    "summary": {
      "kernel_count": 0,
      "source_file_count": 0,
      "syntest_file_count": 0,
      "dma_call_count": 0,
      "sync_call_count": 0,
      "vector_usage_count": 0,
      "edge_hint_count": 0,
    },
    "kernels": [],
    "global_edge_hints": [],
  }


def test_call_and_hint_records_serialize_expected_fields():
  dma = DmaCall(
    line=42,
    src_addr="input->address + offset",
    src_space="HBM",
    dst_addr="vmem_addr",
    dst_space="VMEM",
    length="tile_len",
    src_stride="0",
    dst_stride="0",
    unit_len="tile_len",
    addr_exp="7",
  )
  sync = SyncCall(line=43, handle="dma_handle")
  relation = RelationHint(
    kind="address_exponent",
    line=42,
    evidence="addr_exp=7",
    reason="dlc_dma commonly uses x128 address scaling",
  )
  edge = EdgeHint(
    kind="addr_exp_boundary",
    base=7,
    values=[6, 7, 8],
    source="dlc_kernels/foo.c:42",
    reason="exercise address exponent boundary",
  )

  assert dma.to_dict() == {
    "line": 42,
    "src_addr": "input->address + offset",
    "src_space": "HBM",
    "dst_addr": "vmem_addr",
    "dst_space": "VMEM",
    "length": "tile_len",
    "src_stride": "0",
    "dst_stride": "0",
    "unit_len": "tile_len",
    "addr_exp": "7",
  }
  assert sync.to_dict() == {"line": 43, "handle": "dma_handle"}
  assert relation.to_dict() == {
    "kind": "address_exponent",
    "line": 42,
    "evidence": "addr_exp=7",
    "reason": "dlc_dma commonly uses x128 address scaling",
  }
  assert edge.to_dict() == {
    "kind": "addr_exp_boundary",
    "base": 7,
    "values": [6, 7, 8],
    "source": "dlc_kernels/foo.c:42",
    "reason": "exercise address exponent boundary",
  }


def test_kernel_record_serializes_nested_usage_and_edge_hints():
  relation = RelationHint(
    kind="memory_space_pair",
    line=12,
    evidence="HBM->VMEM",
    reason="DMA transfer from host memory to vector memory",
  )
  edge = EdgeHint(
    kind="dma_length_boundary",
    base=128,
    values=[127, 128, 129],
    source="dlc_kernels/FusedRMSNorm.c:12",
    reason="exercise 128 element DMA boundary",
  )
  usage = KernelUsage(
    dma_calls=[
      DmaCall(
        line=12,
        src_addr="src",
        src_space="HBM",
        dst_addr="dst",
        dst_space="VMEM",
        length="128",
        src_stride="0",
        dst_stride="0",
        unit_len="128",
        addr_exp="7",
      )
    ],
    sync_calls=[SyncCall(line=13, handle="handle")],
    memory_spaces=["HBM", "VMEM"],
    vector_types=["float8_128"],
    intrinsics=["v_f32_ld_tnsr_st_msk"],
    constants=[7, 128, 1024],
    relations=[relation],
  )
  record = KernelRecord(
    name="custom_FusedRMSNorm_f32",
    source="dlc_kernels/FusedRMSNorm.c",
    category="root",
    dtype_hints=["f32"],
    features=["dma", "hbm", "vmem", "vector", "mask"],
    usage=usage,
    edge_hints=[edge],
  )

  assert record.to_dict() == {
    "name": "custom_FusedRMSNorm_f32",
    "source": "dlc_kernels/FusedRMSNorm.c",
    "category": "root",
    "dtype_hints": ["f32"],
    "features": ["dma", "hbm", "vmem", "vector", "mask"],
    "usage": {
      "dma_calls": [
        {
          "line": 12,
          "src_addr": "src",
          "src_space": "HBM",
          "dst_addr": "dst",
          "dst_space": "VMEM",
          "length": "128",
          "src_stride": "0",
          "dst_stride": "0",
          "unit_len": "128",
          "addr_exp": "7",
        }
      ],
      "sync_calls": [{"line": 13, "handle": "handle"}],
      "memory_spaces": ["HBM", "VMEM"],
      "vector_types": ["float8_128"],
      "intrinsics": ["v_f32_ld_tnsr_st_msk"],
      "constants": [7, 128, 1024],
      "relations": [relation.to_dict()],
    },
    "edge_hints": [edge.to_dict()],
  }


def test_kernel_usage_summary_contains_all_count_fields():
  summary = KernelUsageSummary(
    kernel_count=2,
    source_file_count=3,
    syntest_file_count=4,
    dma_call_count=5,
    sync_call_count=6,
    vector_usage_count=7,
    edge_hint_count=8,
  )

  assert summary.to_dict() == {
    "kernel_count": 2,
    "source_file_count": 3,
    "syntest_file_count": 4,
    "dma_call_count": 5,
    "sync_call_count": 6,
    "vector_usage_count": 7,
    "edge_hint_count": 8,
  }


def test_build_kernel_usage_index_resolves_yaml_direct_source(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_src" / "kernel_info.yaml",
    "- name: custom_top_f32\n  src: top_kernel\n",
  )
  _write(kernel_root / "dlc_kernels" / "top_kernel.c")

  index = build_kernel_usage_index(kernel_root)
  records = {record.name: record for record in index.kernels}

  assert index.summary.kernel_count == 1
  assert index.summary.source_file_count == 1
  assert records["custom_top_f32"].source == "dlc_kernels/top_kernel.c"
  assert records["custom_top_f32"].category == "root"
  assert records["custom_top_f32"].dtype_hints == ["f32"]


def test_build_kernel_usage_index_resolves_yaml_recursive_source(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_src" / "kernel_info.yaml",
    "- name: Custom_layer_norm_bf16\n  src: layer_norm_bf16\n",
  )
  _write(kernel_root / "dlc_kernels" / "norm" / "layer_norm" / "layer_norm_bf16.cpp")

  index = build_kernel_usage_index(kernel_root)
  record = index.kernels[0]

  assert record.name == "Custom_layer_norm_bf16"
  assert record.source == "dlc_kernels/norm/layer_norm/layer_norm_bf16.cpp"
  assert record.category == "norm"
  assert record.dtype_hints == ["bf16"]


def test_build_kernel_usage_index_derives_records_when_yaml_is_missing(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(kernel_root / "dlc_kernels" / "alpha_i32.c")
  _write(kernel_root / "dlc_kernels" / "norm" / "beta_long.cpp")

  index = build_kernel_usage_index(kernel_root)
  records = {record.name: record for record in index.kernels}

  assert index.summary.kernel_count == 2
  assert index.summary.source_file_count == 2
  assert records["alpha_i32"].source == "dlc_kernels/alpha_i32.c"
  assert records["alpha_i32"].category == "root"
  assert records["alpha_i32"].dtype_hints == ["i32"]
  assert records["beta_long"].source == "dlc_kernels/norm/beta_long.cpp"
  assert records["beta_long"].category == "norm"
  assert records["beta_long"].dtype_hints == ["long"]


def test_unreferenced_headers_are_counted_but_not_kernel_records(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(kernel_root / "dlc_kernels" / "helper.hpp")
  _write(kernel_root / "dlc_kernels" / "gamma_int8.c")

  index = build_kernel_usage_index(kernel_root)

  assert index.summary.source_file_count == 2
  assert [record.name for record in index.kernels] == ["gamma_int8"]
  assert index.kernels[0].dtype_hints == ["int8"]


def test_missing_yaml_source_keeps_kernel_record(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_src" / "kernel_info.yaml",
    "- name: custom_missing_i64\n  src: missing_source\n",
  )
  (kernel_root / "dlc_kernels").mkdir(parents=True)

  index = build_kernel_usage_index(kernel_root)
  record = index.kernels[0]

  assert record.name == "custom_missing_i64"
  assert record.source is None
  assert record.category == "unknown"
  assert record.dtype_hints == ["i64"]


def test_referenced_header_can_be_kernel_record(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_src" / "kernel_info.yaml",
    "- name: custom_header_bf16\n  src: header_kernel\n",
  )
  _write(kernel_root / "dlc_kernels" / "headers" / "header_kernel.h")

  index = build_kernel_usage_index(kernel_root)
  record = index.kernels[0]

  assert record.name == "custom_header_bf16"
  assert record.source == "dlc_kernels/headers/header_kernel.h"
  assert record.category == "headers"
  assert record.dtype_hints == ["bf16"]


def test_write_kernel_usage_index_writes_schema_json(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(kernel_root / "dlc_kernels" / "alpha.c")
  index = build_kernel_usage_index(kernel_root)
  out = tmp_path / "out" / "kernel-index.json"

  write_kernel_usage_index(index, out)

  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["schema_version"] == 1
  assert data["summary"]["kernel_count"] == 1
  assert data["kernels"][0]["name"] == "alpha"


def test_extracts_multiline_dma_sync_and_source_tokens(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_kernels" / "dma_kernel.c",
    """inline void copy_kernel(tensor hbm_src, tensor vmem_dst, int src_addr, int dst_addr, int length) {
  int handle = dlc_dma(tensor_slice(hbm_src, src_addr / 32), HBM,
                       tensor_slice(vmem_dst, dst_addr / 32), VMEM,
                       length, 128, 128, 128, 7);
  dlc_sync(handle);
  int mask = pre_exp2(length / 128);
  float8_128 x = v_f32_ld_tnsr_st_msk(dst_addr / 32, vmem_dst, 1, mask);
  int8_128 y = __dlc_abs(x);
  bool8_128 z = m_f32_perm(x, y, 0, 0);
}
""",
  )

  index = build_kernel_usage_index(kernel_root)
  record = index.kernels[0]
  usage = record.usage

  assert index.summary.dma_call_count == 1
  assert index.summary.sync_call_count == 1
  assert index.summary.vector_usage_count == (
    len(usage.vector_types) + len(usage.intrinsics)
  )
  assert usage.dma_calls[0].to_dict() == {
    "line": 2,
    "src_addr": "tensor_slice(hbm_src, src_addr / 32)",
    "src_space": "HBM",
    "dst_addr": "tensor_slice(vmem_dst, dst_addr / 32)",
    "dst_space": "VMEM",
    "length": "length",
    "src_stride": "128",
    "dst_stride": "128",
    "unit_len": "128",
    "addr_exp": "7",
  }
  assert usage.sync_calls[0].to_dict() == {"line": 5, "handle": "handle"}
  assert usage.memory_spaces == ["HBM", "VMEM"]
  assert usage.vector_types == ["bool8_128", "float8_128", "int8_128"]
  assert usage.intrinsics == [
    "__dlc_abs",
    "m_f32_perm",
    "pre_exp2",
    "v_f32_ld_tnsr_st_msk",
  ]
  assert usage.constants == [0, 1, 7, 32, 128]


def test_extracts_multiple_dma_calls_and_skips_invalid_arity(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_kernels" / "multi_dma.c",
    """void f(tensor h, tensor v) {
  int a = dlc_dma(h, HBM, v, VMEM, 128, 128, 128, 128, 7);
  int bad = dlc_dma(h, HBM);
  int b = dlc_dma(v, VMEM, h, HBM, 256, 128, 128, 128, 7);
  dlc_sync(a);
  dlc_sync(b);
}
""",
  )

  index = build_kernel_usage_index(kernel_root)
  usage = index.kernels[0].usage

  assert index.summary.dma_call_count == 2
  assert index.summary.sync_call_count == 2
  assert [call.length for call in usage.dma_calls] == ["128", "256"]
  assert [call.handle for call in usage.sync_calls] == ["a", "b"]


def test_source_usage_ignores_comment_only_tokens(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_kernels" / "comments.c",
    """void f(tensor h, tensor v) {
  // dlc_dma(h, HBM, v, VMEM, 1024, 128, 128, 128, 7);
  /* dlc_sync(fake);
     float8_128 hidden = v_f32_ld_tnsr_st_msk(0, v, 1, 1);
     CMEM SMEM -9
  */
  int real_value = -3;
}
""",
  )

  index = build_kernel_usage_index(kernel_root)
  usage = index.kernels[0].usage

  assert usage.dma_calls == []
  assert usage.sync_calls == []
  assert usage.memory_spaces == []
  assert usage.vector_types == []
  assert usage.intrinsics == []
  assert usage.constants == [-3]


def test_missing_yaml_source_keeps_empty_usage(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_src" / "kernel_info.yaml",
    "- name: custom_missing\n  src: missing\n",
  )
  (kernel_root / "dlc_kernels").mkdir(parents=True)

  index = build_kernel_usage_index(kernel_root)
  usage = index.kernels[0].usage

  assert usage.dma_calls == []
  assert usage.sync_calls == []
  assert usage.memory_spaces == []
  assert usage.vector_types == []
  assert usage.intrinsics == []
  assert usage.constants == []
