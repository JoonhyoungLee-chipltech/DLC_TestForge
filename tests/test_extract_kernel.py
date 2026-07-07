from __future__ import annotations

from dlc_testforge.extract_kernel import (
  DmaCall,
  EdgeHint,
  KernelRecord,
  KernelUsage,
  KernelUsageIndex,
  KernelUsageSummary,
  RelationHint,
  SyncCall,
)


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
