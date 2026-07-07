from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DmaCall:
  line: int
  src_addr: str
  src_space: str
  dst_addr: str
  dst_space: str
  length: str
  src_stride: str
  dst_stride: str
  unit_len: str
  addr_exp: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "line": self.line,
      "src_addr": self.src_addr,
      "src_space": self.src_space,
      "dst_addr": self.dst_addr,
      "dst_space": self.dst_space,
      "length": self.length,
      "src_stride": self.src_stride,
      "dst_stride": self.dst_stride,
      "unit_len": self.unit_len,
      "addr_exp": self.addr_exp,
    }


@dataclass(frozen=True)
class SyncCall:
  line: int
  handle: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "line": self.line,
      "handle": self.handle,
    }


@dataclass(frozen=True)
class RelationHint:
  kind: str
  line: int
  evidence: str
  reason: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "line": self.line,
      "evidence": self.evidence,
      "reason": self.reason,
    }


@dataclass(frozen=True)
class EdgeHint:
  kind: str
  base: int
  values: list[int]
  source: str
  reason: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "base": self.base,
      "values": self.values,
      "source": self.source,
      "reason": self.reason,
    }


@dataclass(frozen=True)
class KernelUsage:
  dma_calls: list[DmaCall] = field(default_factory=list)
  sync_calls: list[SyncCall] = field(default_factory=list)
  memory_spaces: list[str] = field(default_factory=list)
  vector_types: list[str] = field(default_factory=list)
  intrinsics: list[str] = field(default_factory=list)
  constants: list[int] = field(default_factory=list)
  relations: list[RelationHint] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "dma_calls": [call.to_dict() for call in self.dma_calls],
      "sync_calls": [call.to_dict() for call in self.sync_calls],
      "memory_spaces": self.memory_spaces,
      "vector_types": self.vector_types,
      "intrinsics": self.intrinsics,
      "constants": self.constants,
      "relations": [relation.to_dict() for relation in self.relations],
    }


@dataclass(frozen=True)
class KernelRecord:
  name: str
  source: str | None
  category: str
  dtype_hints: list[str] = field(default_factory=list)
  features: list[str] = field(default_factory=list)
  usage: KernelUsage = field(default_factory=KernelUsage)
  edge_hints: list[EdgeHint] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "source": self.source,
      "category": self.category,
      "dtype_hints": self.dtype_hints,
      "features": self.features,
      "usage": self.usage.to_dict(),
      "edge_hints": [hint.to_dict() for hint in self.edge_hints],
    }


@dataclass(frozen=True)
class KernelUsageSummary:
  kernel_count: int = 0
  source_file_count: int = 0
  syntest_file_count: int = 0
  dma_call_count: int = 0
  sync_call_count: int = 0
  vector_usage_count: int = 0
  edge_hint_count: int = 0

  def to_dict(self) -> dict[str, int]:
    return {
      "kernel_count": self.kernel_count,
      "source_file_count": self.source_file_count,
      "syntest_file_count": self.syntest_file_count,
      "dma_call_count": self.dma_call_count,
      "sync_call_count": self.sync_call_count,
      "vector_usage_count": self.vector_usage_count,
      "edge_hint_count": self.edge_hint_count,
    }


@dataclass(frozen=True)
class KernelUsageIndex:
  root: Path
  summary: KernelUsageSummary = field(default_factory=KernelUsageSummary)
  kernels: list[KernelRecord] = field(default_factory=list)
  global_edge_hints: list[EdgeHint] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "root": str(self.root),
      "summary": self.summary.to_dict(),
      "kernels": [kernel.to_dict() for kernel in self.kernels],
      "global_edge_hints": [hint.to_dict() for hint in self.global_edge_hints],
    }
