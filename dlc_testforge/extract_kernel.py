from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SOURCE_SUFFIXES = [".c", ".cpp", ".h", ".hpp"]
DERIVED_KERNEL_SUFFIXES = {".c", ".cpp"}
DTYPE_TOKENS = ["bf16", "f32", "i32", "i64", "int8", "long"]


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


def build_kernel_usage_index(kernel_root: Path) -> KernelUsageIndex:
  kernel_root = kernel_root.expanduser().resolve(strict=False)
  if not kernel_root.is_dir():
    raise ValueError(f"kernel root not found: {kernel_root}")

  source_root = kernel_root / "dlc_kernels"
  if not source_root.is_dir():
    raise ValueError(f"dlc_kernels directory not found: {source_root}")

  source_files = _discover_source_files(source_root)
  source_by_stem = _source_files_by_stem(source_files)
  kernels: list[KernelRecord] = []
  referenced_sources: set[Path] = set()

  yaml_entries = _load_yaml_entries(kernel_root / "dlc_src" / "kernel_info.yaml")
  for entry in yaml_entries:
    name = entry["name"]
    source_path = _resolve_yaml_source(source_root, source_by_stem, entry.get("src"))
    if source_path is not None:
      referenced_sources.add(source_path)
    kernels.append(_kernel_record(name, kernel_root, source_path))

  for source_path in source_files:
    if source_path in referenced_sources:
      continue
    if source_path.suffix not in DERIVED_KERNEL_SUFFIXES:
      continue
    kernels.append(_kernel_record(source_path.stem, kernel_root, source_path))

  kernels.sort(key=lambda kernel: (kernel.name.lower(), kernel.source or ""))
  summary = KernelUsageSummary(
    kernel_count=len(kernels),
    source_file_count=len(source_files),
    syntest_file_count=0,
  )
  return KernelUsageIndex(
    root=kernel_root,
    summary=summary,
    kernels=kernels,
  )


def write_kernel_usage_index(index: KernelUsageIndex, out: Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(
    json.dumps(index.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _discover_source_files(source_root: Path) -> list[Path]:
  files = [
    path
    for path in source_root.rglob("*")
    if path.is_file() and path.suffix in SUPPORTED_SOURCE_SUFFIXES
  ]
  return sorted(files, key=lambda path: path.relative_to(source_root).as_posix())


def _source_files_by_stem(source_files: list[Path]) -> dict[str, list[Path]]:
  by_stem: dict[str, list[Path]] = {}
  for path in source_files:
    by_stem.setdefault(path.stem, []).append(path)
  for matches in by_stem.values():
    matches.sort(key=lambda path: path.as_posix())
  return by_stem


def _load_yaml_entries(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    return []
  data = yaml.safe_load(path.read_text(encoding="utf-8"))
  if data is None:
    return []
  if not isinstance(data, list):
    raise ValueError(f"kernel_info.yaml must contain a list: {path}")

  entries: list[dict[str, Any]] = []
  for index, item in enumerate(data):
    if not isinstance(item, dict):
      raise ValueError(f"kernel_info.yaml entry {index} must be a mapping")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f"kernel_info.yaml entry {index} must have a non-empty name")
    entry = dict(item)
    entry["name"] = name.strip()
    entries.append(entry)
  return entries


def _resolve_yaml_source(
  source_root: Path,
  source_by_stem: dict[str, list[Path]],
  src: Any,
) -> Path | None:
  if not isinstance(src, str) or not src.strip():
    return None
  src_name = src.strip()
  src_path = Path(src_name)
  if src_path.suffix in SUPPORTED_SOURCE_SUFFIXES:
    direct = source_root / src_path
    if direct.is_file():
      return direct
    matches = source_by_stem.get(src_path.stem, [])
    return matches[0] if matches else None

  for suffix in SUPPORTED_SOURCE_SUFFIXES:
    direct = source_root / f"{src_name}{suffix}"
    if direct.is_file():
      return direct

  matches = source_by_stem.get(src_name, [])
  return matches[0] if matches else None


def _kernel_record(name: str, kernel_root: Path, source_path: Path | None) -> KernelRecord:
  source = _source_relative_to_root(kernel_root, source_path) if source_path else None
  source_stem = source_path.stem if source_path is not None else ""
  return KernelRecord(
    name=name,
    source=source,
    category=_source_category(kernel_root, source_path),
    dtype_hints=_dtype_hints(name, source_stem),
  )


def _source_relative_to_root(kernel_root: Path, source_path: Path) -> str:
  return source_path.relative_to(kernel_root).as_posix()


def _source_category(kernel_root: Path, source_path: Path | None) -> str:
  if source_path is None:
    return "unknown"
  relative_parts = source_path.relative_to(kernel_root / "dlc_kernels").parts
  if len(relative_parts) <= 1:
    return "root"
  return relative_parts[0]


def _dtype_hints(name: str, source_stem: str) -> list[str]:
  haystack = f"{name} {source_stem}".lower()
  return [token for token in DTYPE_TOKENS if token in haystack]
