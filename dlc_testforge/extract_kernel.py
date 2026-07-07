from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SOURCE_SUFFIXES = [".c", ".cpp", ".h", ".hpp"]
DERIVED_KERNEL_SUFFIXES = {".c", ".cpp"}
DTYPE_TOKENS = ["bf16", "f32", "i32", "i64", "int8", "long"]
MEMORY_SPACES = ["HBM", "VMEM", "CMEM", "SMEM"]
DMA_BOUNDARY_VALUES = {128, 256, 512, 1024}
ADDRESS_EXPONENT_VALUES = {6, 7, 8}
STRIDE_BOUNDARY_VALUES = {128, 256}
TILE_BOUNDARY_VALUES = {1024, 2048, 3072, 4096}
MASK_BOUNDARY_VALUES = {8, 32, 128, 255, 256, 1024}
VECTOR_LANE_VALUES = {8, 32, 128, 1024}
CALL_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
INTEGER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])")
VECTOR_TYPE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_128\b")
CONTROL_CALLS = {
  "if",
  "for",
  "while",
  "switch",
  "return",
  "sizeof",
  "dlc_dma",
  "dlc_sync",
}


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
    dma_call_count=sum(len(kernel.usage.dma_calls) for kernel in kernels),
    sync_call_count=sum(len(kernel.usage.sync_calls) for kernel in kernels),
    vector_usage_count=sum(
      len(kernel.usage.vector_types) + len(kernel.usage.intrinsics)
      for kernel in kernels
    ),
    edge_hint_count=sum(len(kernel.edge_hints) for kernel in kernels),
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
  usage = _extract_source_usage(source_path) if source_path is not None else KernelUsage()
  edge_hints = _generate_edge_hints(usage, source)
  return KernelRecord(
    name=name,
    source=source,
    category=_source_category(kernel_root, source_path),
    dtype_hints=_dtype_hints(name, source_stem),
    usage=usage,
    edge_hints=edge_hints,
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


def _extract_source_usage(source_path: Path) -> KernelUsage:
  text = source_path.read_text(encoding="utf-8", errors="ignore")
  stripped = _strip_comments_for_token_scan(text)
  intrinsics = _extract_intrinsics(stripped)
  usage = KernelUsage(
    dma_calls=_extract_dma_calls(stripped),
    sync_calls=_extract_sync_calls(stripped),
    memory_spaces=[
      space for space in MEMORY_SPACES if re.search(rf"\b{space}\b", stripped)
    ],
    vector_types=sorted(set(VECTOR_TYPE_RE.findall(stripped))),
    intrinsics=intrinsics,
    constants=sorted({int(match.group(0)) for match in INTEGER_RE.finditer(stripped)}),
  )
  return KernelUsage(
    dma_calls=usage.dma_calls,
    sync_calls=usage.sync_calls,
    memory_spaces=usage.memory_spaces,
    vector_types=usage.vector_types,
    intrinsics=usage.intrinsics,
    constants=usage.constants,
    relations=_generate_relation_hints(usage, stripped),
  )


def _generate_relation_hints(usage: KernelUsage, source_text: str) -> list[RelationHint]:
  relations: list[RelationHint] = []
  seen: set[tuple[str, int, str]] = set()

  for call in usage.dma_calls:
    addr_exp_values = _integer_values(call.addr_exp)
    if len(addr_exp_values) == 1 and addr_exp_values[0] in ADDRESS_EXPONENT_VALUES:
      _append_relation(
        relations,
        seen,
        "address_exponent",
        call.line,
        f"addr_exp={call.addr_exp}",
        "dlc_dma address exponent is near the common x128 scaling boundary",
      )

    for field_name, value in [
      ("length", call.length),
      ("unit_len", call.unit_len),
    ]:
      if _contains_any_integer(value, DMA_BOUNDARY_VALUES):
        _append_relation(
          relations,
          seen,
          "dma_length_boundary",
          call.line,
          f"{field_name}={value}",
          "DMA length or unit length references a hardware transfer boundary",
        )

    for field_name, value in [
      ("src_stride", call.src_stride),
      ("dst_stride", call.dst_stride),
    ]:
      if (
        _contains_any_integer(value, STRIDE_BOUNDARY_VALUES)
        or "stride" in value.lower()
      ):
        _append_relation(
          relations,
          seen,
          "stride_boundary",
          call.line,
          f"{field_name}={value}",
          "DMA stride expression may exercise strided transfer boundaries",
        )

    src_space = _normalize_memory_space(call.src_space)
    dst_space = _normalize_memory_space(call.dst_space)
    if src_space is not None and dst_space is not None:
      _append_relation(
        relations,
        seen,
        "memory_space_pair",
        call.line,
        f"{src_space}->{dst_space}",
        "DMA transfer crosses DLC memory spaces",
      )

  for value in usage.constants:
    if value in TILE_BOUNDARY_VALUES:
      _append_relation(
        relations,
        seen,
        "tile_boundary",
        _line_for_text_match(source_text, rf"\b{value}\b"),
        f"tile_constant={value}",
        "source references a common tile-size boundary",
      )

  for intrinsic in usage.intrinsics:
    if _is_mask_relation_intrinsic(intrinsic):
      _append_relation(
        relations,
        seen,
        "mask_boundary",
        _line_for_text_match(source_text, rf"\b{re.escape(intrinsic)}\b"),
        f"intrinsic={intrinsic}",
        "source uses mask or tail-mask related operations",
      )

  if re.search(r"\bget_device_id\b", source_text):
    _append_relation(
      relations,
      seen,
      "dual_xys_split",
      _line_for_text_match(source_text, r"\bget_device_id\b"),
      "get_device_id",
      "source branches or partitions work by DLC device/core id",
    )

  return relations


def _append_relation(
  relations: list[RelationHint],
  seen: set[tuple[str, int, str]],
  kind: str,
  line: int,
  evidence: str,
  reason: str,
) -> None:
  key = (kind, line, evidence)
  if key in seen:
    return
  seen.add(key)
  relations.append(
    RelationHint(
      kind=kind,
      line=line,
      evidence=evidence,
      reason=reason,
    )
  )


def _normalize_memory_space(value: str) -> str | None:
  normalized = value.strip()
  if normalized.startswith("D_"):
    normalized = normalized[2:]
  return normalized if normalized in MEMORY_SPACES else None


def _line_for_text_match(text: str, pattern: str) -> int:
  match = re.search(pattern, text)
  if match is None:
    return 0
  return text.count("\n", 0, match.start()) + 1


def _contains_any_integer(text: str, values: set[int]) -> bool:
  return any(value in values for value in _integer_values(text))


def _integer_values(text: str) -> list[int]:
  return [int(match.group(0)) for match in INTEGER_RE.finditer(text)]


def _is_mask_relation_intrinsic(name: str) -> bool:
  lowered = name.lower()
  return (
    name == "pre_exp2"
    or "mask" in lowered
    or "msk" in lowered
    or "ldmask" in lowered
    or "stmask" in lowered
  )


def _generate_edge_hints(usage: KernelUsage, source: str | None) -> list[EdgeHint]:
  edge_hints: list[EdgeHint] = []
  seen: set[tuple[str, int, str]] = set()

  for relation in usage.relations:
    if relation.kind == "address_exponent":
      base = _first_integer(relation.evidence)
      if base is not None:
        _append_edge_hint(
          edge_hints,
          seen,
          "addr_exp_boundary",
          base,
          _source_at_line(source, relation.line),
          "exercise address exponent boundary",
        )
    elif relation.kind == "dma_length_boundary":
      base = _first_integer(relation.evidence)
      if base is not None:
        _append_edge_hint(
          edge_hints,
          seen,
          "dma_length_boundary",
          base,
          _source_at_line(source, relation.line),
          "exercise DMA length or unit length boundary",
        )
    elif relation.kind == "stride_boundary":
      base = _first_integer_in(relation.evidence, STRIDE_BOUNDARY_VALUES)
      if base is not None:
        _append_edge_hint(
          edge_hints,
          seen,
          "stride_boundary",
          base,
          _source_at_line(source, relation.line),
          "exercise DMA stride boundary",
        )
    elif relation.kind == "tile_boundary":
      base = _first_integer(relation.evidence)
      if base is not None:
        _append_edge_hint(
          edge_hints,
          seen,
          "tile_boundary",
          base,
          _source_at_line(source, relation.line),
          "exercise tile-size boundary",
        )

  if any(relation.kind == "mask_boundary" for relation in usage.relations):
    for base in usage.constants:
      if base in MASK_BOUNDARY_VALUES:
        _append_edge_hint(
          edge_hints,
          seen,
          "mask_boundary",
          base,
          _source_at_line(source, _line_for_relation_kind(usage, "mask_boundary")),
          "exercise mask or tail-mask boundary",
        )

  if usage.vector_types:
    for base in _vector_lane_bases(usage):
      _append_edge_hint(
        edge_hints,
        seen,
        "vector_lane_boundary",
        base,
        _source_at_line(source, 0),
        "exercise vector lane boundary",
      )

  return edge_hints


def _append_edge_hint(
  edge_hints: list[EdgeHint],
  seen: set[tuple[str, int, str]],
  kind: str,
  base: int,
  source: str,
  reason: str,
) -> None:
  key = (kind, base, source)
  if key in seen:
    return
  seen.add(key)
  edge_hints.append(
    EdgeHint(
      kind=kind,
      base=base,
      values=_edge_values(base),
      source=source,
      reason=reason,
    )
  )


def _edge_values(base: int) -> list[int]:
  if base > 1:
    return [base - 1, base, base + 1]
  if base == 1:
    return [0, 1, 2]
  return [base]


def _source_at_line(source: str | None, line: int) -> str:
  if source is None or line <= 0:
    return "unknown"
  return f"{source}:{line}"


def _first_integer(text: str) -> int | None:
  values = _integer_values(text)
  return values[0] if values else None


def _first_integer_in(text: str, allowed: set[int]) -> int | None:
  for value in _integer_values(text):
    if value in allowed:
      return value
  return None


def _line_for_relation_kind(usage: KernelUsage, kind: str) -> int:
  for relation in usage.relations:
    if relation.kind == kind:
      return relation.line
  return 0


def _vector_lane_bases(usage: KernelUsage) -> list[int]:
  bases = set()
  for value in usage.constants:
    if value in VECTOR_LANE_VALUES:
      bases.add(value)
  for vector_type in usage.vector_types:
    for value in _vector_type_integer_values(vector_type):
      if value in VECTOR_LANE_VALUES:
        bases.add(value)
  return sorted(bases)


def _vector_type_integer_values(vector_type: str) -> list[int]:
  return [int(value) for value in re.findall(r"\d+", vector_type)]


def _extract_dma_calls(text: str) -> list[DmaCall]:
  calls = []
  for line, args_text in _find_balanced_calls(text, "dlc_dma"):
    args = _split_call_args(args_text)
    if len(args) != 9:
      continue
    calls.append(
      DmaCall(
        line=line,
        src_addr=args[0],
        src_space=args[1],
        dst_addr=args[2],
        dst_space=args[3],
        length=args[4],
        src_stride=args[5],
        dst_stride=args[6],
        unit_len=args[7],
        addr_exp=args[8],
      )
    )
  return calls


def _extract_sync_calls(text: str) -> list[SyncCall]:
  calls = []
  for line, args_text in _find_balanced_calls(text, "dlc_sync"):
    args = _split_call_args(args_text)
    if not args:
      continue
    calls.append(SyncCall(line=line, handle=args[0]))
  return calls


def _extract_intrinsics(text: str) -> list[str]:
  names = set()
  for match in CALL_TOKEN_RE.finditer(text):
    name = match.group(1)
    if name in CONTROL_CALLS:
      continue
    if _is_intrinsic_name(name):
      names.add(name)
  return sorted(names)


def _is_intrinsic_name(name: str) -> bool:
  return (
    name.startswith("v_")
    or name.startswith("__dlc_")
    or name.startswith("m_")
    or name.startswith("gstf")
    or name.startswith("gsnf")
    or name == "pre_exp2"
    or "ldmask" in name
    or "msk" in name
    or "mask" in name
    or (name.startswith("load") and "mask" in name)
  )


def _strip_comments_for_token_scan(text: str) -> str:
  chars: list[str] = []
  index = 0
  while index < len(text):
    if text.startswith("//", index):
      chars.extend("  ")
      index += 2
      while index < len(text) and text[index] != "\n":
        chars.append(" ")
        index += 1
      continue
    if text.startswith("/*", index):
      chars.extend("  ")
      index += 2
      while index < len(text) and not text.startswith("*/", index):
        chars.append("\n" if text[index] == "\n" else " ")
        index += 1
      if index < len(text):
        chars.extend("  ")
        index += 2
      continue
    chars.append(text[index])
    index += 1
  return "".join(chars)


def _find_balanced_calls(text: str, name: str) -> list[tuple[int, str]]:
  pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(")
  calls = []
  for match in pattern.finditer(text):
    open_index = match.end() - 1
    depth = 0
    for index in range(open_index, len(text)):
      char = text[index]
      if char == "(":
        depth += 1
      elif char == ")":
        depth -= 1
        if depth == 0:
          line = text.count("\n", 0, match.start()) + 1
          calls.append((line, text[open_index + 1 : index].strip()))
          break
  return calls


def _split_call_args(args: str) -> list[str]:
  if not args.strip():
    return []
  result = []
  start = 0
  paren_depth = 0
  bracket_depth = 0
  brace_depth = 0
  for index, char in enumerate(args):
    if char == "(":
      paren_depth += 1
    elif char == ")":
      paren_depth = max(paren_depth - 1, 0)
    elif char == "[":
      bracket_depth += 1
    elif char == "]":
      bracket_depth = max(bracket_depth - 1, 0)
    elif char == "{":
      brace_depth += 1
    elif char == "}":
      brace_depth = max(brace_depth - 1, 0)
    elif (
      char == ","
      and paren_depth == 0
      and bracket_depth == 0
      and brace_depth == 0
    ):
      result.append(args[start:index].strip())
      start = index + 1
  result.append(args[start:].strip())
  return result
