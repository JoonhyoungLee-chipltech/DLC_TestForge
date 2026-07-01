from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dlc_testforge.paths import discover_environment


SPEC_MARKDOWN_FILES = [
  "instruction-format.md",
  "individual-instructions.md",
  "assembly-code-standard.md",
  "abbreviation-dictionary.md",
  "dma-desc.md",
]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

FACT_KEYWORDS = {
  "immediates": ["imm", "immediate"],
  "operands": ["operand", "s*_x", "s*_y", "v*_x", "v*_y", "dest"],
  "reserved_values": ["reserved"],
  "fatal_conditions": ["fatal", "error", "program error", "out-bound"],
  "slot_constraints": ["slot", "s0", "s1", "both slots"],
  "dma_fields": [
    "dma",
    "descriptor",
    "sync flag",
    "src_",
    "dst_",
    "mem_id",
    "core_id",
  ],
}


@dataclass(frozen=True)
class SpecFacts:
  immediates: list[str] = field(default_factory=list)
  operands: list[str] = field(default_factory=list)
  reserved_values: list[str] = field(default_factory=list)
  fatal_conditions: list[str] = field(default_factory=list)
  slot_constraints: list[str] = field(default_factory=list)
  dma_fields: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, list[str]]:
    return {
      "immediates": self.immediates,
      "operands": self.operands,
      "reserved_values": self.reserved_values,
      "fatal_conditions": self.fatal_conditions,
      "slot_constraints": self.slot_constraints,
      "dma_fields": self.dma_fields,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "SpecFacts":
    return cls(
      immediates=list(data.get("immediates", [])),
      operands=list(data.get("operands", [])),
      reserved_values=list(data.get("reserved_values", [])),
      fatal_conditions=list(data.get("fatal_conditions", [])),
      slot_constraints=list(data.get("slot_constraints", [])),
      dma_fields=list(data.get("dma_fields", [])),
    )


@dataclass(frozen=True)
class SpecRecord:
  source: str
  heading_path: list[str]
  topic: str
  kind: str
  text: str
  table: dict[str, list[str]] | None
  facts: SpecFacts

  def to_dict(self) -> dict[str, Any]:
    return {
      "source": self.source,
      "heading_path": self.heading_path,
      "topic": self.topic,
      "kind": self.kind,
      "text": self.text,
      "table": self.table,
      "facts": self.facts.to_dict(),
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "SpecRecord":
    return cls(
      source=data["source"],
      heading_path=list(data["heading_path"]),
      topic=data["topic"],
      kind=data["kind"],
      text=data["text"],
      table=data.get("table"),
      facts=SpecFacts.from_dict(data["facts"]),
    )


@dataclass(frozen=True)
class SpecIndex:
  llvm_root: Path
  spec_root: Path
  records: list[SpecRecord]
  missing_sources: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "llvm_root": str(self.llvm_root),
      "spec_root": str(self.spec_root),
      "records": [record.to_dict() for record in self.records],
      "summary": {
        "sources": _count_sources(self.records),
        "record_count": len(self.records),
        "missing_sources": self.missing_sources,
      },
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "SpecIndex":
    return cls(
      llvm_root=Path(data["llvm_root"]),
      spec_root=Path(data["spec_root"]),
      records=[SpecRecord.from_dict(record) for record in data["records"]],
      missing_sources=list(data["summary"].get("missing_sources", [])),
    )


def build_spec_index(llvm_root: Path) -> SpecIndex:
  env = discover_environment(llvm_root, archer_reference=None, check_versions=False)
  if not env.llvm_root.ok:
    raise ValueError(f"LLVM root not found: {env.llvm_root.path}")

  spec_root_status = env.inputs["dlc_spec_dir"]
  if not spec_root_status.ok:
    raise ValueError(f"DLC spec directory not found: {spec_root_status.path}")

  records: list[SpecRecord] = []
  missing_sources: list[str] = []
  for filename in SPEC_MARKDOWN_FILES:
    source_path = spec_root_status.path / filename
    source = source_path.relative_to(env.llvm_root.path).as_posix()
    if not source_path.is_file():
      missing_sources.append(source)
      continue
    records.extend(_parse_markdown_file(env.llvm_root.path, source_path))

  return SpecIndex(
    llvm_root=env.llvm_root.path,
    spec_root=spec_root_status.path,
    records=records,
    missing_sources=missing_sources,
  )


def write_spec_index(index: SpecIndex, out_path: Path) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(
    json.dumps(index.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def load_spec_index(path: Path) -> SpecIndex:
  return SpecIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def lookup_dlc_spec(index: SpecIndex, topic: str) -> list[SpecRecord]:
  needle = topic.casefold()
  if not needle:
    return []
  return [
    record
    for record in index.records
    if needle in _record_search_text(record).casefold()
  ]


def _parse_markdown_file(llvm_root: Path, source_path: Path) -> list[SpecRecord]:
  source = source_path.relative_to(llvm_root).as_posix()
  lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
  records: list[SpecRecord] = []
  heading_path: list[str] = []
  text_buffer: list[str] = []
  code_buffer: list[str] = []
  in_code = False

  def flush_text() -> None:
    nonlocal text_buffer
    text = _normalize_space(" ".join(text_buffer))
    text_buffer = []
    if text:
      records.append(_make_record(source, heading_path, "text_block", text, None))

  def flush_code() -> None:
    nonlocal code_buffer
    text = "\n".join(code_buffer).strip()
    code_buffer = []
    if text:
      records.append(_make_record(source, heading_path, "code_block", text, None))

  index = 0
  while index < len(lines):
    line = lines[index]
    stripped = line.strip()

    if stripped.startswith("```"):
      if in_code:
        flush_code()
        in_code = False
      else:
        flush_text()
        in_code = True
      index += 1
      continue

    if in_code:
      code_buffer.append(line)
      index += 1
      continue

    heading = HEADING_RE.match(line)
    if heading:
      flush_text()
      depth = len(heading.group(1))
      title = _normalize_space(heading.group(2))
      heading_path = heading_path[: depth - 1] + [title]
      index += 1
      continue

    if _looks_like_table_start(lines, index):
      flush_text()
      table_lines = []
      while index < len(lines) and _is_table_line(lines[index]):
        table_lines.append(lines[index])
        index += 1
      records.extend(_table_records(source, heading_path, table_lines))
      continue

    if not stripped:
      flush_text()
      index += 1
      continue

    text_buffer.append(stripped)
    index += 1

  if in_code:
    flush_code()
  flush_text()
  return records


def _table_records(
  source: str, heading_path: list[str], table_lines: list[str]
) -> list[SpecRecord]:
  if len(table_lines) < 2 or not TABLE_SEPARATOR_RE.match(table_lines[1]):
    return []
  headers = _split_table_row(table_lines[0])
  records: list[SpecRecord] = []
  for line in table_lines[2:]:
    row = _split_table_row(line)
    if not any(cell.strip() for cell in row):
      continue
    padded = row + [""] * max(0, len(headers) - len(row))
    row = padded[: len(headers)] if headers else row
    text = _normalize_space(" | ".join(row))
    table = {"headers": headers, "row": row}
    records.append(_make_record(source, heading_path, "table_row", text, table))
  return records


def _make_record(
  source: str,
  heading_path: list[str],
  kind: str,
  text: str,
  table: dict[str, list[str]] | None,
) -> SpecRecord:
  topic = _topic_for(heading_path, table)
  return SpecRecord(
    source=source,
    heading_path=list(heading_path),
    topic=topic,
    kind=kind,
    text=text,
    table=table,
    facts=_extract_facts(text, table),
  )


def _extract_facts(
  text: str, table: dict[str, list[str]] | None = None
) -> SpecFacts:
  search_text = text
  if table is not None:
    search_text = " ".join([text, *table.get("row", [])])
  folded = search_text.casefold()
  facts: dict[str, list[str]] = {}
  for field_name, keywords in FACT_KEYWORDS.items():
    facts[field_name] = (
      [_snippet(search_text)]
      if any(keyword.casefold() in folded for keyword in keywords)
      else []
    )
  return SpecFacts(**facts)


def _record_search_text(record: SpecRecord) -> str:
  fact_values = []
  for values in record.facts.to_dict().values():
    fact_values.extend(values)
  table_values = []
  if record.table is not None:
    table_values.extend(record.table.get("headers", []))
    table_values.extend(record.table.get("row", []))
  return " ".join(
    [
      record.source,
      *record.heading_path,
      record.topic,
      record.kind,
      record.text,
      *table_values,
      *fact_values,
    ]
  )


def _topic_for(
  heading_path: list[str], table: dict[str, list[str]] | None = None
) -> str:
  if table is not None:
    for cell in table.get("row", []):
      if cell.strip():
        return _normalize_space(cell)
  return heading_path[-1] if heading_path else ""


def _looks_like_table_start(lines: list[str], index: int) -> bool:
  return (
    index + 1 < len(lines)
    and _is_table_line(lines[index])
    and TABLE_SEPARATOR_RE.match(lines[index + 1]) is not None
  )


def _is_table_line(line: str) -> bool:
  stripped = line.strip()
  return stripped.startswith("|") and stripped.endswith("|")


def _split_table_row(line: str) -> list[str]:
  stripped = line.strip()
  if stripped.startswith("|"):
    stripped = stripped[1:]
  if stripped.endswith("|"):
    stripped = stripped[:-1]
  return [_normalize_space(cell) for cell in stripped.split("|")]


def _count_sources(records: list[SpecRecord]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for record in records:
    counts[record.source] = counts.get(record.source, 0) + 1
  return dict(sorted(counts.items()))


def _snippet(text: str, limit: int = 240) -> str:
  normalized = _normalize_space(text)
  if len(normalized) <= limit:
    return normalized
  return normalized[: limit - 3].rstrip() + "..."


def _normalize_space(text: str) -> str:
  return " ".join(text.split())
