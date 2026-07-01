from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dlc_testforge.paths import discover_environment


FIXED_SOURCE_FILES = [
  "llvm/include/llvm/IR/IntrinsicsDLC.td",
  "clang/include/clang/Basic/BuiltinsDLC.def",
  "clang/include/clang/Basic/BuiltinsDLCHHP.def",
  "clang/lib/CodeGen/CGBuiltinDLC.cpp",
]

DEF_HEADER_RE = re.compile(
  r"(?m)^\s*(def|defm|class|multiclass)\s+([A-Za-z_][A-Za-z0-9_#]*)"
)
INTRINSIC_DEF_RE = re.compile(
  r"(?ms)^\s*def\s+(int_dlc_[A-Za-z0-9_]+)\s*(?P<body>.*?);"
)
BUILTIN_RE = re.compile(
  r"^\s*(BUILTIN|TARGET_BUILTIN)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
  r'"([^"]*)"\s*,\s*"([^"]*)"(?:\s*,\s*([^)]+?))?\s*\)'
)
CLANG_BUILTIN_RE = re.compile(r'ClangBuiltin<"([^"]+)">')
IMM_ARG_RE = re.compile(r"ImmArg\s*<\s*ArgIndex\s*<\s*(\d+)\s*>\s*>")
OPERAND_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
IMMEDIATE_TOKEN_RE = re.compile(
  r"\b(?:[if]\d+imm|untyped_imm_\d+|ImmLeaf|AsmImmRange\s*<\s*[^>]+>)"
)
PREDICATES_RE = re.compile(r"let\s+Predicates\s*=\s*\[([^\]]*)\]")
REFERENCE_TOPICS = [
  "DAG",
  "SelectionDAG",
  "GI",
  "Legalizer",
  "RegBank",
  "InstructionSelector",
  "INITIALIZE_PASS",
  "DEBUG_TYPE",
  "BuiltinID",
  "IntrinsicID",
]


@dataclass(frozen=True)
class SourceRecord:
  path: str
  kind: str
  record_count: int

  def to_dict(self) -> dict[str, Any]:
    return {
      "path": self.path,
      "kind": self.kind,
      "record_count": self.record_count,
    }


@dataclass(frozen=True)
class IntrinsicRecord:
  name: str
  llvm_name: str
  source: str
  return_types: list[str]
  operand_types: list[str]
  attributes: list[str]
  clang_builtin: str | None
  imm_args: list[int]

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "llvm_name": self.llvm_name,
      "source": self.source,
      "return_types": self.return_types,
      "operand_types": self.operand_types,
      "attributes": self.attributes,
      "clang_builtin": self.clang_builtin,
      "imm_args": self.imm_args,
    }


@dataclass(frozen=True)
class BuiltinRecord:
  name: str
  source: str
  macro: str
  type_signature: str
  attributes: str
  feature: str | None
  nearby_comment: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "source": self.source,
      "macro": self.macro,
      "type_signature": self.type_signature,
      "attributes": self.attributes,
      "feature": self.feature,
      "nearby_comment": self.nearby_comment,
    }


@dataclass(frozen=True)
class InstructionRecord:
  name: str
  source: str
  kind: str
  base: str | None
  operand_names: list[str]
  immediate_operands: list[str]
  predicates: list[str]
  is_gisel: bool

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "source": self.source,
      "kind": self.kind,
      "base": self.base,
      "operand_names": self.operand_names,
      "immediate_operands": self.immediate_operands,
      "predicates": self.predicates,
      "is_gisel": self.is_gisel,
    }


@dataclass(frozen=True)
class ReferenceRecord:
  topic: str
  name: str
  source: str
  line: int
  text: str

  def to_dict(self) -> dict[str, Any]:
    return {
      "topic": self.topic,
      "name": self.name,
      "source": self.source,
      "line": self.line,
      "text": self.text,
    }


@dataclass(frozen=True)
class TdIndex:
  llvm_root: Path
  sources: list[SourceRecord]
  intrinsics: list[IntrinsicRecord]
  builtins: list[BuiltinRecord]
  instructions: list[InstructionRecord]
  references: list[ReferenceRecord]
  missing_sources: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "llvm_root": str(self.llvm_root),
      "sources": [source.to_dict() for source in self.sources],
      "intrinsics": [record.to_dict() for record in self.intrinsics],
      "builtins": [record.to_dict() for record in self.builtins],
      "instructions": [record.to_dict() for record in self.instructions],
      "references": [record.to_dict() for record in self.references],
      "summary": {
        "source_count": len(self.sources),
        "intrinsic_count": len(self.intrinsics),
        "builtin_count": len(self.builtins),
        "instruction_count": len(self.instructions),
        "reference_count": len(self.references),
        "missing_sources": self.missing_sources,
        "sources": {source.path: source.record_count for source in self.sources},
      },
    }


def build_td_index(llvm_root: Path) -> TdIndex:
  env = discover_environment(llvm_root, archer_reference=None, check_versions=False)
  if not env.llvm_root.ok:
    raise ValueError(f"LLVM root not found: {env.llvm_root.path}")

  target_status = env.inputs["dlc_target_src"]
  if not target_status.ok:
    raise ValueError(f"DLC target source directory not found: {target_status.path}")

  source_paths, missing_sources = _discover_sources(env.llvm_root.path)
  sources: list[SourceRecord] = []
  intrinsics: list[IntrinsicRecord] = []
  builtins: list[BuiltinRecord] = []
  instructions: list[InstructionRecord] = []
  references: list[ReferenceRecord] = []

  for source_path in source_paths:
    source = source_path.relative_to(env.llvm_root.path).as_posix()
    text = source_path.read_text(encoding="utf-8", errors="replace")
    source_intrinsics = _parse_intrinsics(source, text)
    source_builtins = _parse_builtins(source, text)
    source_instructions = _parse_instructions(source, text)
    source_references = _parse_references(source, text)
    record_count = (
      len(source_intrinsics)
      + len(source_builtins)
      + len(source_instructions)
      + len(source_references)
    )
    sources.append(
      SourceRecord(path=source, kind=_source_kind(source_path), record_count=record_count)
    )
    intrinsics.extend(source_intrinsics)
    builtins.extend(source_builtins)
    instructions.extend(source_instructions)
    references.extend(source_references)

  return TdIndex(
    llvm_root=env.llvm_root.path,
    sources=sorted(sources, key=lambda item: item.path),
    intrinsics=sorted(intrinsics, key=lambda item: (item.source, item.name)),
    builtins=sorted(builtins, key=lambda item: (item.source, item.name)),
    instructions=sorted(instructions, key=lambda item: (item.source, item.name, item.kind)),
    references=sorted(references, key=lambda item: (item.source, item.line, item.topic)),
    missing_sources=sorted(missing_sources),
  )


def write_td_index(index: TdIndex, out_path: Path) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(
    json.dumps(index.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _discover_sources(llvm_root: Path) -> tuple[list[Path], list[str]]:
  source_paths: set[Path] = set()
  missing_sources: list[str] = []

  for relative in FIXED_SOURCE_FILES:
    path = llvm_root / relative
    if path.is_file():
      source_paths.add(path)
    else:
      missing_sources.append(relative)

  target_root = llvm_root / "llvm" / "lib" / "Target" / "DLC"
  td_files = sorted(target_root.glob("*.td"))
  if td_files:
    source_paths.update(td_files)
  else:
    missing_sources.append("llvm/lib/Target/DLC/*.td")

  source_paths.update(
    path
    for path in sorted(target_root.iterdir())
    if path.is_file() and path.suffix in {".cpp", ".h"}
  )

  gisel_root = target_root / "GISel"
  if gisel_root.is_dir():
    source_paths.update(path for path in sorted(gisel_root.iterdir()) if path.is_file())
  else:
    missing_sources.append("llvm/lib/Target/DLC/GISel/*")

  return sorted(source_paths, key=lambda path: path.relative_to(llvm_root).as_posix()), missing_sources


def _parse_intrinsics(source: str, text: str) -> list[IntrinsicRecord]:
  if not source.endswith("IntrinsicsDLC.td"):
    return []
  records: list[IntrinsicRecord] = []
  for match in INTRINSIC_DEF_RE.finditer(_strip_tablegen_comments(text)):
    name = match.group(1)
    block = match.group(0)
    groups = _extract_intrinsic_groups(block)
    clang_builtin = _first_match(CLANG_BUILTIN_RE, block)
    records.append(
      IntrinsicRecord(
        name=name,
        llvm_name=_llvm_intrinsic_name(name),
        source=source,
        return_types=_split_list(groups[0]) if len(groups) > 0 else [],
        operand_types=_split_list(groups[1]) if len(groups) > 1 else [],
        attributes=_split_list(groups[2]) if len(groups) > 2 else [],
        clang_builtin=clang_builtin,
        imm_args=sorted({int(value) for value in IMM_ARG_RE.findall(block)}),
      )
    )
  return records


def _parse_builtins(source: str, text: str) -> list[BuiltinRecord]:
  if not source.endswith(".def"):
    return []
  records: list[BuiltinRecord] = []
  comment_buffer: list[str] = []
  for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith("//"):
      comment_buffer.append(stripped.removeprefix("//").strip())
      continue
    if not stripped:
      comment_buffer = []
      continue
    match = BUILTIN_RE.match(line)
    if match:
      records.append(
        BuiltinRecord(
          name=match.group(2),
          source=source,
          macro=match.group(1),
          type_signature=match.group(3),
          attributes=match.group(4),
          feature=_clean_feature(match.group(5)),
          nearby_comment=_snippet(" ".join(comment_buffer)),
        )
      )
    comment_buffer = []
  return records


def _parse_instructions(source: str, text: str) -> list[InstructionRecord]:
  if not source.endswith(".td") or source.endswith("IntrinsicsDLC.td"):
    return []
  clean_text = _strip_tablegen_comments(text)
  records: list[InstructionRecord] = []
  for kind, name, block in _iter_tablegen_blocks(clean_text):
    base = _extract_base(block)
    records.append(
      InstructionRecord(
        name=name,
        source=source,
        kind=kind,
        base=base,
        operand_names=sorted(set(OPERAND_RE.findall(block))),
        immediate_operands=sorted(set(_normalize_space(value) for value in IMMEDIATE_TOKEN_RE.findall(block))),
        predicates=_extract_predicates(block),
        is_gisel=_is_gisel_instruction(source, name, block),
      )
    )
  return records


def _parse_references(source: str, text: str) -> list[ReferenceRecord]:
  records: list[ReferenceRecord] = []
  for line_number, line in enumerate(text.splitlines(), start=1):
    stripped = _normalize_space(line.strip())
    if not stripped or stripped.startswith("//"):
      continue
    for topic in REFERENCE_TOPICS:
      if not _line_has_topic(topic, stripped):
        continue
      records.append(
        ReferenceRecord(
          topic=topic,
          name=_reference_name(topic, stripped),
          source=source,
          line=line_number,
          text=_snippet(stripped),
        )
      )
  return records


def _iter_tablegen_blocks(text: str) -> list[tuple[str, str, str]]:
  blocks: list[tuple[str, str, str]] = []
  for match in DEF_HEADER_RE.finditer(text):
    if _is_commented_line(text, match.start()):
      continue
    end = _tablegen_block_end(text, match.end())
    if end <= match.start():
      continue
    blocks.append((match.group(1), match.group(2), text[match.start() : end]))
  return blocks


def _tablegen_block_end(text: str, start: int) -> int:
  brace_index = text.find("{", start)
  semicolon_index = text.find(";", start)
  if semicolon_index < 0 and brace_index < 0:
    return len(text)
  if brace_index < 0 or (0 <= semicolon_index < brace_index):
    return semicolon_index + 1

  depth = 0
  for index in range(brace_index, len(text)):
    char = text[index]
    if char == "{":
      depth += 1
    elif char == "}":
      depth -= 1
      if depth == 0:
        semicolon_after = text.find(";", index + 1)
        next_line_after = text.find("\n", index + 1)
        if semicolon_after >= 0 and (
          next_line_after < 0 or semicolon_after < next_line_after
        ):
          return semicolon_after + 1
        return index + 1
  return len(text)


def _extract_intrinsic_groups(block: str) -> list[str]:
  intrinsic_index = block.find("Intrinsic<")
  if intrinsic_index < 0:
    return []
  open_index = block.find("<", intrinsic_index)
  close_index = _matching_angle(block, open_index)
  if open_index < 0 or close_index < 0:
    return []
  return _split_top_level_commas(block[open_index + 1 : close_index])


def _matching_angle(text: str, open_index: int) -> int:
  if open_index < 0:
    return -1
  depth = 0
  for index in range(open_index, len(text)):
    char = text[index]
    if char == "<":
      depth += 1
    elif char == ">":
      depth -= 1
      if depth == 0:
        return index
  return -1


def _split_top_level_commas(text: str) -> list[str]:
  parts: list[str] = []
  current: list[str] = []
  angle_depth = 0
  square_depth = 0
  paren_depth = 0
  for char in text:
    if char == "<":
      angle_depth += 1
    elif char == ">":
      angle_depth = max(0, angle_depth - 1)
    elif char == "[":
      square_depth += 1
    elif char == "]":
      square_depth = max(0, square_depth - 1)
    elif char == "(":
      paren_depth += 1
    elif char == ")":
      paren_depth = max(0, paren_depth - 1)
    if char == "," and angle_depth == 0 and square_depth == 0 and paren_depth == 0:
      parts.append(_normalize_space("".join(current)))
      current = []
      continue
    current.append(char)
  if current:
    parts.append(_normalize_space("".join(current)))
  return parts


def _split_list(text: str) -> list[str]:
  stripped = text.strip()
  if stripped.startswith("[") and stripped.endswith("]"):
    stripped = stripped[1:-1]
  if not stripped:
    return []
  return [_normalize_space(part) for part in _split_top_level_commas(stripped) if part.strip()]


def _extract_base(block: str) -> str | None:
  header = block.split("{", 1)[0]
  if ":" not in header:
    return None
  after_colon = header.split(":", 1)[1].strip()
  match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", after_colon)
  return match.group(1) if match else None


def _extract_predicates(block: str) -> list[str]:
  predicates: set[str] = set()
  for match in PREDICATES_RE.finditer(block):
    predicates.update(_split_list(match.group(1)))
  for token in re.findall(r"\bHas[A-Za-z0-9_]+|\bRequires<[A-Za-z0-9_,\s]+>", block):
    predicates.add(_normalize_space(token))
  return sorted(predicates)


def _is_gisel_instruction(source: str, name: str, block: str) -> bool:
  return (
    "/GISel/" in source
    or source.endswith("DLCInstrInfoGISel.td")
    or name.startswith("G_")
    or "GIComplex" in block
    or "GenericInstruction" in block
  )


def _reference_name(topic: str, text: str) -> str:
  if topic == "DEBUG_TYPE":
    match = re.search(r'"([^"]+)"', text)
    return match.group(1) if match else topic
  if topic == "INITIALIZE_PASS":
    match = re.search(r"INITIALIZE_PASS(?:_BEGIN|_END)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    return match.group(1) if match else topic
  if topic == "BuiltinID":
    match = re.search(r"DLC::(BI[A-Za-z0-9_]+)", text)
    return match.group(1) if match else topic
  if topic == "IntrinsicID":
    matches = re.findall(r"Intrinsic::([A-Za-z_][A-Za-z0-9_]*)", text)
    for name in matches:
      if name != "ID":
        return name
    return matches[0] if matches else topic
  if topic == "RegBank":
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*(?:RegBank|RegisterBank)[A-Za-z0-9_]*)\b", text)
    return match.group(1) if match else topic
  return _identifier_containing(topic, text) or topic


def _line_has_topic(topic: str, text: str) -> bool:
  if topic == "RegBank":
    return "RegBank" in text or "RegisterBank" in text
  if topic == "BuiltinID":
    return "DLC::BI" in text
  if topic == "IntrinsicID":
    return "Intrinsic::" in text
  return topic in text


def _identifier_containing(topic: str, text: str) -> str | None:
  for identifier in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text):
    if topic in identifier:
      return identifier
  return None


def _source_kind(path: Path) -> str:
  if path.name.endswith(".td"):
    return "tablegen"
  if path.name.endswith(".def"):
    return "builtin_def"
  if path.suffix in {".cpp", ".h"}:
    return "source"
  return "other"


def _llvm_intrinsic_name(name: str) -> str:
  return "llvm." + name.removeprefix("int_").replace("_", ".")


def _clean_feature(feature: str | None) -> str | None:
  if feature is None:
    return None
  cleaned = feature.strip()
  return cleaned or None


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
  match = pattern.search(text)
  return match.group(1) if match else None


def _strip_block_comments(text: str) -> str:
  return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _strip_tablegen_comments(text: str) -> str:
  without_blocks = _strip_block_comments(text)
  return re.sub(r"//.*", "", without_blocks)


def _is_commented_line(text: str, index: int) -> bool:
  line_start = text.rfind("\n", 0, index) + 1
  return text[line_start:index].lstrip().startswith("//")


def _snippet(text: str, limit: int = 240) -> str:
  normalized = _normalize_space(text)
  if len(normalized) <= limit:
    return normalized
  return normalized[: limit - 3].rstrip() + "..."


def _normalize_space(text: str) -> str:
  return " ".join(text.split())
