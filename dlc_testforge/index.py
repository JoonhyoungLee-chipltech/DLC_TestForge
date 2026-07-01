from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dlc_testforge.paths import discover_environment


RUN_RE = re.compile(r"^\s*[;#]\s*RUN:\s?(.*)$")
FUNCTION_RE = re.compile(r"\bdefine\b[^{@]*@(\"[^\"]+\"|[-A-Za-z0-9_.$]+)\s*\(")
INTRINSIC_RE = re.compile(r"\bllvm\.dlc\.[A-Za-z0-9_.]+")
CHECK_PREFIX_RE = re.compile(r"-{1,2}check-prefix(?:es)?(?:=|\s+)([A-Za-z0-9_,.$-]+)")
CHECK_LINE_RE = re.compile(r"^\s*[;#]\s*([A-Za-z0-9_.$-]+):\s*(.*)$")


@dataclass(frozen=True)
class SkippedTest:
  path: str
  reason: str

  def to_dict(self) -> dict[str, str]:
    return {"path": self.path, "reason": self.reason}


@dataclass(frozen=True)
class TestIndexEntry:
  path: str
  kind: str
  category: str
  run_lines: list[str]
  commands: list[str]
  check_prefixes: list[str]
  functions: list[str]
  intrinsics: list[str]
  features: list[str]
  opcodes_checked: list[str]
  uses_global_isel: bool
  uses_stop_after: bool
  uses_run_pass: bool
  uses_mcpu_hhp: bool
  uses_mattr: bool

  def to_dict(self) -> dict[str, Any]:
    return {
      "path": self.path,
      "kind": self.kind,
      "category": self.category,
      "run_lines": self.run_lines,
      "commands": self.commands,
      "check_prefixes": self.check_prefixes,
      "functions": self.functions,
      "intrinsics": self.intrinsics,
      "features": self.features,
      "opcodes_checked": self.opcodes_checked,
      "uses_global_isel": self.uses_global_isel,
      "uses_stop_after": self.uses_stop_after,
      "uses_run_pass": self.uses_run_pass,
      "uses_mcpu_hhp": self.uses_mcpu_hhp,
      "uses_mattr": self.uses_mattr,
    }


@dataclass(frozen=True)
class TestIndex:
  llvm_root: Path
  test_root: Path
  tests: list[TestIndexEntry]
  skipped: list[SkippedTest] = field(default_factory=list)
  total_files_seen: int = 0

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "llvm_root": str(self.llvm_root),
      "test_root": str(self.test_root),
      "tests": [entry.to_dict() for entry in self.tests],
      "summary": {
        "total_files_seen": self.total_files_seen,
        "indexed": len(self.tests),
        "skipped": [entry.to_dict() for entry in self.skipped],
        "categories": _count_by(self.tests, "category"),
        "kinds": _count_by(self.tests, "kind"),
        "features": _count_features(self.tests),
      },
    }


def build_index(llvm_root: Path) -> TestIndex:
  env = discover_environment(llvm_root, archer_reference=None, check_versions=False)
  if not env.llvm_root.ok:
    raise ValueError(f"LLVM root not found: {env.llvm_root.path}")

  test_root_status = env.inputs["dlc_codegen_tests"]
  if not test_root_status.ok:
    raise ValueError(f"DLC CodeGen test directory not found: {test_root_status.path}")

  test_root = test_root_status.path
  tests: list[TestIndexEntry] = []
  skipped: list[SkippedTest] = []
  files = sorted(
    [*test_root.rglob("*.ll"), *test_root.rglob("*.mir")],
    key=lambda path: path.relative_to(env.llvm_root.path).as_posix(),
  )

  for test_file in files:
    entry = parse_test_file(env.llvm_root.path, test_root, test_file)
    if entry is None:
      skipped.append(
        SkippedTest(
          path=test_file.relative_to(env.llvm_root.path).as_posix(),
          reason="missing RUN",
        )
      )
    else:
      tests.append(entry)

  return TestIndex(
    llvm_root=env.llvm_root.path,
    test_root=test_root,
    tests=tests,
    skipped=skipped,
    total_files_seen=len(files),
  )


def write_index(index: TestIndex, out_path: Path) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(
    json.dumps(index.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def parse_test_file(
  llvm_root: Path, test_root: Path, test_file: Path
) -> TestIndexEntry | None:
  text = test_file.read_text(encoding="utf-8", errors="replace")
  run_lines = _extract_run_lines(text)
  if not run_lines:
    return None

  commands = _join_run_commands(run_lines)
  relative_path = test_file.relative_to(llvm_root).as_posix()
  kind = test_file.suffix.removeprefix(".")
  category = _category_for(test_root, test_file)
  intrinsics = sorted(set(INTRINSIC_RE.findall(text)))
  check_prefixes = _extract_check_prefixes(commands)
  features = _extract_features(text, commands, kind, category, intrinsics)

  uses_global_isel = "global-isel" in features
  uses_stop_after = "stop-after" in features
  uses_run_pass = "run-pass" in features
  uses_mcpu_hhp = "mcpu-hhp" in features
  uses_mattr = "mattr" in features

  return TestIndexEntry(
    path=relative_path,
    kind=kind,
    category=category,
    run_lines=run_lines,
    commands=commands,
    check_prefixes=check_prefixes,
    functions=_extract_functions(text),
    intrinsics=intrinsics,
    features=features,
    opcodes_checked=_extract_opcodes_checked(text, check_prefixes),
    uses_global_isel=uses_global_isel,
    uses_stop_after=uses_stop_after,
    uses_run_pass=uses_run_pass,
    uses_mcpu_hhp=uses_mcpu_hhp,
    uses_mattr=uses_mattr,
  )


def _extract_run_lines(text: str) -> list[str]:
  return [
    match.group(1).strip()
    for line in text.splitlines()
    if (match := RUN_RE.match(line))
  ]


def _join_run_commands(run_lines: list[str]) -> list[str]:
  commands: list[str] = []
  current = ""
  for run_line in run_lines:
    line = run_line.rstrip()
    continued = line.endswith("\\")
    if continued:
      line = line[:-1].rstrip()
    current = f"{current} {line}".strip() if current else line
    if not continued:
      commands.append(_normalize_space(current))
      current = ""
  if current:
    commands.append(_normalize_space(current))
  return commands


def _extract_check_prefixes(commands: list[str]) -> list[str]:
  prefixes: set[str] = set()
  for command in commands:
    for match in CHECK_PREFIX_RE.finditer(command):
      for prefix in match.group(1).split(","):
        if prefix:
          prefixes.add(prefix)
    if "FileCheck" in command and not CHECK_PREFIX_RE.search(command):
      prefixes.add("CHECK")
  return sorted(prefixes)


def _extract_functions(text: str) -> list[str]:
  functions = []
  for match in FUNCTION_RE.finditer(text):
    functions.append(match.group(1).strip('"'))
  return sorted(set(functions))


def _extract_features(
  text: str,
  commands: list[str],
  kind: str,
  category: str,
  intrinsics: list[str],
) -> list[str]:
  command_text = "\n".join(commands)
  command_lower = command_text.lower()
  combined = f"{text}\n{command_text}".lower()
  features: set[str] = set()

  if "llc" in command_lower:
    features.add("llc")
  if "clang" in command_lower:
    features.add("clang")
  if "-global-isel" in combined or category == "GIISel":
    features.add("global-isel")
  if "-stop-after" in combined:
    features.add("stop-after")
    features.add("machine-pass")
  if "-run-pass" in combined:
    features.add("run-pass")
    features.add("machine-pass")
  if "-mcpu=hhp" in combined:
    features.add("mcpu-hhp")
  if "-mattr" in combined:
    features.add("mattr")
  if category == "vector" or re.search(r"<\s*\d+\s+x\s+", text):
    features.add("vector")
  if intrinsics:
    features.add("intrinsic")
  if kind == "mir":
    features.add("mir")
  return sorted(features)


def _extract_opcodes_checked(text: str, check_prefixes: list[str]) -> list[str]:
  prefixes = set(check_prefixes) or {"CHECK"}
  opcodes: set[str] = set()
  for line in text.splitlines():
    match = CHECK_LINE_RE.match(line)
    if not match:
      continue
    directive = match.group(1)
    if not any(directive == prefix or directive.startswith(f"{prefix}-") for prefix in prefixes):
      continue
    body = match.group(2).strip()
    if not body or body.startswith(("!", "[", "{", "}")):
      continue
    token = body.split()[0].rstrip(":,")
    if token and not token.startswith(("{{", "[[")):
      opcodes.add(token)
  return sorted(opcodes)


def _category_for(test_root: Path, test_file: Path) -> str:
  relative_parts = test_file.relative_to(test_root).parts
  if len(relative_parts) > 1 and relative_parts[0] in {"GIISel", "vector", "opt"}:
    return relative_parts[0]
  return "root"


def _count_by(entries: list[TestIndexEntry], field_name: str) -> dict[str, int]:
  counts: dict[str, int] = {}
  for entry in entries:
    value = getattr(entry, field_name)
    counts[value] = counts.get(value, 0) + 1
  return dict(sorted(counts.items()))


def _count_features(entries: list[TestIndexEntry]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for entry in entries:
    for feature in entry.features:
      counts[feature] = counts.get(feature, 0) + 1
  return dict(sorted(counts.items()))


def _normalize_space(text: str) -> str:
  return " ".join(text.split())
