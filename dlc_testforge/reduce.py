from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dlc_testforge.classify import classify_validation, write_classification
from dlc_testforge.validate import validate_candidate


DEFINE_RE = re.compile(r"^\s*define\b")


@dataclass(frozen=True)
class RejectedAttempt:
  attempt: str
  removed_function: str
  state: str
  reason: str

  def to_dict(self) -> dict[str, str]:
    return {
      "attempt": self.attempt,
      "removed_function": self.removed_function,
      "state": self.state,
      "reason": self.reason,
    }


@dataclass(frozen=True)
class ReductionReport:
  bundle_dir: Path
  original_candidate: Path
  original_state: str
  final_state: str
  status: str
  original_path: Path
  reduced_path: Path
  attempt_count: int
  accepted_reduction_count: int
  rejected_attempts: list[RejectedAttempt] = field(default_factory=list)
  llvm_reduce_available: bool = False
  semantic_llvm_reduce_used: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "bundle_dir": str(self.bundle_dir),
      "original_candidate": str(self.original_candidate),
      "original_state": self.original_state,
      "final_state": self.final_state,
      "status": self.status,
      "original_path": str(self.original_path),
      "reduced_path": str(self.reduced_path),
      "attempt_count": self.attempt_count,
      "accepted_reduction_count": self.accepted_reduction_count,
      "rejected_attempts": [
        attempt.to_dict() for attempt in self.rejected_attempts
      ],
      "llvm_reduce_available": self.llvm_reduce_available,
      "semantic_llvm_reduce_used": self.semantic_llvm_reduce_used,
    }


@dataclass(frozen=True)
class FunctionBlock:
  name: str
  start: int
  end: int
  text: str

  @property
  def contains_mutation(self) -> bool:
    return "DLC-MUTATION:" in self.text


def reduce_bug_bundle(
  bundle_dir: Path,
  llvm_root: Path,
  profile: str,
  out_dir: Path,
  *,
  profiles_dir: Path | None = None,
  timeout: int = 30,
) -> ReductionReport:
  bundle_dir = bundle_dir.expanduser().resolve(strict=False)
  out_dir = out_dir.expanduser().resolve(strict=False)
  if out_dir.exists() and any(out_dir.iterdir()):
    raise ValueError(f"output directory is not empty: {out_dir}")

  reproducer = _find_reproducer(bundle_dir)
  classification_path = bundle_dir / "classification.json"
  validation_path = bundle_dir / "validation.json"
  for path in [reproducer, classification_path, validation_path]:
    if not path.is_file():
      raise ValueError(f"required bundle file not found: {path}")

  classification = _load_json_object(classification_path)
  original_state = str(classification.get("state", "unknown"))
  if not original_state.startswith("bug-scout-"):
    raise ValueError(f"bundle is not a bug-scout classification: {original_state}")

  out_dir.mkdir(parents=True, exist_ok=True)
  original_path = out_dir / "original.ll"
  reduced_path = out_dir / "reduced.ll"
  shutil.copyfile(reproducer, original_path)

  llvm_reduce_available = (llvm_root / "build" / "bin" / "llvm-reduce").is_file()
  original_text = reproducer.read_text(encoding="utf-8", errors="replace")
  if reproducer.suffix != ".ll":
    reduced_path.write_text(original_text, encoding="utf-8")
    return _write_report(
      ReductionReport(
        bundle_dir=bundle_dir,
        original_candidate=reproducer,
        original_state=original_state,
        final_state=original_state,
        status="unchanged-non-ll",
        original_path=original_path,
        reduced_path=reduced_path,
        attempt_count=0,
        accepted_reduction_count=0,
        llvm_reduce_available=llvm_reduce_available,
      ),
      out_dir,
    )

  current_text = original_text
  attempt_count = 0
  accepted_count = 0
  rejected_attempts: list[RejectedAttempt] = []

  while True:
    blocks = _find_function_blocks(current_text)
    removable = [block for block in blocks if not block.contains_mutation]
    if len(blocks) <= 1 or not removable:
      break

    accepted_this_round = False
    for block in removable:
      attempt_count += 1
      attempt_name = f"attempt-{attempt_count:04d}"
      attempt_dir = out_dir / "attempts" / attempt_name
      attempt_candidate = attempt_dir / "candidate.ll"
      attempt_candidate.parent.mkdir(parents=True, exist_ok=True)
      attempt_text = _remove_block(current_text, block)
      attempt_candidate.write_text(attempt_text, encoding="utf-8")
      state = _validate_and_classify_attempt(
        llvm_root,
        profile,
        attempt_candidate,
        attempt_dir,
        profiles_dir,
        timeout,
      )
      if state == original_state:
        current_text = attempt_text
        accepted_count += 1
        accepted_this_round = True
        break
      rejected_attempts.append(
        RejectedAttempt(
          attempt=attempt_name,
          removed_function=block.name,
          state=state,
          reason="classification state changed",
        )
      )
    if not accepted_this_round:
      break

  reduced_path.write_text(current_text, encoding="utf-8")
  final_state = original_state if accepted_count == 0 else _classify_final(
    llvm_root, profile, reduced_path, out_dir, profiles_dir, timeout
  )
  if final_state != original_state:
    reduced_path.write_text(original_text, encoding="utf-8")
    final_state = original_state
    status = "unchanged-final-state-changed"
  elif accepted_count == 0:
    status = "unchanged"
  else:
    status = "reduced"

  return _write_report(
    ReductionReport(
      bundle_dir=bundle_dir,
      original_candidate=reproducer,
      original_state=original_state,
      final_state=final_state,
      status=status,
      original_path=original_path,
      reduced_path=reduced_path,
      attempt_count=attempt_count,
      accepted_reduction_count=accepted_count,
      rejected_attempts=rejected_attempts,
      llvm_reduce_available=llvm_reduce_available,
    ),
    out_dir,
  )


def reduction_summary(report: ReductionReport) -> dict[str, Any]:
  return {
    "bundle_dir": str(report.bundle_dir),
    "status": report.status,
    "original_state": report.original_state,
    "final_state": report.final_state,
    "attempt_count": report.attempt_count,
    "accepted_reduction_count": report.accepted_reduction_count,
    "reduced": str(report.reduced_path),
    "report": str(report.reduced_path.parent / "reduction.json"),
  }


def _validate_and_classify_attempt(
  llvm_root: Path,
  profile: str,
  candidate: Path,
  attempt_dir: Path,
  profiles_dir: Path | None,
  timeout: int,
) -> str:
  validation_dir = attempt_dir / "validation"
  validate_candidate(
    llvm_root,
    candidate,
    profile,
    validation_dir,
    profiles_dir=profiles_dir,
    timeout=timeout,
  )
  classification = classify_validation(
    validation_dir / "status.json", profiles_dir=profiles_dir
  )
  write_classification(classification, attempt_dir / "classification.json")
  return classification.state


def _classify_final(
  llvm_root: Path,
  profile: str,
  reduced_path: Path,
  out_dir: Path,
  profiles_dir: Path | None,
  timeout: int,
) -> str:
  final_dir = out_dir / "final"
  validate_candidate(
    llvm_root,
    reduced_path,
    profile,
    final_dir / "validation",
    profiles_dir=profiles_dir,
    timeout=timeout,
  )
  classification = classify_validation(
    final_dir / "validation" / "status.json", profiles_dir=profiles_dir
  )
  write_classification(classification, final_dir / "classification.json")
  return classification.state


def _find_function_blocks(text: str) -> list[FunctionBlock]:
  lines = text.splitlines(keepends=True)
  blocks: list[FunctionBlock] = []
  line_index = 0
  while line_index < len(lines):
    line = lines[line_index]
    if not DEFINE_RE.match(line):
      line_index += 1
      continue
    start = line_index
    balance = line.count("{") - line.count("}")
    line_index += 1
    while line_index < len(lines) and balance > 0:
      balance += lines[line_index].count("{") - lines[line_index].count("}")
      line_index += 1
    end = line_index
    block_text = "".join(lines[start:end])
    blocks.append(
      FunctionBlock(
        name=_function_name(lines[start]),
        start=start,
        end=end,
        text=block_text,
      )
    )
  return blocks


def _find_reproducer(bundle_dir: Path) -> Path:
  preferred = bundle_dir / "reproducer.ll"
  if preferred.is_file():
    return preferred
  matches = sorted(path for path in bundle_dir.glob("reproducer.*") if path.is_file())
  if matches:
    return matches[0]
  return preferred


def _remove_block(text: str, block: FunctionBlock) -> str:
  lines = text.splitlines(keepends=True)
  return "".join(lines[: block.start] + lines[block.end :])


def _function_name(line: str) -> str:
  match = re.search(r"@(\"[^\"]+\"|[-A-Za-z0-9_.$]+)\s*\(", line)
  if match is None:
    return "unknown"
  return match.group(1).strip('"')


def _load_json_object(path: Path) -> dict[str, Any]:
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise ValueError(f"invalid JSON in {path}: {exc}") from exc
  if not isinstance(data, dict):
    raise ValueError(f"JSON file must contain an object: {path}")
  return data


def _write_report(report: ReductionReport, out_dir: Path) -> ReductionReport:
  (out_dir / "reduction.json").write_text(
    json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return report
