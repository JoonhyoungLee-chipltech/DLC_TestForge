from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REGRESSION_STATE = "accepted-regression-candidate"
BUG_PREFIX = "bug-scout-"


@dataclass(frozen=True)
class BundleRecord:
  candidate: str
  state: str
  path: str | None
  skipped_reason: str | None = None

  def to_dict(self) -> dict[str, Any]:
    return {
      "candidate": self.candidate,
      "state": self.state,
      "path": self.path,
      "skipped_reason": self.skipped_reason,
    }


@dataclass(frozen=True)
class ReportSummary:
  run_dir: Path
  classification_count: int
  regression_bundles: list[BundleRecord] = field(default_factory=list)
  bug_bundles: list[BundleRecord] = field(default_factory=list)
  skipped: list[BundleRecord] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": 1,
      "run_dir": str(self.run_dir),
      "classification_count": self.classification_count,
      "regression_bundle_count": len(self.regression_bundles),
      "bug_bundle_count": len(self.bug_bundles),
      "skipped_count": len(self.skipped),
      "reports_dir": str(self.run_dir / "reports"),
      "regression_bundles": [record.to_dict() for record in self.regression_bundles],
      "bug_bundles": [record.to_dict() for record in self.bug_bundles],
      "skipped": [record.to_dict() for record in self.skipped],
    }


def write_report_bundle(run_dir: Path) -> ReportSummary:
  run_dir = run_dir.expanduser().resolve(strict=False)
  manifest_path = run_dir / "manifest.json"
  classifications_dir = run_dir / "results" / "classifications"
  if not manifest_path.is_file():
    raise ValueError(f"manifest.json not found: {manifest_path}")
  if not classifications_dir.is_dir():
    raise ValueError(f"classifications directory not found: {classifications_dir}")

  manifest = _load_json_object(manifest_path)
  classifications = [
    _load_json_object(path)
    for path in sorted(classifications_dir.glob("*.json"))
    if path.is_file()
  ]
  mutation_records = _mutation_records(manifest, run_dir)
  regression_bundles: list[BundleRecord] = []
  bug_bundles: list[BundleRecord] = []
  skipped: list[BundleRecord] = []

  for index, classification in enumerate(classifications, start=1):
    state = str(classification.get("state", "unknown"))
    candidate = str(classification.get("candidate", ""))
    candidate_id = _candidate_id(candidate, index)
    mutation = mutation_records.get(_candidate_key(candidate))

    if state == REGRESSION_STATE:
      bundle = run_dir / "reports" / "regression-candidates" / candidate_id
      _write_regression_bundle(bundle, classification, manifest, mutation)
      regression_bundles.append(
        BundleRecord(candidate=candidate, state=state, path=str(bundle))
      )
    elif state.startswith(BUG_PREFIX):
      bundle = run_dir / "reports" / "bug-scout" / candidate_id
      _write_bug_bundle(bundle, classification, manifest, mutation)
      bug_bundles.append(BundleRecord(candidate=candidate, state=state, path=str(bundle)))
    else:
      skipped.append(
        BundleRecord(
          candidate=candidate,
          state=state,
          path=None,
          skipped_reason="state is not bundled in Phase 9 MVP",
        )
      )

  return ReportSummary(
    run_dir=run_dir,
    classification_count=len(classifications),
    regression_bundles=regression_bundles,
    bug_bundles=bug_bundles,
    skipped=skipped,
  )


def write_report_summary(summary: ReportSummary, out_path: Path) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(
    json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _write_regression_bundle(
  bundle: Path,
  classification: dict[str, Any],
  manifest: dict[str, Any],
  mutation: dict[str, Any] | None,
) -> None:
  bundle.mkdir(parents=True, exist_ok=True)
  _copy_candidate(classification, bundle / "candidate.ll")
  _copy_json_file(classification.get("validation"), bundle / "validation.json")
  _write_json(classification, bundle / "classification.json")
  (bundle / "command.log").write_text(_command_log(classification), encoding="utf-8")
  (bundle / "summary.md").write_text(
    _regression_summary(classification, manifest, mutation), encoding="utf-8"
  )


def _write_bug_bundle(
  bundle: Path,
  classification: dict[str, Any],
  manifest: dict[str, Any],
  mutation: dict[str, Any] | None,
) -> None:
  bundle.mkdir(parents=True, exist_ok=True)
  _copy_candidate(classification, bundle / "reproducer.ll")
  _copy_json_file(classification.get("validation"), bundle / "validation.json")
  _write_json(classification, bundle / "classification.json")
  (bundle / "stdout.txt").write_text(
    str(classification.get("stdout_excerpt", "")), encoding="utf-8"
  )
  (bundle / "stderr.txt").write_text(
    str(classification.get("stderr_excerpt", "")), encoding="utf-8"
  )
  (bundle / "command.sh").write_text(_primary_command(classification) + "\n", encoding="utf-8")
  (bundle / "summary.md").write_text(
    _bug_summary(classification, manifest, mutation), encoding="utf-8"
  )


def _regression_summary(
  classification: dict[str, Any],
  manifest: dict[str, Any],
  mutation: dict[str, Any] | None,
) -> str:
  destination = _suggested_destination(classification, manifest)
  return "\n".join(
    [
      "# Regression Candidate",
      "",
      f"- State: {classification.get('state', 'unknown')}",
      f"- Profile: {classification.get('profile', manifest.get('profile', ''))}",
      f"- Seed: {manifest.get('seed', '')}",
      f"- Candidate: {classification.get('candidate', '')}",
      f"- Mutation: {_mutation_description(mutation)}",
      f"- Validation command: {_primary_command(classification)}",
      f"- Suggested destination: {destination}",
      "",
      "## Review Tasks",
      "",
      "- Confirm the candidate checks the intended DLC behavior.",
      "- Tighten FileCheck patterns before landing.",
      "- Place the final test under the suggested DLC CodeGen path.",
      "",
      _missing_artifacts_note(classification),
    ]
  )


def _bug_summary(
  classification: dict[str, Any],
  manifest: dict[str, Any],
  mutation: dict[str, Any] | None,
) -> str:
  return "\n".join(
    [
      "# Bug Scout Candidate",
      "",
      f"- Failure category: {classification.get('state', 'unknown')}",
      f"- Reason: {classification.get('reason', '')}",
      f"- Profile: {classification.get('profile', manifest.get('profile', ''))}",
      f"- Seed: {manifest.get('seed', '')}",
      f"- Candidate: {classification.get('candidate', '')}",
      f"- Mutation: {_mutation_description(mutation)}",
      f"- Suspected stage: {_suspected_stage(classification)}",
      f"- Command: {_primary_command(classification)}",
      f"- Reduced: no",
      "",
      "## Triage Tasks",
      "",
      "- Reproduce the failure from command.sh.",
      "- Confirm whether the failure is compiler bug, invalid candidate, or missing spec rule.",
      "- Reduce only after the same failure category is reproducible.",
      "",
      _missing_artifacts_note(classification),
    ]
  )


def _load_json_object(path: Path) -> dict[str, Any]:
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise ValueError(f"invalid JSON in {path}: {exc}") from exc
  if not isinstance(data, dict):
    raise ValueError(f"JSON file must contain an object: {path}")
  return data


def _write_json(data: dict[str, Any], path: Path) -> None:
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_candidate(classification: dict[str, Any], destination: Path) -> None:
  candidate = Path(str(classification.get("candidate", "")))
  if candidate.is_file():
    shutil.copyfile(candidate, destination)
  else:
    destination.write_text(
      f"; missing candidate artifact: {classification.get('candidate', '')}\n",
      encoding="utf-8",
    )


def _copy_json_file(source: object, destination: Path) -> None:
  if isinstance(source, str) and Path(source).is_file():
    shutil.copyfile(Path(source), destination)
  else:
    _write_json({"missing_artifact": source}, destination)


def _command_log(classification: dict[str, Any]) -> str:
  lines = ["# Commands", ""]
  for step in _evidence_steps(classification):
    command = step.get("command")
    if command:
      lines.append(str(command))
  lines.extend(["", "# Stdout Excerpt", "", str(classification.get("stdout_excerpt", ""))])
  lines.extend(["", "# Stderr Excerpt", "", str(classification.get("stderr_excerpt", ""))])
  return "\n".join(lines) + "\n"


def _primary_command(classification: dict[str, Any]) -> str:
  for step in _evidence_steps(classification):
    if step.get("level") == "command" and step.get("command"):
      return str(step["command"])
  for step in _evidence_steps(classification):
    command = step.get("command")
    if command:
      return str(command)
  return ""


def _evidence_steps(classification: dict[str, Any]) -> list[dict[str, Any]]:
  steps = classification.get("evidence_steps", [])
  if not isinstance(steps, list):
    return []
  return [step for step in steps if isinstance(step, dict)]


def _mutation_records(manifest: dict[str, Any], run_dir: Path) -> dict[str, dict[str, Any]]:
  records: dict[str, dict[str, Any]] = {}
  for candidate in manifest.get("candidates", []):
    if not isinstance(candidate, dict):
      continue
    path = candidate.get("path")
    if not isinstance(path, str):
      continue
    records[_candidate_key(str(run_dir / path))] = candidate
    records[_candidate_key(path)] = candidate
  return records


def _candidate_key(candidate: str) -> str:
  return Path(candidate).name


def _candidate_id(candidate: str, index: int) -> str:
  name = Path(candidate).stem
  if name:
    return name
  return f"classification-{index:04d}"


def _mutation_description(mutation: dict[str, Any] | None) -> str:
  if mutation is None:
    return "not found in manifest"
  axis = mutation.get("mutation_axis", "")
  source = mutation.get("source_value", "")
  new = mutation.get("new_value", "")
  line = mutation.get("line", "")
  return f"{axis}: {source} -> {new} at line {line}"


def _suggested_destination(
  classification: dict[str, Any], manifest: dict[str, Any]
) -> str:
  seed = str(manifest.get("seed", "llvm/test/CodeGen/DLC"))
  candidate_name = Path(str(classification.get("candidate", ""))).name or "candidate.ll"
  return str(Path(seed).parent / candidate_name)


def _suspected_stage(classification: dict[str, Any]) -> str:
  for step in _evidence_steps(classification):
    level = step.get("level")
    command = step.get("command")
    if level:
      return f"{level}: {command or ''}".strip()
  return "unknown"


def _missing_artifacts_note(classification: dict[str, Any]) -> str:
  missing = []
  candidate = classification.get("candidate")
  if not isinstance(candidate, str) or not Path(candidate).is_file():
    missing.append(f"candidate: {candidate}")
  validation = classification.get("validation")
  if not isinstance(validation, str) or not Path(validation).is_file():
    missing.append(f"validation: {validation}")
  if not missing:
    return ""
  lines = ["## Missing Artifacts", ""]
  lines.extend(f"- {item}" for item in missing)
  return "\n".join(lines)
