from __future__ import annotations

import subprocess
import time
from pathlib import Path

from dlc_testforge.schemas import CommandResult


def run_command(
  argv: list[str],
  *,
  cwd: Path | None = None,
  timeout: int = 30,
) -> CommandResult:
  start = time.monotonic()
  try:
    completed = subprocess.run(
      argv,
      cwd=str(cwd) if cwd is not None else None,
      capture_output=True,
      text=True,
      timeout=timeout,
      check=False,
    )
    duration_ms = int((time.monotonic() - start) * 1000)
    return CommandResult(
      argv=argv,
      exit_code=completed.returncode,
      stdout=completed.stdout,
      stderr=completed.stderr,
      duration_ms=duration_ms,
    )
  except subprocess.TimeoutExpired as exc:
    duration_ms = int((time.monotonic() - start) * 1000)
    return CommandResult(
      argv=argv,
      exit_code=124,
      stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
      stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
      duration_ms=duration_ms,
      timed_out=True,
    )
  except FileNotFoundError as exc:
    duration_ms = int((time.monotonic() - start) * 1000)
    return CommandResult(
      argv=argv,
      exit_code=127,
      stdout="",
      stderr=str(exc),
      duration_ms=duration_ms,
    )
