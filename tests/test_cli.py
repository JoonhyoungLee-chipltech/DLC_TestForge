from __future__ import annotations

from dlc_testforge.cli import main


def test_help_exits_without_llvm_checkout(capsys):
  try:
    main(["--help"])
  except SystemExit as exc:
    assert exc.code == 0

  captured = capsys.readouterr()
  assert "env" in captured.out
  assert "check-tools" in captured.out
  assert "index" in captured.out
  assert "extract-spec" in captured.out
  assert "lookup-spec" in captured.out


def test_env_missing_llvm_root_exits_nonzero(tmp_path, capsys):
  missing = tmp_path / "missing"

  assert main(["env", "--llvm-root", str(missing)]) == 2

  captured = capsys.readouterr()
  assert "llvm_root" in captured.out
  assert str(missing) in captured.out
