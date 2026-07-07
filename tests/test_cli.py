from __future__ import annotations

import json

from dlc_testforge.cli import main


def _write(path, text=""):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


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
  assert "extract-td" in captured.out
  assert "extract-kernel" in captured.out
  assert "lookup-spec" in captured.out
  assert "list-profiles" in captured.out
  assert "generate" in captured.out
  assert "validate" in captured.out
  assert "classify" in captured.out
  assert "report" in captured.out
  assert "reduce" in captured.out


def test_env_missing_llvm_root_exits_nonzero(tmp_path, capsys):
  missing = tmp_path / "missing"

  assert main(["env", "--llvm-root", str(missing)]) == 2

  captured = capsys.readouterr()
  assert "llvm_root" in captured.out
  assert str(missing) in captured.out


def test_cli_extract_kernel_writes_usage_index(tmp_path):
  kernel_root = tmp_path / "DLC_Custom_Kernel"
  _write(
    kernel_root / "dlc_kernels" / "foo.c",
    """void f(tensor h, tensor v, int size) {
  int a = dlc_dma(h, HBM, v, VMEM, 128, 128, 128, 128, 7);
  dlc_sync(a);
  int mask = pre_exp2(size / 128);
  float8_128 x = v_f32_ld_tnsr_st_msk(0, v, 1, mask);
}
""",
  )
  out = tmp_path / "out" / "kernel-usage-index.json"

  assert main([
    "extract-kernel",
    "--kernel-root",
    str(kernel_root),
    "--out",
    str(out),
  ]) == 0

  data = json.loads(out.read_text(encoding="utf-8"))
  assert data["schema_version"] == 1
  assert data["summary"]["kernel_count"] == 1
  assert data["summary"]["dma_call_count"] == 1
  assert data["summary"]["edge_hint_count"] > 0
  assert data["kernels"][0]["source"] == "dlc_kernels/foo.c"


def test_cli_extract_kernel_missing_root_exits_nonzero(tmp_path, capsys):
  missing = tmp_path / "missing"
  out = tmp_path / "out" / "kernel-usage-index.json"

  assert main([
    "extract-kernel",
    "--kernel-root",
    str(missing),
    "--out",
    str(out),
  ]) == 2

  captured = capsys.readouterr()
  assert "error:" in captured.err
  assert not out.exists()
