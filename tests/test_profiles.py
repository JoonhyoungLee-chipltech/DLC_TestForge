from __future__ import annotations

import json

import pytest

from dlc_testforge.cli import main
from dlc_testforge.profiles import get_profile, load_profiles, profile_summary


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _profile_yaml(name="example"):
  return f"""name: {name}
description: Example profile.
seed_selectors:
  paths:
    - llvm/test/CodeGen/DLC/example.ll
  features:
    - llc
commands:
  base:
    - "{{llc}} -mtriple=dlc {{input}} -o -"
validation:
  allow_verify_machineinstrs: false
  required_levels:
    - syntax
mutation_axes:
  immediates:
    enabled: true
    values: [0, 1]
bug_scout:
  enabled: true
  classify_crash: true
"""


def test_default_profiles_load_in_deterministic_order():
  profiles = load_profiles()

  assert [profile.name for profile in profiles] == ["globalisel", "machine_addropt"]
  assert get_profile("machine_addropt").seed_selectors["paths"] == [
    "llvm/test/CodeGen/DLC/machine-addropt-prera.ll"
  ]


def test_profile_summary_is_compact():
  summary = profile_summary(get_profile("globalisel"))

  assert summary["name"] == "globalisel"
  assert summary["seed_count"] == 3
  assert "global-isel" in summary["features"]
  assert summary["source_count"] > 0
  assert summary["spec_source_count"] > 0


def test_invalid_yaml_fails_with_clear_error(tmp_path):
  _write(tmp_path / "broken.yaml", "name: [unterminated\n")

  with pytest.raises(ValueError, match="invalid YAML"):
    load_profiles(tmp_path)


def test_missing_required_field_fails(tmp_path):
  _write(
    tmp_path / "missing.yaml",
    """name: missing
description: Missing required fields.
""",
  )

  with pytest.raises(ValueError, match="missing required field"):
    load_profiles(tmp_path)


def test_duplicate_profile_names_fail(tmp_path):
  _write(tmp_path / "one.yaml", _profile_yaml("dup"))
  _write(tmp_path / "two.yaml", _profile_yaml("dup"))

  with pytest.raises(ValueError, match="duplicate profile"):
    load_profiles(tmp_path)


def test_non_mapping_profile_fails(tmp_path):
  _write(tmp_path / "list.yaml", "- not\n- a\n- mapping\n")

  with pytest.raises(ValueError, match="YAML mapping"):
    load_profiles(tmp_path)


def test_cli_list_profiles_outputs_json(capsys):
  assert main(["list-profiles", "--llvm-root", "/root/LLVM"]) == 0

  captured = capsys.readouterr()
  result = json.loads(captured.out)
  assert result["profile_count"] == 2
  assert [profile["name"] for profile in result["profiles"]] == [
    "globalisel",
    "machine_addropt",
  ]


def test_cli_list_profiles_supports_custom_directory(tmp_path, capsys):
  _write(tmp_path / "example.yaml", _profile_yaml("example"))

  assert (
    main(
      [
        "list-profiles",
        "--llvm-root",
        "/root/LLVM",
        "--profiles-dir",
        str(tmp_path),
      ]
    )
    == 0
  )

  captured = capsys.readouterr()
  result = json.loads(captured.out)
  assert result["profile_count"] == 1
  assert result["profiles"][0]["name"] == "example"


def test_loading_profiles_does_not_scan_indexes(monkeypatch):
  def fail(*args, **kwargs):
    raise AssertionError("profile loading should not scan indexes")

  monkeypatch.setattr("dlc_testforge.index.build_index", fail)
  monkeypatch.setattr("dlc_testforge.extract_spec.build_spec_index", fail)
  monkeypatch.setattr("dlc_testforge.extract_td.build_td_index", fail)

  assert load_profiles()
