from __future__ import annotations

import json

from dlc_testforge.cli import main
from dlc_testforge.extract_spec import build_spec_index, lookup_dlc_spec


SPEC_FILES = [
  "instruction-format.md",
  "individual-instructions.md",
  "assembly-code-standard.md",
  "abbreviation-dictionary.md",
  "dma-desc.md",
]


def _write(path, text):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _make_llvm_root(tmp_path, *, include_all=True):
  llvm_root = tmp_path / "LLVM"
  spec_root = llvm_root / "docs" / "dlc_spec"
  spec_root.mkdir(parents=True)
  files = SPEC_FILES if include_all else SPEC_FILES[:-1]
  for filename in files:
    _write(spec_root / filename, f"# {filename}\n\nplaceholder for {filename}\n")
  return llvm_root


def test_extracts_all_markdown_sources_when_present(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)

  index = build_spec_index(llvm_root).to_dict()

  assert index["summary"]["missing_sources"] == []
  assert sorted(index["summary"]["sources"]) == sorted([
    f"docs/dlc_spec/{filename}" for filename in SPEC_FILES
  ])


def test_missing_markdown_source_is_reported(tmp_path):
  llvm_root = _make_llvm_root(tmp_path, include_all=False)

  index = build_spec_index(llvm_root).to_dict()

  assert index["summary"]["missing_sources"] == [
    "docs/dlc_spec/dma-desc.md"
  ]


def test_headings_tables_text_bullets_and_code_blocks_are_records(tmp_path):
  llvm_root = _make_llvm_root(tmp_path, include_all=False)
  _write(
    llvm_root / "docs" / "dlc_spec" / "dma-desc.md",
    """# DMA Spec

Intro paragraph with a DMA Descriptor.

## Descriptor Fields

- Source operand uses src_core_id.
- Reserved values should not be used.

| Field | Description |
| --- | --- |
| dst_mem_id | Destination memory id can be Reserved |
| length | Out-bound access causes a fatal error |

```text
code block with mask
```
""",
  )

  index = build_spec_index(llvm_root)
  records = index.records

  assert any(record.kind == "text_block" for record in records)
  assert any(record.kind == "table_row" for record in records)
  assert any(record.kind == "code_block" for record in records)
  assert any(record.heading_path == ["DMA Spec", "Descriptor Fields"] for record in records)

  table_records = [record for record in records if record.kind == "table_row"]
  assert table_records[0].table == {
    "headers": ["Field", "Description"],
    "row": ["dst_mem_id", "Destination memory id can be Reserved"],
  }


def test_fact_extraction_is_keyword_based(tmp_path):
  llvm_root = _make_llvm_root(tmp_path, include_all=False)
  _write(
    llvm_root / "docs" / "dlc_spec" / "dma-desc.md",
    """# Facts

Immediate operand uses imm_0 in S0 slot.
Reserved DMA descriptor values can cause a fatal error.
The mask field uses dst_core_id and src_mem_id sync flag bits.
""",
  )

  records = build_spec_index(llvm_root).records
  facts = [record.facts.to_dict() for record in records]

  assert any(fact["immediates"] for fact in facts)
  assert any(fact["operands"] for fact in facts)
  assert any(fact["reserved_values"] for fact in facts)
  assert any(fact["fatal_conditions"] for fact in facts)
  assert any(fact["slot_constraints"] for fact in facts)
  assert any(fact["dma_fields"] for fact in facts)


def test_lookup_is_case_insensitive(tmp_path):
  llvm_root = _make_llvm_root(tmp_path, include_all=False)
  _write(
    llvm_root / "docs" / "dlc_spec" / "dma-desc.md",
    "# DMA\n\nDescriptor contains Sync Flag fields.\n",
  )
  index = build_spec_index(llvm_root)

  assert lookup_dlc_spec(index, "dma")
  assert lookup_dlc_spec(index, "SYNC FLAG")


def test_cli_extract_spec_and_lookup_spec(tmp_path, capsys):
  llvm_root = _make_llvm_root(tmp_path, include_all=False)
  _write(
    llvm_root / "docs" / "dlc_spec" / "dma-desc.md",
    "# DMA\n\nDescriptor contains Sync Flag fields.\n",
  )
  out = tmp_path / "out" / "spec.json"

  assert main(["extract-spec", "--llvm-root", str(llvm_root), "--out", str(out)]) == 0
  assert main(["lookup-spec", "--index", str(out), "--topic", "DMA"]) == 0

  captured = capsys.readouterr()
  result = json.loads(captured.out)
  assert result["topic"] == "DMA"
  assert result["match_count"] > 0


def test_output_json_is_deterministic(tmp_path):
  llvm_root = _make_llvm_root(tmp_path)
  out_one = tmp_path / "one.json"
  out_two = tmp_path / "two.json"

  assert main(["extract-spec", "--llvm-root", str(llvm_root), "--out", str(out_one)]) == 0
  assert main(["extract-spec", "--llvm-root", str(llvm_root), "--out", str(out_two)]) == 0

  assert json.loads(out_one.read_text(encoding="utf-8")) == json.loads(
    out_two.read_text(encoding="utf-8")
  )
