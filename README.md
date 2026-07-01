# DLC TestForge

DLC TestForge is a standalone helper for building DLC CodeGen test mutation
workflows around an existing LLVM checkout.

Current phases discover the local LLVM/DLC environment and build read-only indexes for
existing tests, DLC specs, and DLC TableGen/source evidence. They do not generate tests,
mutate LLVM files, or run the DLC CodeGen suite.

## Usage

```bash
python3 -m dlc_testforge.cli --help
python3 -m dlc_testforge.cli env --llvm-root /root/LLVM
python3 -m dlc_testforge.cli check-tools --llvm-root /root/LLVM
python3 -m dlc_testforge.cli index --llvm-root /root/LLVM --out /tmp/dlc-test-index.json
python3 -m dlc_testforge.cli extract-spec --llvm-root /root/LLVM --out /tmp/dlc-spec-index.json
python3 -m dlc_testforge.cli lookup-spec --index /tmp/dlc-spec-index.json --topic DMA
python3 -m dlc_testforge.cli extract-td --llvm-root /root/LLVM --out /tmp/dlc-td-index.json
python3 -m dlc_testforge.cli list-profiles --llvm-root /root/LLVM
```

Run tests with:

```bash
python3 -m pytest tests
```
