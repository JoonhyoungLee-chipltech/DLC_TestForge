# DLC TestForge

DLC TestForge is a standalone helper for building DLC CodeGen test mutation
workflows around an existing LLVM checkout.

Phase 0 only discovers the local LLVM/DLC environment and checks required tools.
It does not generate tests, mutate LLVM files, or run the DLC CodeGen suite.

## Phase 0 usage

```bash
python3 -m dlc_testforge.cli --help
python3 -m dlc_testforge.cli env --llvm-root /root/LLVM
python3 -m dlc_testforge.cli check-tools --llvm-root /root/LLVM
```

Run tests with:

```bash
python3 -m pytest tests
```
