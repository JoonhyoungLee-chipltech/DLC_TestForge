# DLC TestForge

DLC TestForge is a standalone helper for building DLC CodeGen test mutation
workflows around an existing LLVM checkout.

Current phases discover the local LLVM/DLC environment, build read-only indexes for
existing tests, DLC specs, and DLC TableGen/source evidence, create mutation
workspaces, generate conservative candidate files, and validate individual
candidates. They also classify validation results, write review bundles, and
conservatively reduce bug-scout reproducers. They do not mutate checked-in LLVM
files or run the DLC CodeGen suite.

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
python3 -m dlc_testforge.cli generate --llvm-root /root/LLVM --profile machine_addropt --seed llvm/test/CodeGen/DLC/machine-addropt-prera.ll --out-dir /tmp/dlc-mutation-run --dry-run
python3 -m dlc_testforge.cli generate --llvm-root /root/LLVM --profile machine_addropt --seed llvm/test/CodeGen/DLC/machine-addropt-prera.ll --out-dir /tmp/dlc-mutation-run --max-candidates 10
python3 -m dlc_testforge.cli validate --llvm-root /root/LLVM --candidate /root/LLVM/llvm/test/CodeGen/DLC/machine-addropt-prera.ll --profile machine_addropt --out-dir /tmp/dlc-validation
python3 -m dlc_testforge.cli classify --validation /tmp/dlc-validation/status.json
python3 -m dlc_testforge.cli report --run-dir /tmp/dlc-mutation-run
python3 -m dlc_testforge.cli reduce --bundle-dir /tmp/dlc-mutation-run/reports/bug-scout/candidate-0001 --llvm-root /root/LLVM --profile machine_addropt --out-dir /tmp/dlc-reduction-candidate-0001
```

Run tests with:

```bash
python3 -m pytest tests
```
