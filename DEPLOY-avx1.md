# AVX1 box - deployment state and update runbook

Machine: Xeon E5-2470 v2 (AVX1 only), 78 GB DDR3 (5/6 slots), RTX 5060 Ti 16 GB.
Service: `inference-flashnext.service` -> `run-iq3_s.sh` -> `engine/strata`.

## What carries the optimization (in order of importance)

1. **Upstream PRs** (if merged, everything below becomes unnecessary):
   - #1697 = `avx1-iq-kernels` (tag `avx1-iq-0.1.41`) - iq_avx1 kernels + dispatch.
   - #1701 = `q2-avx1-rung` (tag `q2-avx1-0.1.41`) - fixes #1699 SIGILL.
2. **Fork branches + tags** on https://github.com/nashcap/Strata - rebase targets per upstream release.
3. **This branch (`local-deploy`)**: the tuned config itself (`run-iq3_s.sh`, `strata-iq3_s.json`) - the only copy otherwise lives on this disk.

## Update workflow when a new upstream release lands (PRs NOT yet merged)

```
git fetch origin
git checkout avx1-iq-kernels
git rebase --onto <new-tag-or-origin/main> avx1-iq-0.1.41 avx1-iq-kernels   # conflicts expected only in native_expert.cpp / CMakeLists.txt
cd build && ninja strata iq_avx1_parity
./iq_avx1_parity && ./q8k_quant_parity && ./iq_multi_parity && ./pool_tasks_test   # ALL must pass
systemctl stop inference-flashnext && cp build/strata engine/strata && systemctl start inference-flashnext
```

Verify after start: startup log has `the expert kernels run on AVX1 128-bit (iq_avx1, the older-CPU build)`;
warm decode ~35-38 tok/s, 45k-token prefill ~1300 tok/s. If either number collapses, the dispatch broke in the rebase.

When a PR merges, redo the branch against the new main and expect it to become empty:
`git rebase origin/main avx1-iq-kernels` -> drop the applied commits.

## Profiling (only when needed, revert after)

```
sudo sysctl kernel.perf_event_paranoid=1 kernel.kptr_restrict=0   # enable perf record
# ... perf record -F 399 -g -p $(pgrep -f engine/strata) -o /tmp/x.data sleep 30 ...
sudo sysctl kernel.perf_event_paranoid=2 kernel.kptr_restrict=1   # REVERT (kptr_restrict=0 leaks kernel addrs)
```
Both reset to defaults on reboot anyway.

## Tuned settings that are NOT in the code (survive everything, but check after updates)

- `run-iq3_s.sh`: `export STRATA_IQ_MT_MIN=1` (AVX1 gate/up kernel wins even at nt=1; without it decode gu rows fall back to ggml generic).
- `strata-iq3_s.json`: `--prefill auto:32768` (+17.5% prefill), `--spec 4 --spec-min-p 0.5` (5/6 measured flat), `--kv q4_0 --kv-resident 32768`, `--expert-cache auto`.
- Measured-rejected knobs (do not re-add without new evidence): `--ple-io mmap` (catastrophic), `--pool-workers 18` (-5%), `--ple-inflight 512` (flat), spec 5/6 (flat), swappiness changes (user: no).
- Pending hardware: 6th DDR3 stick -> add `--ple-io ram` (28.8 GB PLE table pinned, biggest remaining prefill win).
