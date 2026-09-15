# Frozen FA3 SASS forward baseline — 2026-09-14

Four medium and twenty large cases completed. Strict SASS timing, TMA on, no instrumentation, no backward. Time comparison: SASS kernel cycles / 1500 MHz; error = (SASS us / NCU us - 1) * 100. Do not substitute simulator total cycles or NCU active cycles.

## Validation policy

- During rebase/refactoring: rerun these four medium binaries with the frozen config, four threads/job; require unchanged kernel cycles and successful execution. Investigate every delta before changing this baseline.
- Before PR: rerun all twenty large forward cases. No large rerun is needed after each structural change.
- Medium and large use different recorded binary packages; do not replace them with freshly built or instrumented binaries. Medium GTest success checks execution, not a new numerical-reference comparison.
- This is a whole-kernel timing baseline, not a claim that all internal phases match hardware.

## Configuration and provenance

Use `config/` here, not the mutable architecture default: CCD=2, core=1500 MHz. Official architecture configs were not changed. `results.csv` identifies every binary, log and hardware source. `provenance.json` distinguishes freeze HEAD from the historical modified build banner; exact original per-run library hashes were not recorded. The evidence snapshot includes the current library, source archive, dirty patch, workload binaries, original logs/configs/statuses and referenced NCU CSVs. Large raw NCU reports remain in their original collection; its summary and provenance are retained here.

Local evidence: `tests/run/fa3-sass-ccd2-20260914-evidence` (ignored, not committed). Verify using `sha256sum -c SHA256SUMS` inside that directory. Compact config/results/provenance are versioned; no generated SASS decode cache is committed.

## Results

| Profile | B | S | H | D | Causal | SASS cycles | SASS us | NCU us | Error |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| medium | 16 | 512 | 32 | 64 | False | 178429 | 118.952667 | 119.392 | -0.37% |
| medium | 16 | 512 | 32 | 64 | True | 122817 | 81.878000 | 88.160 | -7.13% |
| medium | 16 | 512 | 16 | 128 | False | 116150 | 77.433333 | 71.712 | +7.98% |
| medium | 16 | 512 | 16 | 128 | True | 97882 | 65.254667 | 65.312 | -0.09% |
| large | 64 | 512 | 32 | 64 | False | 667970 | 445.313333 | 465.920 | -4.42% |
| large | 64 | 512 | 32 | 64 | True | 474589 | 316.392667 | 336.450 | -5.96% |
| large | 64 | 512 | 16 | 128 | False | 412963 | 275.308667 | 268.260 | +2.63% |
| large | 64 | 512 | 16 | 128 | True | 378229 | 252.152667 | 244.060 | +3.32% |
| large | 32 | 1024 | 32 | 64 | False | 1161411 | 774.274000 | 804.060 | -3.70% |
| large | 32 | 1024 | 32 | 64 | True | 722765 | 481.843333 | 515.300 | -6.49% |
| large | 32 | 1024 | 16 | 128 | False | 722026 | 481.350667 | 457.600 | +5.19% |
| large | 32 | 1024 | 16 | 128 | True | 472394 | 314.929333 | 305.120 | +3.21% |
| large | 16 | 2048 | 32 | 64 | False | 1818194 | 1212.129333 | 1240.000 | -2.25% |
| large | 16 | 2048 | 32 | 64 | True | 1016641 | 677.760667 | 734.400 | -7.71% |
| large | 16 | 2048 | 16 | 128 | False | 1352438 | 901.625333 | 842.720 | +6.99% |
| large | 16 | 2048 | 16 | 128 | True | 748097 | 498.731333 | 470.140 | +6.08% |
| large | 8 | 4096 | 32 | 64 | False | 3477293 | 2318.195333 | 2350.000 | -1.35% |
| large | 8 | 4096 | 32 | 64 | True | 1819008 | 1212.672000 | 1300.000 | -6.72% |
| large | 8 | 4096 | 16 | 128 | False | 2621923 | 1747.948667 | 1610.000 | +8.57% |
| large | 8 | 4096 | 16 | 128 | True | 1344907 | 896.604667 | 841.180 | +6.59% |
| large | 4 | 8192 | 32 | 64 | False | 6491279 | 4327.519333 | 4380.000 | -1.20% |
| large | 4 | 8192 | 32 | 64 | True | 3278488 | 2185.658667 | 2330.000 | -6.19% |
| large | 4 | 8192 | 16 | 128 | False | 5065339 | 3376.892667 | 3110.000 | +8.58% |
| large | 4 | 8192 | 16 | 128 | True | 2578778 | 1719.185333 | 1590.000 | +8.12% |
