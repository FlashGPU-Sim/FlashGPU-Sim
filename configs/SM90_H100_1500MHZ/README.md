# SM90_H100_1500MHZ Configuration

Full H100 configuration with core, interconnect, and L2 clocks fixed at
1.5 GHz for comparisons against NCU runs that report around 1.5 GHz SM clocks.

Key difference from `SM90_H100`:

- `-gpgpu_clock_domains 1500:1500:1500:14000`

Run from `test/`:

```bash
./run_tests.sh -c SM90_H100_1500MHZ refresh
./run_tests.sh -c SM90_H100_1500MHZ hopper Fa3PrefillFp16SmallTest.H32D64FullB32S256
```
