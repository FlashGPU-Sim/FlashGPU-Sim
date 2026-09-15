# RTX 5090 SASS frontend configuration

Select with `--config SM120_RTX5090_SASS_FRONTEND --sass-timing`.
This copies the current local SM120 SASS calibration, including the 1800 MHz
core/interconnect/L2 clocks, without retuning parameters. The original
`SM120_RTX5090` remains the PTX configuration in the PR.

The config explicitly selects `sass-timing`; decoder tools are supplied by
the test runner. Parameter meanings are in
[PARAMETERS.md](../../src/gpgpu-sim/flash/sass/PARAMETERS.md).
This split is not a new hardware calibration result.
