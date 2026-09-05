# SM90_H200 (development only)

This legacy flat 132×1 H200 preset exists only while development is in
progress and will be removed. It cannot exercise the modeled intra-GPC DSM
fabric and must not be used for calibration or published results.

All future H200 realism work targets `SM90_H200_CLUSTER132`. The old hardware
measurements formerly documented here are superseded; the replacement
exclusive H200 run is pending.

The retained hard memory geometry is derived from the H200 NVL datasheet:
6016-bit HBM bus / 64 bits per controller = 94 controllers, with two L2
subpartitions each (188 slices, 58.75 MiB under the current slice geometry).
