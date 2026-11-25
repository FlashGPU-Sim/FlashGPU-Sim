# Native 3D tiled TMA stores

`UTMASTG.3D` uses the same tensor-coordinate traversal, shared-memory swizzle,
global write path and bulk-group completion machinery as the existing ranks.
Its opaque map is decoded by the existing Driver/3D descriptor readers. No new
store timing, latency or bandwidth parameters are introduced.

`DriverTma.Store3DRoundtrip` loads a tile and stores it to a separate tensor at
nonzero coordinates. Three launches cover byte, 16-bit and 32-bit elements,
linear and 128-byte-swizzled shared layouts, and guards outside the destination
tile. The same executable runs through PTX and SASS. PTX already supports this
form; the native handler and regression coverage were missing.

This slice does not add packed/sub-byte store formats, reduce operations, or
new tensor-map encodings.
