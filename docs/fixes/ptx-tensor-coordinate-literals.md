# PTX tensor coordinate literal parsing (private candidate)

The texture-style operand grammar used for `[map,{coordinates}]` accepted
only register names. Valid `cp.async.bulk.tensor` instructions emitted or
accepted by ptxas can mix coordinate registers with integer constants, so they
were rejected before reaching the shared TMA implementation.

Reuse the existing register/literal vector representation and value reader.
Permit integer coordinate literals for tensor copies, validating coordinate
count, 32-bit register types and literal range. Other literal-vector contexts
retain their existing validation. No new timing or SASS behavior is introduced.

`PtxTensorCoordinates` covers load, store and gather4 coordinate lists,
negative values and malformed type/count/width rejection. The compiled packed
TMA launch fixtures additionally exercise literal coordinates in 3D/4D loads.
This is independent of packed data types and SASS; standalone qualification
against public main remains pending before proposing a public fix PR.
