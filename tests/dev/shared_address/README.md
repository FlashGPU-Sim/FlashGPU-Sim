# Pending ordinary native DSM qualification

`remote_read_test.cu` retains the complete failing dual-frontend experiment.
It is not part of the qualified SM100 suite. The PTX run passes. The native
run currently fails at bare `LD.E R21, [R6]`; nvdisasm JSON also concatenates
that form as `R21[R6]`. Native remote shared backing and timing routing need
qualification before enabling this fixture in CI. Do not count the passing
local-address fixture as remote load/store support.
