# PTX vector modifier order

The PTX grammar previously accepted only vector-before-scalar suffix order.
The parser now also accepts scalar-before-vector, retaining the same vector
width/type metadata and execution backend. This is one additional grammar
alternative, with no memory timing or transaction policy changes.

`PtxVectorOrder.ScalarBeforeOrAfterVector` launches a 32-thread CUDA kernel
containing both modifier orders for global loads/stores: u32.v4, v4.u32,
v2.u64 and u64.v2. Two independently checked permutations validate all 256
output words after synchronization and copy-back. CUDA 13.3 ptxas accepts
the source, with vector-order warnings.

The same frozen executable aborts while parsing on public base b3378cb9 and
passes the numerical launch with the corrected runtime. The fixture is in all
three architecture manifests (SM90/SM100/SM120); only the SM100 local run was
qualified. This is not a hosted CI or hardware performance claim.

Workspace evidence: `runs/public-vector-order-20261125/`, including binary and
runtime hashes, configuration, build logs and qualification.json. Initial
`setup-error-` logs lack the standard setup_environment runtime variables and
are not the regression reproducer.
