# Collector-set scheduler ownership

With the subcore model enabled, collector allocation and dispatch partition each
collector set independently. Initialization instead used the flattened array
of all sets. A 16-entry generic set plus a 4-entry specialized set therefore
initialized scheduler IDs in groups of five, while allocation expected groups
of four and one respectively. A collector could look for an input in the wrong
scheduler partition, producing an `allocated` assertion failure.

Initialization now derives scheduler ownership from the index within its own
set. Arbiter IDs retain their original global registration order, including
when set registration order differs from map order. The single-set and
nonpartitioned mappings are unchanged. No latency, capacity or arbitration
policy changes. This is a frontend-independent public bug fix.

`OperandCollectorAllocationTest` now calls the production initialization helper
instead of manually assigning the correct scheduler IDs in the fixture. Two
new cases cover unequal set sizes, an empty set, nonpartitioned operation and
registration order. All 11 allocation/dispatch tests pass in the SM120 host test
binary (the architecture manifest containing these host-only tests).

The resource-contention diagnostic additionally reproduces 12/12 assertion
failures with the frozen pre-fix runtime and passes 12/12 numerical checks after
the fix using identical specialized-collector overrides. Receipts are in the
workspace `runs/CURRENT.json` resource-model entry. These runs do not calibrate
the specialized configuration: reserving an integer collector alone does not
remove the STS/TMA timing mismatch, so the diagnostic preset is not promoted.
