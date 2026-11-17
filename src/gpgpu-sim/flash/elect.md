# elect.sync Support

FlashGPU-Sim models warp leader election for active lanes selected by the
instruction's membership mask:

```ptx
elect.sync leader|is_leader, membermask;
```

The simulator deterministically selects the lowest-numbered lane in the
intersection of the active mask and membership mask. Participating active
lanes receive the elected lane ID; the elected lane receives a true predicate
and the other active lanes receive false. Inactive lanes are unchanged.

This deterministic choice describes the simulator's model. Workloads should
use valid, converged membership masks for collective execution.

See the [configuration guide](../../../configs/README.md#scalar-predicate-execution)
for predicate execution timing and the [test guide](../../../tests/README.md)
for architecture-selected validation.
