# Pipeline dumps with empty thread slots

`ptx_thread_done()` deliberately returns true for an unallocated thread slot.
The optional done-cycle diagnostics then dereferenced that null thread pointer.
Consequently, a pipeline dump could crash even on an idle ordinary PTX core;
native SASS cores exposed the same assumption during MoE debugging.

Guard the done-cycle read on an actual thread object. This changes diagnostics
only; active warp state, timing and frontend selection are untouched. No
SASS-specific lifecycle workaround is involved.

This branch starts from public `flash` at `b3378cb9`. The test invokes the real
pipeline dump on an initialized idle core, with and without the done-cycle
mask. Before the fix, the control passes and the done-cycle call segfaults;
with the same test executable and configuration, both pass after the fix.
Fixtures are included in the SM90/SM100/SM120 unit manifests. Local SM100
qualification is recorded under workspace `runs/public-pipeline-dump-20261125/`;
no hosted CI or public PR is claimed.
