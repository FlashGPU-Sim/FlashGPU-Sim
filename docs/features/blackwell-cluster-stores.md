# Asynchronous shared cluster stores

This slice migrates the retained PTX `st.async` path and connects it to the
per-lane mbarrier completion transport already used by native STAS. The PTX
adapter decodes registers and resolves cluster ownership; both frontends use
`cluster_store.h` to validate payload size, alignment, cluster size and matching
destination/barrier owners. Payload bytes use the existing shared-memory
objects. There is no additional PTX-only completion engine.

The shared-memory forms follow
[PTX st.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async):
4/8/16-byte payloads, scalar and supported vector types, `.b128`, explicit
`.shared::cluster` or generic shared pointers, and implicit/explicit `.weak`.
The destination can be local or remote within a cluster of more than one CTA.
Each active lane credits exactly its payload byte count to the destination
barrier. PTX predecode records both address registers and the payload as inputs;
the instruction uses barrier completion metadata rather than an ordinary LSU
store transaction.

The independent parser fix accepts both `.v4.u32` and `.u32.v4` suffix order.
Its standalone vector load/store fixture contains no cluster instruction, so
it can qualify independently of this feature. Public-main-only cherry-pick
qualification remains separate from this private-stack test.

`CI_JOB=sm100-cluster-store` checks the shared request contract, PTX dependency
metadata and numerical launches through both frontends. Two independent
clusters execute two barrier phases per launch. The matrix covers every listed
scalar element type across representative widths, local/peer/mixed ownership,
a predicated nonzero lane subset, `.b128`, generic addressing and both vector
suffix orders. Sentinels verify inactive lanes and untouched payload words.

Global `.release` / `.mmio` async stores are a different operation and are not
implemented by this shared-cluster adapter; they fail closed. Invalid owner,
alignment, width and single-CTA-cluster contracts are rejected. No new physical
DSM payload bandwidth or latency calibration is claimed: functional payload
publication and the existing timed completion path are retained. Full DeepInfer
acceptance requires its own frozen-runtime numerical replay.

The qualified launch API supplies cluster dimensions explicitly through
`cudaLaunchKernelEx`. The default-launch probe found that PTX already honors
`.reqnctapercluster`, while native cubin required-cluster metadata has not yet
been migrated into this extraction stack. Its original failing evidence is
retained; automatic native cluster launch needs a separate metadata fix.
