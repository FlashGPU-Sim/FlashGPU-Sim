# Native linear bulk-copy qualifiers

The retained MegaMoE lowering uses `UBLKCP.S.G [URstate][URglobal],URsize,desc[UR]`.
Support the optional qualifier on both load and store directions, reusing the
existing bulk payload, mbarrier and TMA timing paths. Qualifier validation is
shared with the existing ordinary tensor/reduction forms; unknown values fail
before payload writes or completion. `desc[URZ]` denotes the default form.

Native dependency projection includes both words of the global pointer, both
shared/mbarrier state words for loads, the transfer-size register, and both
qualifier words when present. Stores/reductions retain a single shared-address
word. Payload size and configured TMA timing constants are unchanged.

`sm100-bulk-descriptor` checks payload/completion, invalid qualifiers and complete
register dependencies, then executes plain and hinted load/store roundtrips
through both frontends using the same CUDA executable. The fixtures use recorded
CUDA 13.3 SM100 opaque qualifier encodings (first/last eviction hints). They do
not qualify the separate PTX `createpolicy` instruction, other policy encodings,
or hardware cache replacement behavior. Full retained MoE replay remains an
independent acceptance gate.
