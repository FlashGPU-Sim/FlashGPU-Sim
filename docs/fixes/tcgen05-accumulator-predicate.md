# TCGen05 negated accumulator predicate

PTX `enable-input-d` may name a negated predicate: CUDA 13.3 ptxas accepts
`tcgen05.mma.cta_group::1.kind::f16 ..., !p`. The PTX adapter previously ignored
that operand flag. Consequently the second MMA could overwrite its accumulator
when it should add to it. In the complete FP16/BF16 fixture this returns 4 where
the independent reference expects 8.

Apply the operand inversion in `tcgen05_read_enable_input_d`, before passing the
boolean to the existing common MMA backend. All PTX MMA families using this
adapter receive the correction; no arithmetic, timing or native adapter is
replaced. This fix depends on the new TCGen05 feature and belongs with its MMA
unit, rather than being proposed as an unrelated public-main fix.

`Tcgen05Mma.NegatedAccumulatorPredicate` uses two CTAs, nonuniform inputs, an
initial overwrite and a subsequent accumulation, with FP16 M64 and BF16 M128.
The same compiled binary is run through both frontends. The complete MMA CI
matrix becomes 24 numerical tests / 51 launches per frontend. This does not
claim new real-GPU calibration or complete coverage of every negated-predicate
instruction form.
