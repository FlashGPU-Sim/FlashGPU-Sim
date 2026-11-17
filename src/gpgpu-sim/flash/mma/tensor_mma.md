# MMA Numerical Behavior

See [MMA support](README.md) for the integration-tested shapes and input types.
The following describes the functional model's current precision behavior.

| Conversion | Model behavior |
| --- | --- |
| F16 to F32 | Expand the sign, exponent and significand, including half subnormals |
| F32 to F16 | Truncate the significand; underflow is flushed to signed zero and overflow produces infinity |
| BF16 to F32 | Preserve the BF16 bits in the upper half of the F32 representation |
| F32 to BF16 | Truncate the lower 16 bits |
| F32 to TF32 precision | Clear the lower 13 significand bits |

Floating-point MMA uses FP32 arithmetic for the modeled accumulation. The conversion
helpers apply the output behavior listed above where used. Integer
MMA accumulates into S32; the integration tests compare integer results exactly.

These conversions are model approximations. In particular, the F32-to-F16
helper does not implement round-to-nearest-even, and the TF32 path truncates
inputs. Floating-point integration checks use format-appropriate tolerances;
they do not establish bit-exact agreement with every hardware rounding or
exceptional-value case.

Use the [MMA tests](../../../../tests/src/mma/README.md) for functional
validation, [MMA microbenchmarks](../../../../tests/src/microbench/mma/README.md)
for timing probes, and the [configuration guide](../../../../configs/README.md#matrix-execution)
for latency, initiation interval and resource settings.
