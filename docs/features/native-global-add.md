# Native discarded global add (private draft)

`REDG.E.ADD.STRONG.GPU` and `REDG.E.ADD.S32.STRONG.GPU` reuse the existing
32-bit `ATOMG.ADD` functional RMW, mutex and common memory timing proxy.
They have address/source dependencies and no destination register. The
integer addition wraps at 32 bits; no new atomic performance calibration is
introduced.

This is a SASS instruction-form extension, independent of CLC and TCGen05.
It was exposed by CLC's work-count validation kernel. The focused unit test
covers multiple lanes, negative increments, unchanged input registers and
projection dependencies. Full CLC launch acceptance additionally exercises
concurrent discarded additions across SMs. Other REDG operations, widths and
memory scopes are not enabled by this change.
