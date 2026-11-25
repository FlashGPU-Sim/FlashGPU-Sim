# PTX device-call argument ownership

`arg_buffer_t` owned a raw parameter pointer but lacked a C++ copy constructor:
the existing two-argument constructor did not intercept standard container
copies. A list insertion could therefore copy the pointer, and destruction
of either value freed storage still owned by the other. Assignment also
allocated new storage without retiring the previous allocation.

Parameter bytes now have value ownership through `std::vector<unsigned char>`.
Normal copies, assignment and destruction use its existing ownership semantics;
the context-taking compatibility constructor delegates to the actual copy.
Register and parameter access interfaces are retained. No scheduler or
instruction timing change is included.

The standalone numerical regression passes and returns a 32-byte aggregate
through three non-inlined device calls in each of 32 lanes, then validates all
128 output fields against a host calculation. The same frozen CUDA executable
aborts with `double free detected` on public base b3378cb9 and passes after the
fix. Tests are registered for SM90, SM100 and SM120; local CUDA 13.3 / SM100 was
executed. This fixture requires no TCGen05 or private SASS extensions.

Evidence: workspace `runs/public-call-ownership-20261125/qualification.json`.
No hosted CI result or new public PR is claimed.
