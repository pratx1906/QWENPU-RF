"""RF DSP programs for QwenPU's RV32 CPU, executed free-running on its fused executor.

Numerical contract (Q15): samples and coefficients are signed 16-bit codes; products are
exact 32/64-bit integers; requantization is round-half-up, (x + 2^(s-1)) >> s, arithmetic.
Every kernel has an independent pure-Python integer reference. All sample arithmetic runs
as RV32 instructions on the CPU compiled into the weights; the host only writes input words
into RAM, starts the machine and reads back words the machine itself loaded.
"""
