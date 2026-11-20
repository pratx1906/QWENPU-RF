# Commits in `rfpu-vx.patch` (on QwenPU `5af1fc0`, oldest first)

- `1dad4e4` VFIR: reference semantics, decode (uop 81, custom-2 funct7=1), vfir mnemonic, direct-form oracle tests
- `6cd06ad` vxcirc: NEW flag (slot 1 bit 11) gates the VEW datapath cases off
- `91a781d` VFIR circuit: shifted bank-A query, complex product on the RoPE MAC with z<<15 and rounding, value record via lag head; circuit tests
- `c5d55c1` Remove accidentally committed __pycache__ files
- `4255937` VPOW: reference semantics, decode (uop 82, custom-2 funct7=1 funct3=1), vpow mnemonic, independent oracle tests
- `86e3385` VPOW circuit: pair-per-element bank-A query, |a|^2 on the RoPE MAC plus the running sum from the lags, value record via lag head (VH bit 15 at lags 1-2)
- `0e64cfb` VPOW circuit tests: vpow_program, 6 seeds teacher-forced every token (rd = x0 / F0 / F1 / F2 records, VFIR / VEW / VQ / VPUT / VGET mixed in)
- `15bb774` VBFLY: reference semantics (constant-geometry radix-2 butterflies, 32 RoPE twiddle pairs, i & mask), decode (uop 83, custom-2 funct7=1 funct3=2), vbfly mnemonic, independent oracle tests (full FFT64 exact)
- `824c694` VBFLY circuit: bank A / B read pair hoff/2 + 16 rhi + j (L1 forms), RoPE table 4 -> 32 pairs read at (i & mask), butterfly on the RoPE MAC (b x twiddle, +-, a << 15, 2^(14+sc), >> 15+sc); circuit tests (6 seeds, every token); 13 layers, 137 heads, +904 FFN units
- `b72623e` VBFLY track mode: outputs as VE VE then a word from the lags, max |y| tracked as VEW's (keep), so VGET 3 gives a stage's largest output bit length (block floating point); 13 layers (a compare on the MAC output made 14); ref + circuit tests
- `2037c86` VFIR guard bit (rs2 bit 22, slot 1 bit 9): rd = sat16(y >> 2), for taps pre-scaled by 4 with a chained rounding constant (one final rounding); ref + circuit tests; 13 layers
