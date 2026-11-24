#pragma once

// Test-side canonical layouts, independent of simulator address helpers.
// PTX 9.7.18.3 and CUTLASS UMMA::make_umma_desc specify these strides.
// The host fills the physical image; the numerical oracle uses logical A/B.
template <unsigned Mode, bool Transpose, unsigned Rows, unsigned K,
          unsigned Bytes, unsigned Prefix = 0>
struct MmaTestLayout {
  static_assert(Mode != 1 || Transpose, "32B atoms require MN-major");
  static constexpr unsigned Width = Mode == 6 ? 32 : Mode == 4 ? 64 : 128;
  static constexpr unsigned KAtom = Mode == 1 ? 4 : 8;
  static constexpr unsigned MnAtom = Width / Bytes;
  static constexpr unsigned Lbo = !Mode ? 128 : Transpose ? K * Width : 16;
  static constexpr unsigned Sbo = !Mode ? 256 : (Transpose ? KAtom : 8) * Width;
  static constexpr unsigned Storage =
      !Mode ? Rows * K * Bytes
            : (Transpose ? (Rows + MnAtom - 1) / MnAtom * K * Width
                         : Rows * Width) +
                  1024;
  static unsigned offset(unsigned row, unsigned k) {
    unsigned address;
    if (!Mode) {
      address = Transpose ? (row / 8) * 256 + (row % 8) * Bytes +
                                (k / 8) * 128 + (k % 8) * 16
                          : (row / 8) * 256 + (row % 8) * 16 +
                                (k * Bytes / 16) * 128 + k * Bytes % 16;
      return address;
    }
    address =
        Prefix + (Transpose ? (row / MnAtom) * Lbo + (row % MnAtom) * Bytes +
                                  (k / KAtom) * Sbo + (k % KAtom) * Width
                            : row * Width + k * Bytes);
    return Mode == 1 ? address ^ (((address >> 7) & 3) << 5)
                     : address ^ (((address >> 7) & (Width / 16 - 1)) << 4);
  }
};
