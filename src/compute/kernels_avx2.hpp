#pragma once
// Internal header for AVX2+FMA kernel implementations (compiled with /arch:AVX2 or -mavx2 -mfma).
// Never call these directly without checking vu::compute::avx2KernelsAvailable().
#include <cstdint>

namespace vu::kern::avx2 {

void sepFilterHorizRow(const float* L, float* dst, int w, const float* kc, int r);
void sepFilterVertRow(float* dst, const float* center, const float* const* upRows,
                      const float* const* downRows, int w, const float* kc, int r);
void btvAccumRow(const float* xr, const float* xf, const float* xb, float* g, int w, int dx, float wgt);
void u8ToFloatRow(const uint8_t* src, float* dst, int n);
void floatToU8Row(const float* src, uint8_t* dst, int n);

}  // namespace vu::kern::avx2
