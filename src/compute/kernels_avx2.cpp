#include "compute/kernels_avx2.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>

namespace vu::kern::avx2 {

void sepFilterHorizRow(const float* L, float* dst, int w, const float* kc, int r) {
  const __m256 k0 = _mm256_set1_ps(kc[0]);
  int x = 0;
  for (; x + 8 <= w; x += 8) {
    __m256 acc = _mm256_mul_ps(k0, _mm256_loadu_ps(L + x));
    for (int i = 1; i <= r; ++i) {
      const __m256 ki = _mm256_set1_ps(kc[i]);
      const __m256 s = _mm256_add_ps(_mm256_loadu_ps(L + x - i), _mm256_loadu_ps(L + x + i));
      acc = _mm256_fmadd_ps(ki, s, acc);
    }
    _mm256_storeu_ps(dst + x, acc);
  }
  for (; x < w; ++x) {
    float acc = kc[0] * L[x];
    for (int i = 1; i <= r; ++i) acc += kc[i] * (L[x - i] + L[x + i]);
    dst[x] = acc;
  }
}

void sepFilterVertRow(float* dst, const float* center, const float* const* upRows,
                      const float* const* downRows, int w, const float* kc, int r) {
  const __m256 k0 = _mm256_set1_ps(kc[0]);
  int x = 0;
  for (; x + 8 <= w; x += 8) {
    __m256 acc = _mm256_mul_ps(k0, _mm256_loadu_ps(center + x));
    for (int i = 1; i <= r; ++i) {
      const __m256 ki = _mm256_set1_ps(kc[i]);
      const __m256 s = _mm256_add_ps(_mm256_loadu_ps(upRows[i - 1] + x), _mm256_loadu_ps(downRows[i - 1] + x));
      acc = _mm256_fmadd_ps(ki, s, acc);
    }
    _mm256_storeu_ps(dst + x, acc);
  }
  for (; x < w; ++x) {
    float acc = kc[0] * center[x];
    for (int i = 1; i <= r; ++i) acc += kc[i] * (upRows[i - 1][x] + downRows[i - 1][x]);
    dst[x] = acc;
  }
}

void btvAccumRow(const float* xr, const float* xf, const float* xb, float* g, int w, int dx, float wgt) {
  constexpr float eps = 1e-6f;
  const __m256 veps = _mm256_set1_ps(eps);
  const __m256 vwgt = _mm256_set1_ps(wgt);

  const int xStart = std::max(0, std::max(-dx, dx));
  const int xEnd = std::min(w, std::min(w - dx, w + dx));

  auto scalarAt = [&](int x) {
    const int xp = std::clamp(x + dx, 0, w - 1);
    const int xm = std::clamp(x - dx, 0, w - 1);
    const float d1 = xr[x] - xf[xp];
    const float d2 = xb[xm] - xr[x];
    const float s1 = d1 / std::sqrt(d1 * d1 + eps);
    const float s2 = d2 / std::sqrt(d2 * d2 + eps);
    g[x] += wgt * (s1 - s2);
  };

  int x = 0;
  for (; x < xStart && x < w; ++x) scalarAt(x);
  for (; x + 8 <= xEnd; x += 8) {
    const __m256 vx = _mm256_loadu_ps(xr + x);
    const __m256 vxf = _mm256_loadu_ps(xf + x + dx);
    const __m256 vxb = _mm256_loadu_ps(xb + x - dx);

    const __m256 d1 = _mm256_sub_ps(vx, vxf);
    const __m256 d2 = _mm256_sub_ps(vxb, vx);
    const __m256 s1 = _mm256_div_ps(d1, _mm256_sqrt_ps(_mm256_fmadd_ps(d1, d1, veps)));
    const __m256 s2 = _mm256_div_ps(d2, _mm256_sqrt_ps(_mm256_fmadd_ps(d2, d2, veps)));
    const __m256 vg = _mm256_loadu_ps(g + x);
    _mm256_storeu_ps(g + x, _mm256_fmadd_ps(vwgt, _mm256_sub_ps(s1, s2), vg));
  }
  for (; x < w; ++x) scalarAt(x);
}

void u8ToFloatRow(const uint8_t* src, float* dst, int n) {
  const __m256 inv255 = _mm256_set1_ps(1.0f / 255.0f);
  int x = 0;
  for (; x + 8 <= n; x += 8) {
    const __m128i u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + x));
    const __m256i i32 = _mm256_cvtepu8_epi32(u8);
    const __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), inv255);
    _mm256_storeu_ps(dst + x, f);
  }
  for (; x < n; ++x) dst[x] = static_cast<float>(src[x]) * (1.0f / 255.0f);
}

void floatToU8Row(const float* src, uint8_t* dst, int n) {
  const __m256 scale = _mm256_set1_ps(255.0f);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 maxv = _mm256_set1_ps(255.0f);
  int x = 0;
  for (; x + 8 <= n; x += 8) {
    __m256 f = _mm256_mul_ps(_mm256_loadu_ps(src + x), scale);
    f = _mm256_min_ps(maxv, _mm256_max_ps(zero, f));
    const __m256i i32 = _mm256_cvtps_epi32(f);
    // Pack 8x int32 from two 128-bit lanes -> 8x uint8
    const __m128i lo = _mm256_castsi256_si128(i32);
    const __m128i hi = _mm256_extracti128_si256(i32, 1);
    const __m128i i16 = _mm_packus_epi32(lo, hi);
    const __m128i u8 = _mm_packus_epi16(i16, i16);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), u8);
  }
  for (; x < n; ++x) {
    int v = static_cast<int>(std::lround(src[x] * 255.0f));
    dst[x] = static_cast<uint8_t>(std::clamp(v, 0, 255));
  }
}

}  // namespace vu::kern::avx2
