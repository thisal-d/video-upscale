#include "compute/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "compute/backend.hpp"
#ifdef VU_HAVE_AVX2_KERNELS
#include "compute/kernels_avx2.hpp"
#endif

namespace vu::kern {

std::vector<float> gaussianKernel(float sigma, int radius) {
  if (sigma <= 0.f) return {1.f};
  if (radius < 0) radius = std::max(1, static_cast<int>(std::ceil(3.f * sigma)));
  std::vector<float> k(2 * radius + 1);
  double sum = 0;
  for (int i = -radius; i <= radius; ++i) {
    k[i + radius] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
    sum += k[i + radius];
  }
  for (float& v : k) v = static_cast<float>(v / sum);
  return k;
}

namespace {

void sepFilterImpl(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& k, cv::Mat& tmp, bool useAvx2,
                   int threads) {
  CV_Assert(src.type() == CV_32FC1 && src.data != dst.data);
  const int w = src.cols, h = src.rows, r = static_cast<int>(k.size()) / 2;
  tmp.create(h, w, CV_32FC1);
  dst.create(h, w, CV_32FC1);
  if (r == 0) {
    src.copyTo(dst);
    return;
  }
  const float* kc = k.data() + r;

#pragma omp parallel num_threads(threads) if (threads > 1 && h >= 64)
  {
    std::vector<float> line(w + 2 * r);
#pragma omp for schedule(static)
    for (int y = 0; y < h; ++y) {
      const float* s = src.ptr<float>(y);
      for (int x = 0; x < r; ++x) line[x] = s[0];
      std::copy(s, s + w, line.begin() + r);
      for (int x = 0; x < r; ++x) line[w + r + x] = s[w - 1];
      float* t = tmp.ptr<float>(y);
      const float* L = line.data() + r;
#ifdef VU_HAVE_AVX2_KERNELS
      if (useAvx2) {
        avx2::sepFilterHorizRow(L, t, w, kc, r);
        continue;
      }
#else
      (void)useAvx2;
#endif
      for (int x = 0; x < w; ++x) {
        float acc = kc[0] * L[x];
        for (int i = 1; i <= r; ++i) acc += kc[i] * (L[x - i] + L[x + i]);
        t[x] = acc;
      }
    }
  }

#pragma omp parallel num_threads(threads) if (threads > 1 && h >= 64)
  {
    std::vector<const float*> upRows(r), downRows(r);
#pragma omp for schedule(static)
    for (int y = 0; y < h; ++y) {
      float* d = dst.ptr<float>(y);
      const float* c = tmp.ptr<float>(y);
      for (int i = 1; i <= r; ++i) {
        upRows[i - 1] = tmp.ptr<float>(std::max(0, y - i));
        downRows[i - 1] = tmp.ptr<float>(std::min(h - 1, y + i));
      }
#ifdef VU_HAVE_AVX2_KERNELS
      if (useAvx2) {
        avx2::sepFilterVertRow(d, c, upRows.data(), downRows.data(), w, kc, r);
        continue;
      }
#endif
      for (int x = 0; x < w; ++x) d[x] = kc[0] * c[x];
      for (int i = 1; i <= r; ++i) {
        const float* a = upRows[i - 1];
        const float* b = downRows[i - 1];
        const float kv = kc[i];
        for (int x = 0; x < w; ++x) d[x] += kv * (a[x] + b[x]);
      }
    }
  }
}

void addBtvGradientImpl(const cv::Mat& X, cv::Mat& grad, int radius, float alpha, float lambda, bool useAvx2,
                        int threads) {
  CV_Assert(X.type() == CV_32FC1 && grad.size() == X.size());
  const int w = X.cols, h = X.rows;
  constexpr float eps = 1e-6f;
  auto ssign = [](float v) { return v / std::sqrt(v * v + eps); };

  struct Shift {
    int dx, dy;
    float wgt;
  };
  std::vector<Shift> shifts;
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) continue;
      shifts.push_back({dx, dy, lambda * std::pow(alpha, static_cast<float>(std::abs(dx) + std::abs(dy)))});
    }
  }

#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && h >= 64)
  for (int y = 0; y < h; ++y) {
    const float* xr = X.ptr<float>(y);
    float* g = grad.ptr<float>(y);
    for (const auto& s : shifts) {
      const float* xf = X.ptr<float>(std::clamp(y + s.dy, 0, h - 1));
      const float* xb = X.ptr<float>(std::clamp(y - s.dy, 0, h - 1));
#ifdef VU_HAVE_AVX2_KERNELS
      if (useAvx2) {
        avx2::btvAccumRow(xr, xf, xb, g, w, s.dx, s.wgt);
        continue;
      }
#else
      (void)useAvx2;
#endif
      for (int x = 0; x < w; ++x) {
        const int xp = std::clamp(x + s.dx, 0, w - 1), xm = std::clamp(x - s.dx, 0, w - 1);
        g[x] += s.wgt * (ssign(xr[x] - xf[xp]) - ssign(xb[xm] - xr[x]));
      }
    }
  }
}

}  // namespace

void sepFilter(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& k, cv::Mat& tmp) {
  sepFilterImpl(src, dst, k, tmp, compute::avx2KernelsAvailable(), 1);
}

void sepFilterParallel(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& k, cv::Mat& tmp) {
  sepFilterImpl(src, dst, k, tmp, compute::avx2KernelsAvailable(), compute::maxThreads());
}

void sepFilterScalar(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& k, cv::Mat& tmp) {
  sepFilterImpl(src, dst, k, tmp, false, 1);
}

void addBtvGradient(const cv::Mat& X, cv::Mat& grad, int radius, float alpha, float lambda) {
  addBtvGradientImpl(X, grad, radius, alpha, lambda, compute::avx2KernelsAvailable(), 1);
}

void addBtvGradientParallel(const cv::Mat& X, cv::Mat& grad, int radius, float alpha, float lambda) {
  addBtvGradientImpl(X, grad, radius, alpha, lambda, compute::avx2KernelsAvailable(), compute::maxThreads());
}

void addBtvGradientScalar(const cv::Mat& X, cv::Mat& grad, int radius, float alpha, float lambda) {
  addBtvGradientImpl(X, grad, radius, alpha, lambda, false, 1);
}

void u8ToFloat(const cv::Mat& src, cv::Mat& dst) {
  CV_Assert(src.type() == CV_8UC1);
#ifdef VU_HAVE_AVX2_KERNELS
  if (compute::avx2KernelsAvailable()) {
    dst.create(src.size(), CV_32FC1);
    for (int y = 0; y < src.rows; ++y) avx2::u8ToFloatRow(src.ptr<uint8_t>(y), dst.ptr<float>(y), src.cols);
    return;
  }
#endif
  src.convertTo(dst, CV_32F, 1.0 / 255.0);
}

void floatToU8(const cv::Mat& src, cv::Mat& dst) {
  CV_Assert(src.type() == CV_32FC1);
#ifdef VU_HAVE_AVX2_KERNELS
  if (compute::avx2KernelsAvailable()) {
    dst.create(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y) avx2::floatToU8Row(src.ptr<float>(y), dst.ptr<uint8_t>(y), src.cols);
    return;
  }
#endif
  src.convertTo(dst, CV_8U, 255.0);
}

}  // namespace vu::kern
