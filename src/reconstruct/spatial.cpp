#include "reconstruct/spatial.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "compute/backend.hpp"
#include "core/error.hpp"

namespace vu::recon {

namespace {
int cvInterp(Upscaler m) {
  switch (m) {
    case Upscaler::Nearest: return cv::INTER_NEAREST_EXACT;
    case Upscaler::Bilinear: return cv::INTER_LINEAR;
    case Upscaler::Bicubic: return cv::INTER_CUBIC;
    case Upscaler::Lanczos: return cv::INTER_LANCZOS4;
    default: return cv::INTER_CUBIC;
  }
}

inline float smoothstep(float e0, float e1, float x) {
  float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}
}  // namespace

void upscalePlane(const cv::Mat& src, cv::Mat& dst, cv::Size dstSize, Upscaler method, float edgeBlend) {
  CV_Assert(src.type() == CV_32FC1);
  switch (method) {
    case Upscaler::Nearest:
    case Upscaler::Bilinear:
    case Upscaler::Bicubic:
    case Upscaler::Lanczos:
      cv::resize(src, dst, dstSize, 0, 0, cvInterp(method));
      return;
    case Upscaler::EdgeAware:
      edgeDirectedUpscale(src, dst, dstSize, edgeBlend);
      return;
    default:
      fail(ErrorCode::InvalidArgument, std::string("upscalePlane: not a spatial method: ") + toString(method));
  }
}

void edgeDirectedUpscale(const cv::Mat& src, cv::Mat& dst, cv::Size dstSize, float blend) {
  CV_Assert(src.type() == CV_32FC1);
  cv::Mat base;
  cv::resize(src, base, dstSize, 0, 0, cv::INTER_CUBIC);
  if (blend <= 0.f || src.cols < 4 || src.rows < 4) {
    dst = base;
    return;
  }
  // Structure tensor of the low-resolution image.
  cv::Mat gx, gy;
  cv::Sobel(src, gx, CV_32F, 1, 0, 3, 1.0 / 8.0, 0, cv::BORDER_REPLICATE);
  cv::Sobel(src, gy, CV_32F, 0, 1, 3, 1.0 / 8.0, 0, cv::BORDER_REPLICATE);
  cv::Mat jxx = gx.mul(gx), jxy = gx.mul(gy), jyy = gy.mul(gy);
  cv::GaussianBlur(jxx, jxx, cv::Size(0, 0), 1.0, 0, cv::BORDER_REPLICATE);
  cv::GaussianBlur(jxy, jxy, cv::Size(0, 0), 1.0, 0, cv::BORDER_REPLICATE);
  cv::GaussianBlur(jyy, jyy, cv::Size(0, 0), 1.0, 0, cv::BORDER_REPLICATE);

  const int lw = src.cols, lh = src.rows;
  // Per-LR-pixel steering parameters: unit normal (nx, ny), gate g, tangential sigma.
  cv::Mat params(lh, lw, CV_32FC4);
  const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1)
  for (int y = 0; y < lh; ++y) {
    const float* a = jxx.ptr<float>(y);
    const float* b = jxy.ptr<float>(y);
    const float* c = jyy.ptr<float>(y);
    cv::Vec4f* p = params.ptr<cv::Vec4f>(y);
    for (int x = 0; x < lw; ++x) {
      const float tr = a[x] + c[x];
      const float diff = 0.5f * (a[x] - c[x]);
      const float root = std::sqrt(diff * diff + b[x] * b[x]);
      const float l1 = 0.5f * tr + root, l2 = 0.5f * tr - root;
      const float coh = (l1 - l2) / (l1 + l2 + 1e-8f);
      const float phi = 0.5f * std::atan2(2.f * b[x], a[x] - c[x]);  // gradient (normal) direction
      const float strength = std::sqrt(std::max(l1, 0.f));
      const float g = coh * coh * smoothstep(0.035f, 0.10f, strength);
      p[x] = cv::Vec4f(std::cos(phi), std::sin(phi), g, 0.40f + 1.10f * coh);
    }
  }

  dst.create(dstSize, CV_32FC1);
  const float sx = static_cast<float>(lw) / dstSize.width;
  const float sy = static_cast<float>(lh) / dstSize.height;
  constexpr float sigmaN = 0.32f;
  const float invN = 1.f / (2.f * sigmaN * sigmaN);
#pragma omp parallel for schedule(dynamic, 8) num_threads(nthr) if (nthr > 1)
  for (int Y = 0; Y < dstSize.height; ++Y) {
    const float* brow = base.ptr<float>(Y);
    float* out = dst.ptr<float>(Y);
    const float y = (Y + 0.5f) * sy - 0.5f;
    const int yn = std::clamp(static_cast<int>(std::lround(y)), 0, lh - 1);
    const int y0 = static_cast<int>(std::floor(y));
    for (int X = 0; X < dstSize.width; ++X) {
      const float x = (X + 0.5f) * sx - 0.5f;
      const int xn = std::clamp(static_cast<int>(std::lround(x)), 0, lw - 1);
      const cv::Vec4f& pr = params.at<cv::Vec4f>(yn, xn);
      const float g = pr[2] * blend;
      if (g < 0.02f) {
        out[X] = brow[X];
        continue;
      }
      const float nx = pr[0], ny = pr[1];
      const float invT = 1.f / (2.f * pr[3] * pr[3]);
      const int x0 = static_cast<int>(std::floor(x));
      float sw = 0.f, sv = 0.f;
      for (int j = y0 - 1; j <= y0 + 2; ++j) {
        const float* srow = src.ptr<float>(std::clamp(j, 0, lh - 1));
        const float dy = j - y;
        for (int i = x0 - 1; i <= x0 + 2; ++i) {
          const float dx = i - x;
          const float dn = dx * nx + dy * ny;
          const float dt = -dx * ny + dy * nx;
          const float w = std::exp(-(dn * dn * invN + dt * dt * invT));
          sw += w;
          sv += w * srow[std::clamp(i, 0, lw - 1)];
        }
      }
      const float d = sv / std::max(sw, 1e-8f);
      out[X] = brow[X] + g * (d - brow[X]);
    }
  }
}

void upscaleProgressive(const cv::Mat& src, cv::Mat& dst, int scale, Upscaler method, float edgeBlend) {
  cv::Mat cur = src;
  int s = 1;
  while (s < scale) {
    cv::Mat next;
    upscalePlane(cur, next, cv::Size(cur.cols * 2, cur.rows * 2), method, edgeBlend);
    cur = next;
    s *= 2;
  }
  dst = cur;
}

}  // namespace vu::recon
