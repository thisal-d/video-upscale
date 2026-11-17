#include "enhance/spatial_enhance.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "compute/backend.hpp"

namespace vu::enhance {

namespace {
inline float smoothstep(float e0, float e1, float x) {
  float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}
}  // namespace

bool adaptiveDenoise(const cv::Mat& src, cv::Mat& dst, float noiseSigma, float strength) {
  CV_Assert(src.type() == CV_32FC1);
  if (strength <= 0.f || noiseSigma <= 1e-4f) {
    if (src.data != dst.data) src.copyTo(dst);
    return false;
  }
  const float sigmaColor = std::clamp(2.8f * noiseSigma * strength, 0.008f, 0.18f);
  const float sigmaSpace = std::clamp(1.2f + 12.f * noiseSigma * strength, 1.0f, 2.5f);
  const int d = (sigmaSpace > 1.7f) ? 7 : 5;
  cv::Mat tmp;
  cv::bilateralFilter(src, tmp, d, sigmaColor, sigmaSpace, cv::BORDER_REPLICATE);
  // Soft blend in very flat vs high-detail texture so fine grain/texture isn't plasticized
  cv::Mat mag;
  analysis::computeEdgeMagnitude(src, mag);
  dst.create(src.size(), CV_32FC1);
  const float blendCap = std::clamp(0.45f + 12.f * noiseSigma * strength, 0.45f, 0.95f);
  for (int r = 0; r < src.rows; ++r) {
    const float* s = src.ptr<float>(r);
    const float* t = tmp.ptr<float>(r);
    const float* m = mag.ptr<float>(r);
    float* out = dst.ptr<float>(r);
    for (int c = 0; c < src.cols; ++c) {
      // Preserve very crisp micro-edges slightly while cleaning flat/moderate regions
      const float w = blendCap * (1.f - 0.35f * smoothstep(0.05f, 0.25f, m[c]));
      out[c] = s[c] + w * (t[c] - s[c]);
    }
  }
  return true;
}

bool deblockPlane(const cv::Mat& src, cv::Mat& dst, float strength) {
  CV_Assert(src.type() == CV_32FC1);
  if (strength <= 0.f || src.cols < 16 || src.rows < 16) {
    if (src.data != dst.data) src.copyTo(dst);
    return false;
  }
  cv::Mat work = src.clone();
  const float beta = std::clamp(0.045f * strength, 0.01f, 0.14f);  // edge preservation threshold
  const float alpha = std::clamp(0.08f * strength, 0.02f, 0.20f);  // boundary step cap
  const float interiorThresh = beta * 0.45f;

  // 1. Vertical 8-pixel boundaries (x = 8, 16, ...)
  for (int y = 0; y < work.rows; ++y) {
    float* row = work.ptr<float>(y);
    for (int x = 8; x + 1 < work.cols; x += 8) {
      const float p1 = row[x - 2], p0 = row[x - 1], q0 = row[x], q1 = row[x + 1];
      const float diff = q0 - p0;
      const float adiff = std::abs(diff);
      if (adiff < 1e-4f || adiff >= alpha) continue;
      if (std::abs(p1 - p0) >= interiorThresh || std::abs(q1 - q0) >= interiorThresh) continue;
      // 4-tap smoothing step proportional to (q0 - p0) while damping interior shoulders
      const float delta = std::clamp((4.f * (q0 - p0) + (p1 - q1)) / 8.f, -0.5f * adiff, 0.5f * adiff);
      const float d1 = delta * 0.35f;
      row[x - 1] = std::clamp(p0 + delta, 0.f, 1.f);
      row[x] = std::clamp(q0 - delta, 0.f, 1.f);
      row[x - 2] = std::clamp(p1 + d1, 0.f, 1.f);
      row[x + 1] = std::clamp(q1 - d1, 0.f, 1.f);
    }
  }

  // 2. Horizontal 8-pixel boundaries (y = 8, 16, ...)
  for (int y = 8; y + 1 < work.rows; y += 8) {
    float* rp1 = work.ptr<float>(y - 2);
    float* rp0 = work.ptr<float>(y - 1);
    float* rq0 = work.ptr<float>(y);
    float* rq1 = work.ptr<float>(y + 1);
    for (int x = 0; x < work.cols; ++x) {
      const float p1 = rp1[x], p0 = rp0[x], q0 = rq0[x], q1 = rq1[x];
      const float diff = q0 - p0;
      const float adiff = std::abs(diff);
      if (adiff < 1e-4f || adiff >= alpha) continue;
      if (std::abs(p1 - p0) >= interiorThresh || std::abs(q1 - q0) >= interiorThresh) continue;
      const float delta = std::clamp((4.f * (q0 - p0) + (p1 - q1)) / 8.f, -0.5f * adiff, 0.5f * adiff);
      const float d1 = delta * 0.35f;
      rp0[x] = std::clamp(p0 + delta, 0.f, 1.f);
      rq0[x] = std::clamp(q0 - delta, 0.f, 1.f);
      rp1[x] = std::clamp(p1 + d1, 0.f, 1.f);
      rq1[x] = std::clamp(q1 - d1, 0.f, 1.f);
    }
  }
  dst = std::move(work);
  return true;
}

void adaptiveSharpen(const cv::Mat& src, cv::Mat& dst, float amount, float radius, bool haloSuppression,
                     float haloOvershoot) {
  CV_Assert(src.type() == CV_32FC1);
  if (amount <= 0.f) {
    if (src.data != dst.data) src.copyTo(dst);
    return;
  }

  cv::Mat blur, mag;
  cv::GaussianBlur(src, blur, cv::Size(0, 0), std::max(0.4f, radius), 0, cv::BORDER_REPLICATE);
  analysis::computeEdgeMagnitude(src, mag);

  cv::Mat minLocal, maxLocal;
  if (haloSuppression) {
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(src, minLocal, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    cv::dilate(src, maxLocal, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
  }

  dst.create(src.size(), CV_32FC1);
  const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1)
  for (int r = 0; r < src.rows; ++r) {
    const float* s = src.ptr<float>(r);
    const float* b = blur.ptr<float>(r);
    const float* m = mag.ptr<float>(r);
    const float* mn = haloSuppression ? minLocal.ptr<float>(r) : nullptr;
    const float* mx = haloSuppression ? maxLocal.ptr<float>(r) : nullptr;
    float* out = dst.ptr<float>(r);
    for (int c = 0; c < src.cols; ++c) {
      const float detail = s[c] - b[c];
      // Coring + contrast masking: suppress sharpening in flat noisy zones and very high-contrast step edges
      const float noiseGate = smoothstep(0.008f, 0.035f, m[c]);
      const float edgeTaper = 1.f - 0.55f * smoothstep(0.18f, 0.45f, m[c]);
      const float gain = amount * noiseGate * edgeTaper;
      float v = s[c] + gain * detail;
      if (haloSuppression) {
        v = std::clamp(v, mn[c] - haloOvershoot, mx[c] + haloOvershoot);
      }
      out[c] = std::clamp(v, 0.f, 1.f);
    }
  }
}

EnhanceReport preprocessLuma(const cv::Mat& srcY, cv::Mat& dstY, const EnhanceParams& cfg,
                             const analysis::FrameMetrics& metrics) {
  EnhanceReport rep;
  rep.estimatedNoise = metrics.noiseSigma;
  rep.estimatedBlockiness = metrics.blockiness;

  cv::Mat cur = srcY;
  // Auto thresholds calibrated on compressed & noisy sequences:
  const bool doDeblock =
      (cfg.deblock == Toggle::On) || (cfg.deblock == Toggle::Auto && metrics.blockiness > 0.12f);
  if (doDeblock) {
    cv::Mat deb;
    const float str = cfg.deblockStrength * (cfg.deblock == Toggle::Auto ? std::clamp(metrics.blockiness * 4.f, 0.5f, 1.5f) : 1.0f);
    if (deblockPlane(cur, deb, str)) {
      cur = std::move(deb);
      rep.deblocked = true;
    }
  }

  const bool doDenoise =
      (cfg.denoise == Toggle::On) || (cfg.denoise == Toggle::Auto && metrics.noiseSigma > 0.009f);
  if (doDenoise) {
    cv::Mat den;
    const float sigma = (cfg.denoise == Toggle::On && metrics.noiseSigma < 0.008f) ? 0.012f : metrics.noiseSigma;
    if (adaptiveDenoise(cur, den, sigma, cfg.denoiseStrength)) {
      cur = std::move(den);
      rep.denoised = true;
    }
  }

  if (cur.data == srcY.data) srcY.copyTo(dstY);
  else dstY = std::move(cur);
  return rep;
}

}  // namespace vu::enhance
