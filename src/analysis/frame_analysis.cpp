#include "analysis/frame_analysis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace vu::analysis {

namespace {
float nthPercentile(std::vector<float>& vals, double frac) {
  if (vals.empty()) return 0.f;
  size_t idx = std::min(vals.size() - 1, static_cast<size_t>(vals.size() * frac));
  std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
  return vals[idx];
}
}  // namespace

void computeEdgeMagnitude(const cv::Mat& y, cv::Mat& mag) {
  CV_Assert(y.type() == CV_32FC1);
  cv::Mat gx, gy;
  cv::Sobel(y, gx, CV_32F, 1, 0, 3, 1.0 / 8.0, 0, cv::BORDER_REPLICATE);
  cv::Sobel(y, gy, CV_32F, 0, 1, 3, 1.0 / 8.0, 0, cv::BORDER_REPLICATE);
  cv::magnitude(gx, gy, mag);
}

float estimateNoiseSigma(const cv::Mat& y) {
  CV_Assert(y.type() == CV_32FC1);
  if (y.cols < 6 || y.rows < 6) return 0.f;

  // Immerkaer (1996) 3x3 zero-mean noise operator N = [1 -2 1; -2 4 -2; 1 -2 1].
  // Standard deviation of (Y * N) on Gaussian noise N(0, sigma^2) is 6 * sigma.
  // Using the MAD of pixels in the smoother half of local gradients avoids
  // overestimating noise on sharp textures/edges.
  cv::Mat grad;
  computeEdgeMagnitude(y, grad);

  const int step = (y.cols * y.rows > 640 * 360) ? 2 : 1;
  std::vector<float> grads;
  grads.reserve(((y.rows - 2) / step) * ((y.cols - 2) / step));
  for (int r = 1; r < y.rows - 1; r += step) {
    const float* gr = grad.ptr<float>(r);
    for (int c = 1; c < y.cols - 1; c += step) grads.push_back(gr[c]);
  }
  const float gradThresh = std::max(0.01f, nthPercentile(grads, 0.55));

  std::vector<float> absResp;
  absResp.reserve(grads.size());
  for (int r = 1; r < y.rows - 1; r += step) {
    const float* p0 = y.ptr<float>(r - 1);
    const float* p1 = y.ptr<float>(r);
    const float* p2 = y.ptr<float>(r + 1);
    const float* gr = grad.ptr<float>(r);
    for (int c = 1; c < y.cols - 1; c += step) {
      if (gr[c] <= gradThresh) {
        const float n = p0[c - 1] - 2.f * p0[c] + p0[c + 1] - 2.f * p1[c - 1] + 4.f * p1[c] -
                        2.f * p1[c + 1] + p2[c - 1] - 2.f * p2[c] + p2[c + 1];
        absResp.push_back(std::abs(n));
      }
    }
  }
  if (absResp.size() < 16) return 0.f;
  const float med = nthPercentile(absResp, 0.5);
  // 1.4826 / 6.0 = 0.2471
  return 0.2471f * med;
}

float estimateBlurScore(const cv::Mat& y) {
  CV_Assert(y.type() == CV_32FC1);
  if (y.cols < 8 || y.rows < 8) return 0.f;
  // Crete et al. (2007) re-blurring blur annoyance metric.
  cv::Mat blurH, blurV;
  cv::blur(y, blurH, cv::Size(9, 1), cv::Point(-1, -1), cv::BORDER_REPLICATE);
  cv::blur(y, blurV, cv::Size(1, 9), cv::Point(-1, -1), cv::BORDER_REPLICATE);

  double sumOrigH = 0, sumDiffH = 0, sumOrigV = 0, sumDiffV = 0;
  for (int r = 1; r < y.rows; ++r) {
    const float* yr = y.ptr<float>(r);
    const float* yp = y.ptr<float>(r - 1);
    const float* bhr = blurH.ptr<float>(r);
    const float* bvr = blurV.ptr<float>(r);
    const float* bvp = blurV.ptr<float>(r - 1);
    for (int c = 1; c < y.cols; ++c) {
      const float dh = std::abs(yr[c] - yr[c - 1]);
      const float dbh = std::abs(bhr[c] - bhr[c - 1]);
      sumOrigH += dh;
      sumDiffH += std::max(0.f, dh - dbh);

      const float dv = std::abs(yr[c] - yp[c]);
      const float dbv = std::abs(bvr[c] - bvp[c]);
      sumOrigV += dv;
      sumDiffV += std::max(0.f, dv - dbv);
    }
  }
  const double bh = sumOrigH > 1e-6 ? 1.0 - (sumDiffH / sumOrigH) : 0.0;
  const double bv = sumOrigV > 1e-6 ? 1.0 - (sumDiffV / sumOrigV) : 0.0;
  return static_cast<float>(std::clamp(std::max(bh, bv), 0.0, 1.0));
}

float estimateBlockiness(const cv::Mat& y) {
  CV_Assert(y.type() == CV_32FC1);
  if (y.cols < 16 || y.rows < 16) return 0.f;

  double bndSum = 0, intSum = 0;
  int64_t bndCnt = 0, intCnt = 0;

  // Horizontal differences across vertical 8px boundaries
  for (int r = 0; r < y.rows; ++r) {
    const float* row = y.ptr<float>(r);
    for (int c = 1; c < y.cols; ++c) {
      const float d = std::abs(row[c] - row[c - 1]);
      if ((c & 7) == 0) {
        bndSum += d;
        ++bndCnt;
      } else {
        intSum += d;
        ++intCnt;
      }
    }
  }
  // Vertical differences across horizontal 8px boundaries
  for (int r = 1; r < y.rows; ++r) {
    const float* r0 = y.ptr<float>(r - 1);
    const float* r1 = y.ptr<float>(r);
    const bool isBnd = ((r & 7) == 0);
    for (int c = 0; c < y.cols; ++c) {
      const float d = std::abs(r1[c] - r0[c]);
      if (isBnd) {
        bndSum += d;
        ++bndCnt;
      } else {
        intSum += d;
        ++intCnt;
      }
    }
  }
  if (bndCnt == 0 || intCnt == 0) return 0.f;
  const double b = bndSum / bndCnt;
  const double i = intSum / intCnt;
  if (b <= i || b < 1e-4) return 0.f;
  return static_cast<float>(std::clamp((b - i) / (b + i + 1e-6), 0.0, 1.0));
}

FrameMetrics analyzeFrame(const cv::Mat& y) {
  CV_Assert(y.type() == CV_32FC1);
  FrameMetrics m;
  cv::Scalar mean, stddev;
  cv::meanStdDev(y, mean, stddev);
  m.meanLuma = static_cast<float>(mean[0]);
  m.lumaStd = static_cast<float>(stddev[0]);

  cv::Mat mag;
  computeEdgeMagnitude(y, mag);

  double gradSum = 0.0;
  int64_t edgeCount = 0;
  double nonEdgeHf = 0.0;
  int64_t nonEdgeCnt = 0;

  cv::Mat smooth;
  cv::GaussianBlur(y, smooth, cv::Size(5, 5), 1.0, 0, cv::BORDER_REPLICATE);

  for (int r = 0; r < y.rows; ++r) {
    const float* mr = mag.ptr<float>(r);
    const float* yr = y.ptr<float>(r);
    const float* sr = smooth.ptr<float>(r);
    for (int c = 0; c < y.cols; ++c) {
      const float g = mr[c];
      gradSum += g;
      if (g > 0.08f) {
        ++edgeCount;
      } else {
        nonEdgeHf += std::abs(yr[c] - sr[c]);
        ++nonEdgeCnt;
      }
    }
  }
  const double total = static_cast<double>(y.rows) * y.cols;
  m.sharpness = static_cast<float>(gradSum / std::max(1.0, total));
  m.edgeDensity = static_cast<float>(edgeCount / std::max(1.0, total));
  m.textureRatio = nonEdgeCnt > 0 ? static_cast<float>(nonEdgeHf / nonEdgeCnt) : 0.f;
  m.noiseSigma = estimateNoiseSigma(y);
  m.blurScore = estimateBlurScore(y);
  m.blockiness = estimateBlockiness(y);
  return m;
}

SceneCutResult detectSceneCut(const cv::Mat& prevY, const cv::Mat& currY, float threshold) {
  CV_Assert(prevY.type() == CV_32FC1 && currY.type() == CV_32FC1 && prevY.size() == currY.size());
  SceneCutResult res;
  constexpr int kBins = 64;
  std::array<double, kBins> h1{}, h2{};

  double madSum = 0.0;
  const int step = (prevY.cols * prevY.rows > 640 * 360) ? 2 : 1;
  int64_t samples = 0;
  for (int r = 0; r < prevY.rows; r += step) {
    const float* p1 = prevY.ptr<float>(r);
    const float* p2 = currY.ptr<float>(r);
    for (int c = 0; c < prevY.cols; c += step) {
      const float v1 = std::clamp(p1[c], 0.f, 0.9999f);
      const float v2 = std::clamp(p2[c], 0.f, 0.9999f);
      ++h1[static_cast<int>(v1 * kBins)];
      ++h2[static_cast<int>(v2 * kBins)];
      madSum += std::abs(v1 - v2);
      ++samples;
    }
  }
  if (samples == 0) return res;
  const double invN = 1.0 / samples;
  double l1 = 0.0;
  for (int b = 0; b < kBins; ++b) l1 += std::abs(h1[b] * invN - h2[b] * invN);

  res.histDistance = static_cast<float>(0.5 * l1);  // in [0, 1]
  res.mad = static_cast<float>(madSum * invN);
  // Combine structural change (MAD) and global appearance change (histogram L1)
  res.score = 0.65f * res.histDistance + 0.35f * res.mad;
  res.isCut = res.score >= threshold;
  return res;
}

}  // namespace vu::analysis
