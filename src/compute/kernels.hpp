#pragma once
// Low-level float image kernels used by the reconstruction code.
// All functions operate on CV_32FC1 matrices and are *serial* unless the name says
// otherwise, so they can safely be called from inside OpenMP tile loops (no nested
// parallelism, no OpenCV thread-pool usage).
#include <vector>

#include <opencv2/core.hpp>

namespace vu::kern {

// Normalized 1-D Gaussian kernel; radius = ceil(3*sigma) if radius < 0.
std::vector<float> gaussianKernel(float sigma, int radius = -1);

// Separable symmetric convolution with replicated borders. dst may not alias src.
void sepFilter(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& k, cv::Mat& tmp);
// Row-parallel version (uses OpenMP; only call from serial outer code).
void sepFilterParallel(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& k, cv::Mat& tmp);

// Bilinear sample at continuous pixel-index coordinates (x, y) with clamped borders.
inline float sampleBilinear(const cv::Mat& img, float x, float y) {
  const int w = img.cols, h = img.rows;
  x = x < 0.f ? 0.f : (x > w - 1 ? static_cast<float>(w - 1) : x);
  y = y < 0.f ? 0.f : (y > h - 1 ? static_cast<float>(h - 1) : y);
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
  const float fx = x - x0, fy = y - y0;
  const float* r0 = img.ptr<float>(y0);
  const float* r1 = img.ptr<float>(y1);
  const float a = r0[x0] + fx * (r0[x1] - r0[x0]);
  const float b = r1[x0] + fx * (r1[x1] - r1[x0]);
  return a + fy * (b - a);
}

// Transpose of sampleBilinear: distributes value v onto the 4 neighbours of (x, y).
// Points outside [0,w-1]x[0,h-1] are ignored (returns false).
inline bool splatBilinear(cv::Mat& img, float x, float y, float v) {
  const int w = img.cols, h = img.rows;
  if (x < 0.f || y < 0.f || x > w - 1 || y > h - 1) return false;
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
  const float fx = x - x0, fy = y - y0;
  float* r0 = img.ptr<float>(y0);
  float* r1 = img.ptr<float>(y1);
  r0[x0] += v * (1 - fx) * (1 - fy);
  r0[x1] += v * fx * (1 - fy);
  r1[x0] += v * (1 - fx) * fy;
  r1[x1] += v * fx * fy;
  return true;
}

// Gradient of the bilateral total variation (BTV) regularizer (Farsiu et al. 2004):
//   R(X) = sum_{l,m in [-P,P]} alpha^{|l|+|m|} || X - S_x^l S_y^m X ||_1
// accumulated (scaled by lambda) into grad. Uses a smooth sign (x/sqrt(x^2+eps)).
void addBtvGradient(const cv::Mat& X, cv::Mat& grad, int radius, float alpha, float lambda);
void addBtvGradientParallel(const cv::Mat& X, cv::Mat& grad, int radius, float alpha, float lambda);

// Converts 8-bit plane to float in [0,1] and back (rounded, saturated).
void u8ToFloat(const cv::Mat& src, cv::Mat& dst);
void floatToU8(const cv::Mat& src, cv::Mat& dst);

// Explicit scalar variants (for testing/benchmarking against SIMD dispatch).
void sepFilterScalar(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& k, cv::Mat& tmp);
void addBtvGradientScalar(const cv::Mat& X, cv::Mat& grad, int radius, float alpha, float lambda);

}  // namespace vu::kern
