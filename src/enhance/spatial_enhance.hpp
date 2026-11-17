#pragma once
#include <opencv2/core.hpp>

#include "analysis/frame_analysis.hpp"
#include "core/config.hpp"

namespace vu::enhance {

struct EnhanceReport {
  bool deblocked = false;
  bool denoised = false;
  bool sharpened = false;
  float estimatedNoise = 0.f;
  float estimatedBlockiness = 0.f;
};

// Adaptive edge-preserving bilateral denoising on CV_32FC1 [0, 1].
// Returns true if filtering was applied.
bool adaptiveDenoise(const cv::Mat& src, cv::Mat& dst, float noiseSigma, float strength = 1.0f);

// Adaptive 8x8 block-boundary deblocking filter on CV_32FC1 [0, 1].
// Preserves strong real edges while smoothing DCT quantization steps across 8-px grid lines.
bool deblockPlane(const cv::Mat& src, cv::Mat& dst, float strength = 1.0f);

// Edge-masked unsharp detail enhancement with optional local min/max halo suppression.
void adaptiveSharpen(const cv::Mat& src, cv::Mat& dst, float amount, float radius, bool haloSuppression,
                     float haloOvershoot);

// Applies configured LR pre-enhancement (deblock -> denoise) to a luma plane.
EnhanceReport preprocessLuma(const cv::Mat& srcY, cv::Mat& dstY, const EnhanceParams& cfg,
                             const analysis::FrameMetrics& metrics);

}  // namespace vu::enhance
