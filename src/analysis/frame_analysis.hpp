#pragma once
#include <vector>

#include <opencv2/core.hpp>

namespace vu::analysis {

struct FrameMetrics {
  float meanLuma = 0.f;        // [0, 1]
  float lumaStd = 0.f;         // contrast / spread
  float noiseSigma = 0.f;      // estimated additive white noise std in [0, 1] (Immerkaer MAD)
  float blurScore = 0.f;       // [0, 1] Crete re-blur metric (higher = blurrier)
  float sharpness = 0.f;       // mean gradient energy
  float edgeDensity = 0.f;     // fraction of pixels with Sobel |grad| > 0.08
  float blockiness = 0.f;      // [0, 1] excess 8x8 boundary discontinuity over interior
  float textureRatio = 0.f;    // high-frequency energy in non-edge regions
};

struct SceneCutResult {
  bool isCut = false;
  float score = 0.f;           // combined normalized cut score in [0, 1]
  float histDistance = 0.f;    // L1 distance of 64-bin luma histograms / 2
  float mad = 0.f;             // mean absolute difference in [0, 1]
};

// All functions expect single-channel CV_32FC1 luma in [0, 1].
float estimateNoiseSigma(const cv::Mat& y);
float estimateBlurScore(const cv::Mat& y);
float estimateBlockiness(const cv::Mat& y);
FrameMetrics analyzeFrame(const cv::Mat& y);

// Compares consecutive luma frames (CV_32FC1 in [0, 1]) to detect scene cuts.
SceneCutResult detectSceneCut(const cv::Mat& prevY, const cv::Mat& currY, float threshold);

// Computes edge strength map (Sobel gradient magnitude in [0, ~1]).
void computeEdgeMagnitude(const cv::Mat& y, cv::Mat& mag);

}  // namespace vu::analysis
