#pragma once
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "core/config.hpp"

namespace vu::recon {

struct MfsrInputFrame {
  cv::Mat yFloat;    // CV_32FC1 in [0, 1] (pre-enhanced LR luma)
  cv::Mat yU8;       // CV_8UC1 (for DIS optical flow)
  int segment = 0;   // scene segment ID; neighbors with segment != ref.segment are ignored
  int64_t index = 0;
};

struct MfsrStats {
  int neighborsUsed = 0;
  float meanNeighborConfidence = 0.f;
  float meanOccludedFraction = 0.f;
  double initialRefRmse = 0.0;
  double finalRefRmse = 0.0;
  bool consistencyFallback = false;
};

// Reconstructs a single HR reference luma frame from a temporal window of LR frames
// (typically 5 frames: I(t-2)..I(t+2)), with `refIdx` pointing to I(t).
// Implements the 10-stage classical MFSR pipeline: frame selection -> DIS optical flow ->
// sub-pixel alignment -> FB occlusion detection -> motion confidence -> confidence-weighted
// fusion -> Huber+BTV multi-frame IBP -> observation consistency check -> optional temporal refinement.
MfsrStats reconstructMultiFrame(const std::vector<MfsrInputFrame>& window, int refIdx, cv::Mat& hrOut,
                                const ProcessingConfig& cfg, cv::DISOpticalFlow* flowEst = nullptr,
                                const cv::Mat* prevHrOut = nullptr, const cv::Mat* prevYFloat = nullptr,
                                const cv::Mat* prevYU8 = nullptr);

}  // namespace vu::recon
