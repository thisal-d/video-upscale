#pragma once
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "core/config.hpp"

namespace vu::motion {

struct MotionStats {
  float meanConfidence = 0.f;
  float occludedFraction = 0.f;
  float meanFbError = 0.f;
  float meanMotionMag = 0.f;
};

struct MotionResult {
  cv::Mat flowFwd;       // CV_32FC2: ref -> nbr (LR px)
  cv::Mat flowBwd;       // CV_32FC2: nbr -> ref (LR px)
  cv::Mat fbError;       // CV_32FC1: ||u(p) + v(p + u(p))||_2 (LR px)
  cv::Mat occlusionMask; // CV_8UC1: 255 = valid/visible, 0 = occluded or OOB
  cv::Mat confidence;    // CV_32FC1: [0, 1] observation confidence
  cv::Mat alignedLr;     // CV_32FC1: nbr warped into ref coordinates
  MotionStats stats;
};

cv::Ptr<cv::DISOpticalFlow> createFlowEstimator(const std::string& preset);

// Estimates bidirectional optical flow between ref and nbr, computes forward-backward
// consistency, occlusion mask, photometric & geometric confidence, and aligns nbr -> ref.
// refY, nbrY: CV_32FC1 in [0, 1]. refU8, nbrU8: CV_8UC1 (same size).
MotionResult estimateMotion(const cv::Mat& refY, const cv::Mat& nbrY, const cv::Mat& refU8,
                            const cv::Mat& nbrU8, const TemporalParams& params,
                            cv::DISOpticalFlow* flowEst = nullptr);

// Warps `src` (CV_32FC1) using `flow` (CV_32FC2): dst(p) = src(p + flow(p)).
void warpWithFlow(const cv::Mat& src, const cv::Mat& flow, cv::Mat& dst, int interp = cv::INTER_CUBIC);

// Resizes an LR flow field to `hrSize` and scales vector magnitudes by `scale`.
void upscaleFlow(const cv::Mat& lrFlow, cv::Mat& hrFlow, cv::Size hrSize, int scale);

}  // namespace vu::motion
