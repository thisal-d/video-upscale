#pragma once
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/config.hpp"

namespace vu::quality {

// Computes PSNR (dB) on CV_32FC1 [0, 1] images (border pixels can be excluded).
double computePsnr(const cv::Mat& ref, const cv::Mat& test, int border = 0);

// Computes Wang et al. (2004) SSIM on CV_32FC1 [0, 1] images (11x11 Gaussian sigma=1.5).
double computeSsim(const cv::Mat& ref, const cv::Mat& test, int border = 0);

struct VideoQualityStats {
  int64_t framesCompared = 0;
  double meanPsnr = 0.0;
  double minPsnr = 0.0;
  double meanSsim = 0.0;
  double minSsim = 0.0;
  double meanTemporalWarpedRmse = 0.0;  // motion-compensated frame-to-frame RMSE on output
  double refTemporalWarpedRmse = 0.0;   // motion-compensated frame-to-frame RMSE on reference
  double temporalConsistencyError = 0.0;// |out_t - out_{t-1}| vs |ref_t - ref_{t-1}| RMSE
};

// Compares a reconstructed video file against its ground-truth reference video file.
VideoQualityStats evaluateVideoPair(const std::string& referencePath, const std::string& testPath,
                                    int border = 8, int maxFrames = 0);

}  // namespace vu::quality
