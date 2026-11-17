#pragma once
#include <opencv2/core.hpp>

#include "core/config.hpp"

namespace vu::recon {

struct IbpStats {
  double initialRmse = 0.0;
  double finalRmse = 0.0;
  int tilesProcessed = 0;
};

// Single-frame Huber + BTV Iterative Back-Projection (see ALGORITHMS.md).
// Initializes from edge-directed interpolation, then solves:
//   min_X sum_i rho_huber(D_s(H * X)_i - Y_i) + lambda * BTV(X)
// using tiled OpenMP execution over CV_32FC1 [0, 1] images.
IbpStats iterativeBackProjection(const cv::Mat& lr, cv::Mat& hr, int scale,
                                 const ReconstructionParams& params, int tileSize = 192);

// Progressive 2x stages of IBP for 4x/8x scaling (for direct vs progressive comparison).
IbpStats iterativeBackProjectionProgressive(const cv::Mat& lr, cv::Mat& hr, int scale,
                                            const ReconstructionParams& params, int tileSize = 192);

// Exposed for MFSR / observation consistency checks:
// Forward observation operator Y_hat = D_s(H * X) using exact s x s box-average.
void downsampleBoxPsf(const cv::Mat& hr, cv::Mat& lrOut, int scale, const std::vector<float>& psfKernel,
                      cv::Mat& blurScratch, cv::Mat& tmpScratch);

}  // namespace vu::recon
