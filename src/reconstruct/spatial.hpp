#pragma once
#include <opencv2/core.hpp>

#include "core/config.hpp"

namespace vu::recon {

// Classical interpolation of a CV_32FC1 plane to dstSize (pixel-centre aligned:
// x_src = (x_dst + 0.5) * src/dst - 0.5). Supported: Nearest, Bilinear, Bicubic,
// Lanczos, EdgeAware. Called from serial code (may use OpenCV / OpenMP internally).
void upscalePlane(const cv::Mat& src, cv::Mat& dst, cv::Size dstSize, Upscaler method,
                  float edgeBlend = 1.0f);

// Edge-directed steering-kernel interpolation (see ALGORITHMS.md §3):
// bicubic base, replaced near coherent edges by an anisotropic Nadaraya-Watson
// estimate whose kernel is elongated along the local edge tangent (structure tensor).
void edgeDirectedUpscale(const cv::Mat& src, cv::Mat& dst, cv::Size dstSize, float blend);

// Applies `method` as successive 2x stages until reaching `scale` (progressive mode).
void upscaleProgressive(const cv::Mat& src, cv::Mat& dst, int scale, Upscaler method, float edgeBlend = 1.0f);

// Tyranox Algorithm family:
// 1. TyranoxCrazy ("Tyranox - crazy"): pure 1 -> s x s exact pixel replication
//    (1 -> 4 at 2x, 1 -> 9 at 3x, 1 -> 16 at 4x, 1 -> 64 at 8x).
// 2. TyranoxEdge: Scale2x/Scale3x-style diagonal anti-staircasing pixel replication.
// 3. TyranoxSmooth: pixel replication + edge-preserving bilateral anti-aliasing.
void tyranoxCrazyUpscale(const cv::Mat& src, cv::Mat& dst, cv::Size dstSize);
void tyranoxEdgeUpscale(const cv::Mat& src, cv::Mat& dst, cv::Size dstSize);
void tyranoxSmoothUpscale(const cv::Mat& src, cv::Mat& dst, cv::Size dstSize);

}  // namespace vu::recon
