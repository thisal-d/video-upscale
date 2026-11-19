#include "reconstruct/ibp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "compute/backend.hpp"
#include "compute/kernels.hpp"
#include "core/error.hpp"
#include "reconstruct/spatial.hpp"

namespace vu::recon {

void downsampleBoxPsf(const cv::Mat& hr, cv::Mat& lrOut, int scale, const std::vector<float>& psfKernel,
                      cv::Mat& blurScratch, cv::Mat& tmpScratch) {
  CV_Assert(hr.type() == CV_32FC1);
  kern::sepFilter(hr, blurScratch, psfKernel, tmpScratch);
  const int lw = hr.cols / scale, lh = hr.rows / scale;
  lrOut.create(lh, lw, CV_32FC1);
  const float invS2 = 1.0f / static_cast<float>(scale * scale);
  for (int y = 0; y < lh; ++y) {
    float* dst = lrOut.ptr<float>(y);
    for (int x = 0; x < lw; ++x) dst[x] = 0.f;
    for (int sy = 0; sy < scale; ++sy) {
      const float* b = blurScratch.ptr<float>(y * scale + sy);
      for (int x = 0; x < lw; ++x) {
        float acc = 0.f;
        const int bx = x * scale;
        for (int sx = 0; sx < scale; ++sx) acc += b[bx + sx];
        dst[x] += acc;
      }
    }
    for (int x = 0; x < lw; ++x) dst[x] *= invS2;
  }
}

namespace {

struct TileStats {
  double initSq = 0.0;
  double finalSq = 0.0;
  int64_t count = 0;
};

// Serial IBP refinement on an already-initialized HR tile and its corresponding LR tile.
// Safe to call concurrently on independent tiles inside an OpenMP loop.
TileStats refineTileIbp(const cv::Mat& lrTile, cv::Mat& hrTile, int scale, const ReconstructionParams& p,
                        const std::vector<float>& psfK, int padLr) {
  TileStats st;
  const int lw = lrTile.cols, lh = lrTile.rows;
  const int hw = lw * scale, hh = lh * scale;
  cv::Mat blur(hh, hw, CV_32FC1), tmp(hh, hw, CV_32FC1);
  cv::Mat pred(lh, lw, CV_32FC1), upErr(hh, hw, CV_32FC1), grad(hh, hw, CV_32FC1);

  auto evalRmse = [&](double& sqSum) {
    sqSum = 0.0;
    int64_t n = 0;
    for (int y = padLr; y < lh - padLr; ++y) {
      const float* pr = pred.ptr<float>(y);
      const float* lr = lrTile.ptr<float>(y);
      for (int x = padLr; x < lw - padLr; ++x) {
        const double d = static_cast<double>(pr[x]) - lr[x];
        sqSum += d * d;
        ++n;
      }
    }
    st.count = n;
  };

  const float delta = p.huberDelta;
  for (int it = 0; it < p.iterations; ++it) {
    downsampleBoxPsf(hrTile, pred, scale, psfK, blur, tmp);
    if (it == 0) evalRmse(st.initSq);

    // Huber gradient on LR residual, splatted (exact transpose of box avg * s^2) to HR
    for (int y = 0; y < lh; ++y) {
      const float* pr = pred.ptr<float>(y);
      const float* lr = lrTile.ptr<float>(y);
      for (int sy = 0; sy < scale; ++sy) {
        float* ur = upErr.ptr<float>(y * scale + sy);
        for (int x = 0; x < lw; ++x) {
          const float r = pr[x] - lr[x];
          const float g = (std::abs(r) <= delta) ? r : (r > 0.f ? delta : -delta);
          const int bx = x * scale;
          for (int sx = 0; sx < scale; ++sx) ur[bx + sx] = g;
        }
      }
    }
    kern::sepFilter(upErr, grad, psfK, tmp);
    if (p.btvLambda > 0.f) kern::addBtvGradient(hrTile, grad, p.btvRadius, p.btvAlpha, p.btvLambda);

    for (int y = 0; y < hh; ++y) {
      float* hr = hrTile.ptr<float>(y);
      const float* gr = grad.ptr<float>(y);
      for (int x = 0; x < hw; ++x) hr[x] = std::clamp(hr[x] - p.step * gr[x], 0.f, 1.f);
    }
  }
  downsampleBoxPsf(hrTile, pred, scale, psfK, blur, tmp);
  evalRmse(st.finalSq);
  return st;
}

}  // namespace

IbpStats iterativeBackProjection(const cv::Mat& lr, cv::Mat& hr, int scale,
                                 const ReconstructionParams& params, int tileSize) {
  CV_Assert(lr.type() == CV_32FC1);
  require(scale == 2 || scale == 3 || scale == 4 || scale == 8, ErrorCode::InvalidArgument,
          "IBP scale must be 2, 3, 4 or 8");

  const cv::Size hrSize(lr.cols * scale, lr.rows * scale);
  // 1. Edge-directed initial HR estimate (already OpenMP parallelized across rows)
  edgeDirectedUpscale(lr, hr, hrSize, params.edgeBlend);

  IbpStats stats;
  if (params.iterations <= 0) return stats;

  const float psfSigma = params.psfSigma > 0.f ? params.psfSigma : 0.28f * scale;
  const std::vector<float> psfK = kern::gaussianKernel(psfSigma);

  const int lw = lr.cols, lh = lr.rows;
  const int ts = std::max(32, tileSize);
  const int nx = (lw + ts - 1) / ts;
  const int ny = (lh + ts - 1) / ts;
  const int totalTiles = nx * ny;
  stats.tilesProcessed = totalTiles;

  if (totalTiles == 1) {
    TileStats ts0 = refineTileIbp(lr, hr, scale, params, psfK, 0);
    if (ts0.count > 0) {
      stats.initialRmse = std::sqrt(ts0.initSq / ts0.count);
      stats.finalRmse = std::sqrt(ts0.finalSq / ts0.count);
    }
    return stats;
  }

  // Tiled refinement with LR overlap halo so PSF & BTV boundary effects never show seams
  const int pad = 8;
  cv::Mat initHr = hr.clone();
  std::vector<TileStats> tstats(totalTiles);
  const int nthr = compute::maxThreads();

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthr) if (nthr > 1)
  for (int tid = 0; tid < totalTiles; ++tid) {
    const int ty = tid / nx;
    const int tx = tid % nx;
    const int x0 = tx * ts;
    const int y0 = ty * ts;
    const int x1 = std::min(lw, x0 + ts);
    const int y1 = std::min(lh, y0 + ts);

    const int rx0 = std::max(0, x0 - pad);
    const int ry0 = std::max(0, y0 - pad);
    const int rx1 = std::min(lw, x1 + pad);
    const int ry1 = std::min(lh, y1 + pad);

    cv::Mat lrTile = lr(cv::Rect(rx0, ry0, rx1 - rx0, ry1 - ry0));
    cv::Mat hrTile = initHr(cv::Rect(rx0 * scale, ry0 * scale, (rx1 - rx0) * scale, (ry1 - ry0) * scale)).clone();
    tstats[tid] = refineTileIbp(lrTile, hrTile, scale, params, psfK, 0);

    // Copy non-overlapping valid interior back to hr
    const int ox = (x0 - rx0) * scale;
    const int oy = (y0 - ry0) * scale;
    const int ow = (x1 - x0) * scale;
    const int oh = (y1 - y0) * scale;
    hrTile(cv::Rect(ox, oy, ow, oh)).copyTo(hr(cv::Rect(x0 * scale, y0 * scale, ow, oh)));
  }

  double initSq = 0.0, finalSq = 0.0;
  int64_t cnt = 0;
  for (const auto& s : tstats) {
    initSq += s.initSq;
    finalSq += s.finalSq;
    cnt += s.count;
  }
  if (cnt > 0) {
    stats.initialRmse = std::sqrt(initSq / cnt);
    stats.finalRmse = std::sqrt(finalSq / cnt);
  }
  return stats;
}

IbpStats iterativeBackProjectionProgressive(const cv::Mat& lr, cv::Mat& hr, int scale,
                                            const ReconstructionParams& params, int tileSize) {
  if (scale != 2 && scale != 4 && scale != 8) {
    return iterativeBackProjection(lr, hr, scale, params, tileSize);
  }
  IbpStats total;
  cv::Mat cur = lr;
  int s = 1;
  while (s < scale) {
    cv::Mat next;
    IbpStats st = iterativeBackProjection(cur, next, 2, params, tileSize);
    total.initialRmse = (s == 1) ? st.initialRmse : total.initialRmse;
    total.finalRmse = st.finalRmse;
    total.tilesProcessed += st.tilesProcessed;
    cur = std::move(next);
    s *= 2;
  }
  hr = std::move(cur);
  return total;
}

}  // namespace vu::recon
