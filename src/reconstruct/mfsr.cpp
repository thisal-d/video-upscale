#include "reconstruct/mfsr.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "compute/backend.hpp"
#include "compute/kernels.hpp"
#include "core/error.hpp"
#include "motion/optical_flow.hpp"
#include "reconstruct/ibp.hpp"
#include "reconstruct/spatial.hpp"

namespace vu::recon {

namespace {

void buildFastMaps(const cv::Mat& flow, cv::Mat& map1, cv::Mat& map2) {
  const int w = flow.cols, h = flow.rows;
  cv::Mat mx(h, w, CV_32FC1), my(h, w, CV_32FC1);
  const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && h >= 128)
  for (int y = 0; y < h; ++y) {
    const cv::Vec2f* fl = flow.ptr<cv::Vec2f>(y);
    float* px = mx.ptr<float>(y);
    float* py = my.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      px[x] = static_cast<float>(x) + fl[x][0];
      py[x] = static_cast<float>(y) + fl[x][1];
    }
  }
  cv::convertMaps(mx, my, map1, map2, CV_16SC2);
}

void warpLrToHrSubpixel(const cv::Mat& lrSrc, const cv::Mat& lrFlow, cv::Mat& hrDst, int scale) {
  const int lw = lrSrc.cols, lh = lrSrc.rows;
  const int hw = lw * scale, hh = lh * scale;
  cv::Mat hrFlow;
  cv::resize(lrFlow, hrFlow, cv::Size(hw, hh), 0, 0, cv::INTER_LINEAR);

  cv::Mat mapX(hh, hw, CV_32FC1), mapY(hh, hw, CV_32FC1);
  const float invS = 1.0f / static_cast<float>(scale);
  const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && hh >= 128)
  for (int Y = 0; Y < hh; ++Y) {
    const float yBase = (Y + 0.5f) * invS - 0.5f;
    const cv::Vec2f* fl = hrFlow.ptr<cv::Vec2f>(Y);
    float* mx = mapX.ptr<float>(Y);
    float* my = mapY.ptr<float>(Y);
    for (int X = 0; X < hw; ++X) {
      mx[X] = (X + 0.5f) * invS - 0.5f + fl[X][0];
      my[X] = yBase + fl[X][1];
    }
  }
  cv::remap(lrSrc, hrDst, mapX, mapY, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
}

void downsampleBoxPsfParallel(const cv::Mat& hr, cv::Mat& lrOut, int scale, const std::vector<float>& psfK,
                              cv::Mat& blur, cv::Mat& tmp) {
  kern::sepFilterParallel(hr, blur, psfK, tmp);
  const int lw = hr.cols / scale, lh = hr.rows / scale;
  lrOut.create(lh, lw, CV_32FC1);
  const float invS2 = 1.0f / static_cast<float>(scale * scale);
  const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && lh >= 64)
  for (int y = 0; y < lh; ++y) {
    float* dst = lrOut.ptr<float>(y);
    for (int x = 0; x < lw; ++x) dst[x] = 0.f;
    for (int sy = 0; sy < scale; ++sy) {
      const float* b = blur.ptr<float>(y * scale + sy);
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

void splatErrorToHrParallel(const cv::Mat& predLr, const cv::Mat& obsLr, int scale, float huberDelta,
                            cv::Mat& upErr, const cv::Mat* maskLr = nullptr) {
  const int lw = predLr.cols, lh = predLr.rows;
  upErr.create(lh * scale, lw * scale, CV_32FC1);
  const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && lh >= 64)
  for (int y = 0; y < lh; ++y) {
    const float* pr = predLr.ptr<float>(y);
    const float* ob = obsLr.ptr<float>(y);
    const float* mk = maskLr ? maskLr->ptr<float>(y) : nullptr;
    for (int sy = 0; sy < scale; ++sy) {
      float* ur = upErr.ptr<float>(y * scale + sy);
      for (int x = 0; x < lw; ++x) {
        const float r = pr[x] - ob[x];
        float g = (std::abs(r) <= huberDelta) ? r : (r > 0.f ? huberDelta : -huberDelta);
        if (mk) g *= mk[x];
        const int bx = x * scale;
        for (int sx = 0; sx < scale; ++sx) ur[bx + sx] = g;
      }
    }
  }
}

double computeRmse(const cv::Mat& a, const cv::Mat& b) {
  double sq = 0.0;
  for (int y = 0; y < a.rows; ++y) {
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    for (int x = 0; x < a.cols; ++x) {
      const double d = static_cast<double>(pa[x]) - pb[x];
      sq += d * d;
    }
  }
  return std::sqrt(sq / std::max<size_t>(1, a.total()));
}

}  // namespace

MfsrStats reconstructMultiFrame(const std::vector<MfsrInputFrame>& window, int refIdx, cv::Mat& hrOut,
                                const ProcessingConfig& cfg, cv::DISOpticalFlow* flowEst,
                                const cv::Mat* prevHrOut, const cv::Mat* prevYFloat,
                                const cv::Mat* prevYU8) {
  require(refIdx >= 0 && refIdx < static_cast<int>(window.size()), ErrorCode::InvalidArgument, "bad refIdx");
  const auto& ref = window[refIdx];
  const int lw = ref.yFloat.cols, lh = ref.yFloat.rows;
  const bool useProgressive = cfg.progressive && (cfg.scale == 4 || cfg.scale == 8);
  const int scale = useProgressive ? 2 : cfg.scale;
  const int hw = lw * scale, hh = lh * scale;
  const cv::Size hrSize(hw, hh);

  cv::Ptr<cv::DISOpticalFlow> ownedFlow;
  if (!flowEst) {
    ownedFlow = motion::createFlowEstimator(cfg.temporal.flowPreset);
    flowEst = ownedFlow.get();
  }

  // Stage 7: Edge-directed base reconstruction
  cv::Mat xSpatial;
  edgeDirectedUpscale(ref.yFloat, xSpatial, hrSize, cfg.recon.edgeBlend);

  struct AlignedNeighbor {
    int dt = 0;
    float temporalDecay = 1.f;
    cv::Mat yObsLr;
    cv::Mat mapFwd1, mapFwd2;  // precomputed CV_16SC2 remap tables for ref -> nbr
    cv::Mat mapBwd1, mapBwd2;  // precomputed CV_16SC2 remap tables for nbr -> ref
    cv::Mat hrConf;            // CV_32FC1 in [0, 1] on ref HR grid
    cv::Mat lrConfWarped;      // CV_32FC1 validity mask on nbr LR grid
    cv::Mat warpedHrObs;       // sub-pixel warped nbr observation on ref HR grid
  };
  std::vector<AlignedNeighbor> neighbors;

  MfsrStats stats;
  double sumConf = 0.0, sumOcc = 0.0;

  // Stages 1-5: Frame selection, motion estimation, sub-pixel alignment, occlusion & confidence
  for (int i = 0; i < static_cast<int>(window.size()); ++i) {
    if (i == refIdx) continue;
    const auto& nbr = window[i];
    if (nbr.segment != ref.segment) continue;

    auto mr = motion::estimateMotion(ref.yFloat, nbr.yFloat, ref.yU8, nbr.yU8, cfg.temporal, flowEst);
    if (mr.stats.meanConfidence < 0.03f || mr.stats.occludedFraction > 0.88f) continue;

    cv::Mat erodedConf;
    cv::erode(mr.confidence, erodedConf, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
              cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);

    AlignedNeighbor an;
    an.dt = std::abs(i - refIdx);
    an.temporalDecay = std::pow(0.85f, static_cast<float>(an.dt - 1));
    an.yObsLr = nbr.yFloat;

    cv::Mat hrFlowFwd, hrFlowBwd;
    motion::upscaleFlow(mr.flowFwd, hrFlowFwd, hrSize, scale);
    motion::upscaleFlow(mr.flowBwd, hrFlowBwd, hrSize, scale);
    buildFastMaps(hrFlowFwd, an.mapFwd1, an.mapFwd2);
    buildFastMaps(hrFlowBwd, an.mapBwd1, an.mapBwd2);

    cv::resize(erodedConf, an.hrConf, hrSize, 0, 0, cv::INTER_LINEAR);
    motion::warpWithFlow(erodedConf, mr.flowBwd, an.lrConfWarped, cv::INTER_LINEAR);
    warpLrToHrSubpixel(nbr.yFloat, mr.flowFwd, an.warpedHrObs, scale);

    sumConf += mr.stats.meanConfidence;
    sumOcc += mr.stats.occludedFraction;
    neighbors.push_back(std::move(an));
  }

  stats.neighborsUsed = static_cast<int>(neighbors.size());
  if (stats.neighborsUsed > 0) {
    stats.meanNeighborConfidence = static_cast<float>(sumConf / stats.neighborsUsed);
    stats.meanOccludedFraction = static_cast<float>(sumOcc / stats.neighborsUsed);
  }

  // Stage 6: Confidence-weighted temporal fusion for initial HR state X^(0)
  cv::Mat X(hh, hw, CV_32FC1);
  if (neighbors.empty()) {
    xSpatial.copyTo(X);
  } else {
    const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1)
    for (int Y = 0; Y < hh; ++Y) {
      const float* sp = xSpatial.ptr<float>(Y);
      float* dst = X.ptr<float>(Y);
      for (int Xx = 0; Xx < hw; ++Xx) {
        float wSum = 1.0f;
        float vSum = sp[Xx];
        for (const auto& an : neighbors) {
          const float c = an.hrConf.ptr<float>(Y)[Xx] * an.temporalDecay;
          if (c > 0.f) {
            wSum += c;
            vSum += c * an.warpedHrObs.ptr<float>(Y)[Xx];
          }
        }
        dst[Xx] = std::clamp(vSum / wSum, 0.f, 1.f);
      }
    }
  }

  // Stage 8: Multi-frame Huber + BTV Iterative Back-Projection
  const float psfSigma = cfg.recon.psfSigma > 0.f ? cfg.recon.psfSigma : 0.28f * scale;
  const std::vector<float> psfK = kern::gaussianKernel(psfSigma);
  cv::Mat blur(hh, hw, CV_32FC1), tmp(hh, hw, CV_32FC1);
  cv::Mat predLr(lh, lw, CV_32FC1), upErr(hh, hw, CV_32FC1);
  cv::Mat gradTotal(hh, hw, CV_32FC1);
  cv::Mat xInNbr, nbrGradInNbr, nbrGradInRef;

  cv::Mat weightMap(hh, hw, CV_32FC1, cv::Scalar(1.0f));
  for (const auto& an : neighbors) {
    cv::addWeighted(weightMap, 1.0, an.hrConf, an.temporalDecay, 0.0, weightMap);
  }

  const int mfIters = std::min(cfg.recon.iterations, cfg.temporal.flowPreset == "medium" ? 4 : 3);
  const int nthr = compute::maxThreads();

  for (int it = 0; it < cfg.recon.iterations; ++it) {
    downsampleBoxPsfParallel(X, predLr, scale, psfK, blur, tmp);
    if (it == 0) stats.initialRefRmse = computeRmse(predLr, ref.yFloat);

    splatErrorToHrParallel(predLr, ref.yFloat, scale, cfg.recon.huberDelta, upErr);
    kern::sepFilterParallel(upErr, gradTotal, psfK, tmp);

    const bool useNeighborsThisIter = (it < mfIters) && !neighbors.empty();
    if (useNeighborsThisIter) {
      for (const auto& an : neighbors) {
        cv::remap(X, xInNbr, an.mapBwd1, an.mapBwd2, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        downsampleBoxPsfParallel(xInNbr, predLr, scale, psfK, blur, tmp);
        splatErrorToHrParallel(predLr, an.yObsLr, scale, cfg.recon.huberDelta, upErr, &an.lrConfWarped);
        kern::sepFilterParallel(upErr, nbrGradInNbr, psfK, tmp);
        cv::remap(nbrGradInNbr, nbrGradInRef, an.mapFwd1, an.mapFwd2, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && hh >= 64)
        for (int Y = 0; Y < hh; ++Y) {
          float* gt = gradTotal.ptr<float>(Y);
          const float* gn = nbrGradInRef.ptr<float>(Y);
          const float* cf = an.hrConf.ptr<float>(Y);
          const float dec = an.temporalDecay;
          for (int Xx = 0; Xx < hw; ++Xx) gt[Xx] += dec * cf[Xx] * gn[Xx];
        }
      }

#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && hh >= 64)
      for (int Y = 0; Y < hh; ++Y) {
        float* gt = gradTotal.ptr<float>(Y);
        const float* wm = weightMap.ptr<float>(Y);
        for (int Xx = 0; Xx < hw; ++Xx) gt[Xx] /= wm[Xx];
      }
    }

    if (cfg.recon.btvLambda > 0.f) {
      kern::addBtvGradientParallel(X, gradTotal, cfg.recon.btvRadius, cfg.recon.btvAlpha, cfg.recon.btvLambda);
    }

    const float step = cfg.recon.step;
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && hh >= 64)
    for (int Y = 0; Y < hh; ++Y) {
      float* xRow = X.ptr<float>(Y);
      const float* gRow = gradTotal.ptr<float>(Y);
      for (int Xx = 0; Xx < hw; ++Xx) xRow[Xx] = std::clamp(xRow[Xx] - step * gRow[Xx], 0.f, 1.f);
    }
  }

  // Stage 9: Observation consistency check
  downsampleBoxPsfParallel(X, predLr, scale, psfK, blur, tmp);
  stats.finalRefRmse = computeRmse(predLr, ref.yFloat);
  if (cfg.recon.iterations > 0 && stats.initialRefRmse > 0.0 &&
      stats.finalRefRmse > stats.initialRefRmse * 1.08 && !neighbors.empty()) {
    stats.consistencyFallback = true;
    auto st = iterativeBackProjection(ref.yFloat, X, scale, cfg.recon, cfg.tileSize);
    stats.finalRefRmse = st.finalRmse;
  }

  // Progressive 2x stage(s) if progressive=true and power-of-two scale > 2
  if (useProgressive) {
    iterativeBackProjectionProgressive(X, hrOut, cfg.scale / 2, cfg.recon, cfg.tileSize);
  } else {
    hrOut = std::move(X);
  }

  // Stage 10: Optional temporal refinement (experimental)
  const cv::Size targetSize(lw * cfg.scale, lh * cfg.scale);
  if (cfg.temporal.temporalRefine > 0.f && prevHrOut && !prevHrOut->empty() && prevYFloat && prevYU8 &&
      prevHrOut->size() == targetSize) {
    auto prevMr = motion::estimateMotion(ref.yFloat, *prevYFloat, ref.yU8, *prevYU8, cfg.temporal, flowEst);
    if (prevMr.stats.meanConfidence > 0.25f) {
      cv::Mat hrPrevFlow, warpedPrevHr, hrPrevConf;
      motion::upscaleFlow(prevMr.flowFwd, hrPrevFlow, targetSize, cfg.scale);
      motion::warpWithFlow(*prevHrOut, hrPrevFlow, warpedPrevHr, cv::INTER_CUBIC);
      cv::resize(prevMr.confidence, hrPrevConf, targetSize, 0, 0, cv::INTER_LINEAR);
      const float alpha = cfg.temporal.temporalRefine;
      for (int Y = 0; Y < targetSize.height; ++Y) {
        float* xr = hrOut.ptr<float>(Y);
        const float* pr = warpedPrevHr.ptr<float>(Y);
        const float* cr = hrPrevConf.ptr<float>(Y);
        for (int Xx = 0; Xx < targetSize.width; ++Xx) {
          const float w = alpha * cr[Xx];
          xr[Xx] = std::clamp((1.f - w) * xr[Xx] + w * pr[Xx], 0.f, 1.f);
        }
      }
    }
  }
  return stats;
}

}  // namespace vu::recon
