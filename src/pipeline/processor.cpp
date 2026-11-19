#include "pipeline/processor.hpp"

#include <deque>
#include <format>

#include "analysis/frame_analysis.hpp"
#include "compute/kernels.hpp"
#include "core/error.hpp"
#include "enhance/spatial_enhance.hpp"
#include "motion/optical_flow.hpp"
#include "reconstruct/ibp.hpp"
#include "reconstruct/mfsr.hpp"
#include "reconstruct/spatial.hpp"

namespace vu {

namespace {

class SpatialProcessor final : public FrameProcessor {
 public:
  explicit SpatialProcessor(const ProcessingConfig& cfg) : cfg_(cfg) {}

  void begin(const VideoInfo& in, FramePool& outPool) override {
    in_ = in;
    outPool_ = &outPool;
    outW_ = in.width * cfg_.scale;
    outH_ = in.height * cfg_.scale;
  }

  void push(ConstFramePtr f, const Emit& emit) override {
    FramePtr out = outPool_->acquire();
    out->pts = f->pts;
    out->duration = f->duration;
    out->index = f->index;
    out->segment = f->segment;
    out->sceneCut = f->sceneCut;

    kern::u8ToFloat(f->y, yf_);

    const bool isPureReplication = (cfg_.upscaler == Upscaler::TyranoxCrazy);
    const bool isBaseline = (cfg_.upscaler == Upscaler::Nearest || cfg_.upscaler == Upscaler::Bilinear ||
                             cfg_.upscaler == Upscaler::Bicubic || cfg_.upscaler == Upscaler::Lanczos ||
                             isPureReplication);
    const bool forceEnhance = (cfg_.enhance.denoise == Toggle::On || cfg_.enhance.deblock == Toggle::On);
    const bool useProgressive = cfg_.progressive && (cfg_.scale == 4 || cfg_.scale == 8) && !isPureReplication;
    const Upscaler effectiveChroma = isPureReplication ? Upscaler::TyranoxCrazy : cfg_.chromaUpscaler;

    cv::Mat lumaIn = yf_;
    if (!isBaseline || forceEnhance) {
      if (frames_ % 4 == 0) lastMetrics_ = analysis::analyzeFrame(yf_);
      sumNoise_ += lastMetrics_.noiseSigma;
      sumBlockiness_ += lastMetrics_.blockiness;

      EnhanceParams ep = cfg_.enhance;
      if (isBaseline) {
        if (ep.denoise == Toggle::Auto) ep.denoise = Toggle::Off;
        if (ep.deblock == Toggle::Auto) ep.deblock = Toggle::Off;
      }
      auto er = enhance::preprocessLuma(yf_, enhancedY_, ep, lastMetrics_);
      if (er.deblocked) ++deblockedFrames_;
      if (er.denoised) ++denoisedFrames_;
      lumaIn = enhancedY_;
    }

    if (cfg_.upscaler == Upscaler::BackProjection) {
      ReconstructionParams rp = cfg_.recon;
      if (rp.psfSigma <= 0.f) {
        // Adapt PSF to measured blur so sharp sources aren't over-deconvolved while blurry ones are restored.
        const float t = std::clamp((lastMetrics_.blurScore - 0.25f) / 0.35f, 0.f, 1.f);
        rp.psfSigma = (0.26f + 0.16f * t) * (useProgressive ? 2 : cfg_.scale);
      }
      if (lastMetrics_.noiseSigma > 0.008f || lastMetrics_.blockiness > 0.12f) {
        rp.btvLambda *= (1.0f + 50.0f * lastMetrics_.noiseSigma + 3.0f * lastMetrics_.blockiness);
      }
      recon::IbpStats ist =
          useProgressive
              ? recon::iterativeBackProjectionProgressive(lumaIn, outYf_, cfg_.scale, rp, cfg_.tileSize)
              : recon::iterativeBackProjection(lumaIn, outYf_, cfg_.scale, rp, cfg_.tileSize);
      sumInitRmse_ += ist.initialRmse;
      sumFinalRmse_ += ist.finalRmse;
    } else if (useProgressive) {
      recon::upscaleProgressive(lumaIn, outYf_, cfg_.scale, cfg_.upscaler, cfg_.recon.edgeBlend);
    } else {
      recon::upscalePlane(lumaIn, outYf_, cv::Size(outW_, outH_), cfg_.upscaler, cfg_.recon.edgeBlend);
    }

    if (cfg_.enhance.sharpen > 0.f) {
      enhance::adaptiveSharpen(outYf_, sharpenedY_, cfg_.enhance.sharpen, cfg_.enhance.sharpenRadius,
                               cfg_.enhance.haloSuppression, cfg_.enhance.haloOvershoot);
      kern::floatToU8(sharpenedY_, out->y);
    } else {
      kern::floatToU8(outYf_, out->y);
    }

    const cv::Size cSize((outW_ + 1) / 2, (outH_ + 1) / 2);
    kern::u8ToFloat(f->u, uf_);
    recon::upscalePlane(uf_, outUf_, cSize, effectiveChroma);
    kern::floatToU8(outUf_, out->u);

    kern::u8ToFloat(f->v, vf_);
    recon::upscalePlane(vf_, outVf_, cSize, effectiveChroma);
    kern::floatToU8(outVf_, out->v);

    ++frames_;
    emit(std::move(out));
  }

  void flush(const Emit&) override {}

  std::string describe() const override {
    const bool isPureReplication = (cfg_.upscaler == Upscaler::TyranoxCrazy);
    const bool useProgressive = cfg_.progressive && (cfg_.scale == 4 || cfg_.scale == 8) && !isPureReplication;
    const Upscaler effectiveChroma = isPureReplication ? Upscaler::TyranoxCrazy : cfg_.chromaUpscaler;
    return std::format("spatial({}, {}x, {}, chroma={})", toString(cfg_.upscaler), cfg_.scale,
                       useProgressive ? "progressive" : "direct", toString(effectiveChroma));
  }

  nlohmann::json statistics() const override {
    const double n = frames_ > 0 ? static_cast<double>(frames_) : 1.0;
    nlohmann::json j = {
        {"frames_processed", frames_},
        {"mode", describe()},
        {"deblocked_frames", deblockedFrames_},
        {"denoised_frames", denoisedFrames_},
        {"mean_noise_sigma", sumNoise_ / n},
        {"mean_blockiness", sumBlockiness_ / n},
    };
    if (cfg_.upscaler == Upscaler::BackProjection) {
      j["mean_ibp_initial_rmse"] = sumInitRmse_ / n;
      j["mean_ibp_final_rmse"] = sumFinalRmse_ / n;
    }
    return j;
  }

 private:
  ProcessingConfig cfg_;
  VideoInfo in_;
  FramePool* outPool_ = nullptr;
  int outW_ = 0, outH_ = 0;
  int64_t frames_ = 0;
  int64_t deblockedFrames_ = 0, denoisedFrames_ = 0;
  double sumNoise_ = 0.0, sumBlockiness_ = 0.0;
  double sumInitRmse_ = 0.0, sumFinalRmse_ = 0.0;
  analysis::FrameMetrics lastMetrics_;
  cv::Mat yf_, uf_, vf_, enhancedY_, outYf_, sharpenedY_, outUf_, outVf_;
};

class MultiFrameProcessor final : public FrameProcessor {
 public:
  explicit MultiFrameProcessor(const ProcessingConfig& cfg) : cfg_(cfg) {}

  void begin(const VideoInfo& in, FramePool& outPool) override {
    in_ = in;
    outPool_ = &outPool;
    outW_ = in.width * cfg_.scale;
    outH_ = in.height * cfg_.scale;
    flowEst_ = motion::createFlowEstimator(cfg_.temporal.flowPreset);
  }

  int lookahead() const override { return cfg_.temporal.radius; }

  void push(ConstFramePtr f, const Emit& emit) override {
    cv::Mat yf, enhanced;
    kern::u8ToFloat(f->y, yf);

    bool isCut = false;
    if (!prevRawY_.empty()) {
      auto sc = analysis::detectSceneCut(prevRawY_, yf, cfg_.sceneCutThreshold);
      if (sc.isCut) {
        isCut = true;
        ++segment_;
        ++sceneCuts_;
      }
    }
    yf.copyTo(prevRawY_);

    if (pushed_ % 4 == 0) lastMetrics_ = analysis::analyzeFrame(yf);
    sumNoise_ += lastMetrics_.noiseSigma;
    sumBlockiness_ += lastMetrics_.blockiness;

    auto er = enhance::preprocessLuma(yf, enhanced, cfg_.enhance, lastMetrics_);
    if (er.deblocked) ++deblockedFrames_;
    if (er.denoised) ++denoisedFrames_;

    Entry e;
    e.raw = f;
    e.mfsr.yFloat = std::move(enhanced);
    kern::floatToU8(e.mfsr.yFloat, e.mfsr.yU8);
    e.mfsr.segment = segment_;
    e.mfsr.index = f->index;
    e.sceneCut = isCut;
    win_.push_back(std::move(e));
    ++pushed_;

    // Once we have `radius` future frames beyond the center, process and emit center.
    if (nextEmitIdx_ + cfg_.temporal.radius < static_cast<int>(win_.size())) {
      emitCenter(emit);
    }
  }

  void flush(const Emit& emit) override {
    while (nextEmitIdx_ < static_cast<int>(win_.size())) {
      emitCenter(emit);
    }
    win_.clear();
  }

  std::string describe() const override {
    const bool useProgressive = cfg_.progressive && (cfg_.scale == 4 || cfg_.scale == 8);
    return std::format("mfsr(window={}, {}x, {}, flow={})", 2 * cfg_.temporal.radius + 1, cfg_.scale,
                       useProgressive ? "progressive" : "direct", cfg_.temporal.flowPreset);
  }

  nlohmann::json statistics() const override {
    const double n = emitted_ > 0 ? static_cast<double>(emitted_) : 1.0;
    return {
        {"frames_processed", emitted_},
        {"mode", describe()},
        {"scene_cuts", sceneCuts_},
        {"deblocked_frames", deblockedFrames_},
        {"denoised_frames", denoisedFrames_},
        {"mean_noise_sigma", sumNoise_ / n},
        {"mean_blockiness", sumBlockiness_ / n},
        {"mean_neighbors_used", sumNeighbors_ / n},
        {"mean_neighbor_confidence", sumConfidence_ / n},
        {"mean_occluded_fraction", sumOccluded_ / n},
        {"mean_initial_rmse", sumInitRmse_ / n},
        {"mean_final_rmse", sumFinalRmse_ / n},
        {"consistency_fallbacks", fallbacks_},
    };
  }

 private:
  struct Entry {
    ConstFramePtr raw;
    recon::MfsrInputFrame mfsr;
    bool sceneCut = false;
  };

  void emitCenter(const Emit& emit) {
    const int refIdx = nextEmitIdx_;
    const Entry& center = win_[refIdx];

    std::vector<recon::MfsrInputFrame> frames;
    frames.reserve(win_.size());
    for (const auto& w : win_) frames.push_back(w.mfsr);

    const bool samePrevSeg = (prevOutSeg_ == center.mfsr.segment) && !prevHrOut_.empty();
    ProcessingConfig runCfg = cfg_;
    if (runCfg.recon.psfSigma <= 0.f) {
      const float t = std::clamp((lastMetrics_.blurScore - 0.25f) / 0.35f, 0.f, 1.f);
      const bool useProgressive = runCfg.progressive && (runCfg.scale == 4 || runCfg.scale == 8);
      const int effScale = useProgressive ? 2 : runCfg.scale;
      runCfg.recon.psfSigma = (0.26f + 0.16f * t) * effScale;
    }
    if (lastMetrics_.noiseSigma > 0.008f || lastMetrics_.blockiness > 0.12f) {
      runCfg.recon.btvLambda *= (1.0f + 50.0f * lastMetrics_.noiseSigma + 3.0f * lastMetrics_.blockiness);
    }
    cv::Mat hrY;
    auto st = recon::reconstructMultiFrame(frames, refIdx, hrY, runCfg, flowEst_.get(),
                                           samePrevSeg ? &prevHrOut_ : nullptr,
                                           samePrevSeg ? &prevOutLrY_ : nullptr,
                                           samePrevSeg ? &prevOutLrU8_ : nullptr);

    sumNeighbors_ += st.neighborsUsed;
    sumConfidence_ += st.meanNeighborConfidence;
    sumOccluded_ += st.meanOccludedFraction;
    sumInitRmse_ += st.initialRefRmse;
    sumFinalRmse_ += st.finalRefRmse;
    if (st.consistencyFallback) ++fallbacks_;

    if (cfg_.temporal.temporalRefine > 0.f) {
      hrY.copyTo(prevHrOut_);
      center.mfsr.yFloat.copyTo(prevOutLrY_);
      center.mfsr.yU8.copyTo(prevOutLrU8_);
      prevOutSeg_ = center.mfsr.segment;
    }

    FramePtr out = outPool_->acquire();
    out->pts = center.raw->pts;
    out->duration = center.raw->duration;
    out->index = center.raw->index;
    out->segment = center.mfsr.segment;
    out->sceneCut = center.sceneCut;

    if (cfg_.enhance.sharpen > 0.f) {
      enhance::adaptiveSharpen(hrY, sharpenedY_, cfg_.enhance.sharpen, cfg_.enhance.sharpenRadius,
                               cfg_.enhance.haloSuppression, cfg_.enhance.haloOvershoot);
      kern::floatToU8(sharpenedY_, out->y);
    } else {
      kern::floatToU8(hrY, out->y);
    }

    const cv::Size cSize((outW_ + 1) / 2, (outH_ + 1) / 2);
    kern::u8ToFloat(center.raw->u, uf_);
    recon::upscalePlane(uf_, outUf_, cSize, cfg_.chromaUpscaler);
    kern::floatToU8(outUf_, out->u);

    kern::u8ToFloat(center.raw->v, vf_);
    recon::upscalePlane(vf_, outVf_, cSize, cfg_.chromaUpscaler);
    kern::floatToU8(outVf_, out->v);

    ++emitted_;
    emit(std::move(out));

    // Slide window: keep at most `radius` past frames before the next center
    if (nextEmitIdx_ >= cfg_.temporal.radius) {
      win_.pop_front();
    } else {
      ++nextEmitIdx_;
    }
  }

  ProcessingConfig cfg_;
  VideoInfo in_;
  FramePool* outPool_ = nullptr;
  int outW_ = 0, outH_ = 0;
  cv::Ptr<cv::DISOpticalFlow> flowEst_;
  std::deque<Entry> win_;
  int nextEmitIdx_ = 0;
  int segment_ = 0;
  int64_t pushed_ = 0, emitted_ = 0, sceneCuts_ = 0;
  int64_t deblockedFrames_ = 0, denoisedFrames_ = 0, fallbacks_ = 0;
  double sumNoise_ = 0.0, sumBlockiness_ = 0.0;
  double sumNeighbors_ = 0.0, sumConfidence_ = 0.0, sumOccluded_ = 0.0;
  double sumInitRmse_ = 0.0, sumFinalRmse_ = 0.0;
  analysis::FrameMetrics lastMetrics_;
  cv::Mat prevRawY_;
  cv::Mat prevHrOut_, prevOutLrY_, prevOutLrU8_;
  int prevOutSeg_ = -1;
  cv::Mat sharpenedY_, uf_, vf_, outUf_, outVf_;
};

}  // namespace

std::unique_ptr<FrameProcessor> createProcessor(const ProcessingConfig& cfg) {
  cfg.validate();
  switch (cfg.upscaler) {
    case Upscaler::Nearest:
    case Upscaler::Bilinear:
    case Upscaler::Bicubic:
    case Upscaler::Lanczos:
    case Upscaler::EdgeAware:
    case Upscaler::BackProjection:
    case Upscaler::TyranoxCrazy:
    case Upscaler::TyranoxEdge:
    case Upscaler::TyranoxSmooth:
      return std::make_unique<SpatialProcessor>(cfg);
    case Upscaler::MultiFrame:
      return std::make_unique<MultiFrameProcessor>(cfg);
  }
  fail(ErrorCode::InvalidArgument, "unknown upscaler");
}

}  // namespace vu
