#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "compute/backend.hpp"
#include "compute/kernels.hpp"
#include "core/config.hpp"
#include "core/frame_pool.hpp"
#include "motion/optical_flow.hpp"
#include "pipeline/processor.hpp"
#include "reconstruct/ibp.hpp"
#include "reconstruct/mfsr.hpp"
#include "reconstruct/spatial.hpp"

using namespace vu;

namespace {

// Evaluates a multi-scale non-periodic pattern (radial chirp + localized texture + sharp edges)
// at continuous HR coordinates so sub-pixel shifted frames have exact analytic ground truth.
cv::Mat renderAnalyticPattern(int w, int h, float shiftX, float shiftY) {
  cv::Mat img(h, w, CV_32FC1);
  for (int y = 0; y < h; ++y) {
    float* r = img.ptr<float>(y);
    const float py = static_cast<float>(y) + shiftY;
    for (int x = 0; x < w; ++x) {
      const float px = static_cast<float>(x) + shiftX;
      const float dx = px - w * 0.5f;
      const float dy = py - h * 0.5f;
      const float rad = std::sqrt(dx * dx + dy * dy);
      // Smooth low-frequency envelope for unambiguous coarse-to-fine optical flow
      float v = 0.35f + 0.15f * std::sin(0.04f * px) * std::cos(0.035f * py);
      // Multi-frequency radial zone-plate chirp (stays trackable by DIS while stressing 2x SR)
      const float env = std::exp(-0.00010f * rad * rad);
      v += 0.22f * env * std::sin(0.0045f * rad * rad);
      // Sharp rotated edge & corner square
      const float d = 0.7f * dx + 0.5f * dy;
      v += 0.18f * std::tanh(d * 1.4f);
      if (px > w * 0.12f && px < w * 0.32f && py > h * 0.15f && py < h * 0.35f) v += 0.15f;
      r[x] = std::clamp(v, 0.f, 1.f);
    }
  }
  return img;
}

double psnr01(const cv::Mat& a, const cv::Mat& b) {
  double mse = 0.0;
  for (int y = 0; y < a.rows; ++y) {
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    for (int x = 0; x < a.cols; ++x) {
      const double d = static_cast<double>(pa[x]) - pb[x];
      mse += d * d;
    }
  }
  mse /= static_cast<double>(a.rows) * a.cols;
  return mse <= 1e-12 ? 100.0 : 10.0 * std::log10(1.0 / mse);
}

double mseInterior(const cv::Mat& a, const cv::Mat& b, int pad) {
  double mse = 0.0;
  int64_t cnt = 0;
  for (int y = pad; y < a.rows - pad; ++y) {
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    for (int x = pad; x < a.cols - pad; ++x) {
      const double d = static_cast<double>(pa[x]) - pb[x];
      mse += d * d;
      ++cnt;
    }
  }
  return mse / std::max<int64_t>(1, cnt);
}

}  // namespace

TEST(MotionTest, SubpixelShiftEstimatedAndAlignedAccurately) {
  const int lw = 128, lh = 128;
  cv::Mat ref = renderAnalyticPattern(lw, lh, 0.f, 0.f);
  cv::Mat nbr = renderAnalyticPattern(lw, lh, 1.3f, -0.7f);
  cv::Mat refU8, nbrU8;
  kern::floatToU8(ref, refU8);
  kern::floatToU8(nbr, nbrU8);

  TemporalParams tp;
  tp.flowPreset = "medium";
  auto mr = motion::estimateMotion(ref, nbr, refU8, nbrU8, tp);

  EXPECT_GT(mr.stats.meanConfidence, 0.65f);
  EXPECT_LT(mr.stats.occludedFraction, 0.15f);
  EXPECT_LT(mseInterior(ref, mr.alignedLr, 8), mseInterior(ref, nbr, 8) * 0.05);
}

TEST(MotionTest, OcclusionAndForegroundMotionDetected) {
  cv::Mat ref = renderAnalyticPattern(128, 128, 0.f, 0.f);
  cv::Mat nbr = ref.clone();
  // Introduce a bright object jumping 12 px right in nbr (occluding background & leaving disoccluded region)
  ref(cv::Rect(30, 40, 28, 28)).setTo(0.95f);
  nbr(cv::Rect(46, 40, 28, 28)).setTo(0.95f);
  cv::Mat refU8, nbrU8;
  kern::floatToU8(ref, refU8);
  kern::floatToU8(nbr, nbrU8);

  TemporalParams tp;
  tp.flowPreset = "medium";
  auto mr = motion::estimateMotion(ref, nbr, refU8, nbrU8, tp);
  EXPECT_GT(mr.stats.occludedFraction, 0.01f);
}

TEST(MfsrTest, FiveFrameSubpixelReconstructionBeatsSingleFrameIbpAndBicubic) {
  const int scale = 2;
  const int hw = 256, hh = 256;
  const std::vector<float> psfK = kern::gaussianKernel(0.45f * scale);

  // 5 frames with sub-pixel HR shifts (-1.0, -0.5, 0.0, +0.5, +1.0) in X and (-0.5..+0.5) in Y
  const float shiftsX[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  const float shiftsY[5] = {-0.6f, -0.3f, 0.0f, 0.3f, 0.6f};

  cv::Mat hrTruth = renderAnalyticPattern(hw, hh, 0.f, 0.f);
  std::vector<recon::MfsrInputFrame> win(5);
  cv::Mat blur, tmp;
  for (int i = 0; i < 5; ++i) {
    cv::Mat hrI = renderAnalyticPattern(hw, hh, shiftsX[i], shiftsY[i]);
    recon::downsampleBoxPsf(hrI, win[i].yFloat, scale, psfK, blur, tmp);
    kern::floatToU8(win[i].yFloat, win[i].yU8);
    win[i].segment = 0;
    win[i].index = i;
  }

  cv::Mat bicubic, ibp, mfsr;
  recon::upscalePlane(win[2].yFloat, bicubic, hrTruth.size(), Upscaler::Bicubic);

  ProcessingConfig cfg;
  cfg.scale = scale;
  cfg.upscaler = Upscaler::MultiFrame;
  cfg.temporal.flowPreset = "medium";
  cfg.recon.iterations = 8;
  cfg.recon.psfSigma = 0.45f * scale;
  cfg.recon.btvLambda = 0.0004f;
  cfg.recon.edgeBlend = 1.0f;

  recon::iterativeBackProjection(win[2].yFloat, ibp, scale, cfg.recon, cfg.tileSize);
  auto st = recon::reconstructMultiFrame(win, 2, mfsr, cfg);

  EXPECT_EQ(st.neighborsUsed, 4);
  EXPECT_FALSE(st.consistencyFallback);
  EXPECT_LT(st.finalRefRmse, st.initialRefRmse);

  const cv::Rect interior(16, 16, hw - 32, hh - 32);
  const double psnrBic = psnr01(hrTruth(interior), bicubic(interior));
  const double psnrIbp = psnr01(hrTruth(interior), ibp(interior));
  const double psnrMfsr = psnr01(hrTruth(interior), mfsr(interior));

  EXPECT_GT(psnrMfsr, psnrBic + 0.5) << "MFSR=" << psnrMfsr << " Bicubic=" << psnrBic;
  EXPECT_GT(psnrMfsr, psnrIbp + 0.2) << "MFSR=" << psnrMfsr << " IBP=" << psnrIbp;
}

TEST(MfsrTest, SceneCutResetsTemporalBufferAndRejectsCrossScenePixels) {
  const int lw = 96, lh = 96;
  cv::Mat sceneA = cv::Mat(lh, lw, CV_32FC1, cv::Scalar(0.9f));
  cv::Mat sceneB = renderAnalyticPattern(lw, lh, 0.f, 0.f) * 0.4f;

  // Direct unit check on segment isolation
  std::vector<recon::MfsrInputFrame> win(5);
  for (int i = 0; i < 5; ++i) {
    win[i].yFloat = (i < 2) ? sceneA.clone() : sceneB.clone();
    kern::floatToU8(win[i].yFloat, win[i].yU8);
    win[i].segment = (i < 2) ? 0 : 1;
    win[i].index = i;
  }
  ProcessingConfig cfg;
  cfg.scale = 2;
  cfg.upscaler = Upscaler::MultiFrame;
  cv::Mat hrOut;
  auto st = recon::reconstructMultiFrame(win, 2, hrOut, cfg);
  EXPECT_EQ(st.neighborsUsed, 2);  // frames 0 & 1 from segment 0 must be excluded

  // End-to-end streaming MultiFrameProcessor scene cut check
  cfg.temporal.radius = 2;
  cfg.enhance.denoise = Toggle::Off;
  cfg.enhance.deblock = Toggle::Off;
  auto proc = createProcessor(cfg);

  VideoInfo vi;
  vi.width = lw;
  vi.height = lh;
  FramePool inPool(lw, lh, 10), outPool(lw * 2, lh * 2, 10);
  proc->begin(vi, outPool);

  std::vector<FramePtr> outputs;
  auto emit = [&](FramePtr f) { outputs.push_back(std::move(f)); };
  for (int i = 0; i < 7; ++i) {
    auto f = inPool.acquire();
    f->index = i;
    f->pts = i * 40;
    f->duration = 40;
    cv::Mat src = (i < 3) ? sceneA : sceneB;
    kern::floatToU8(src, f->y);
    f->u.setTo(128);
    f->v.setTo(128);
    proc->push(f, emit);
  }
  proc->flush(emit);

  ASSERT_EQ(outputs.size(), 7u);
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(outputs[i]->pts, i * 40);
    EXPECT_EQ(outputs[i]->segment, i < 3 ? 0 : 1);
  }
  EXPECT_TRUE(outputs[3]->sceneCut);
  EXPECT_EQ(proc->statistics().value("scene_cuts", 0), 1);
}

TEST(SimdTest, Avx2KernelsMatchScalarPrecision) {
  if (!compute::avx2KernelsAvailable()) GTEST_SKIP() << "AVX2 not available";
  cv::Mat img(129, 195, CV_32FC1);
  cv::randu(img, 0.f, 1.f);
  const auto k = kern::gaussianKernel(1.1f);

  cv::Mat dstAvx, dstScalar, tmp;
  kern::sepFilter(img, dstAvx, k, tmp);
  kern::sepFilterScalar(img, dstScalar, k, tmp);
  double maxDiffSep = 0.0;
  cv::minMaxLoc(cv::abs(dstAvx - dstScalar), nullptr, &maxDiffSep);
  EXPECT_LT(maxDiffSep, 1e-5);

  cv::Mat gAvx = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::Mat gScalar = cv::Mat::zeros(img.size(), CV_32FC1);
  kern::addBtvGradient(img, gAvx, 1, 0.7f, 0.002f);
  kern::addBtvGradientScalar(img, gScalar, 1, 0.7f, 0.002f);
  double maxDiffBtv = 0.0;
  cv::minMaxLoc(cv::abs(gAvx - gScalar), nullptr, &maxDiffBtv);
  EXPECT_LT(maxDiffBtv, 1e-5);
}
