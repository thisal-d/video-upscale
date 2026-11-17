#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "analysis/frame_analysis.hpp"
#include "compute/kernels.hpp"
#include "enhance/spatial_enhance.hpp"
#include "reconstruct/ibp.hpp"
#include "reconstruct/spatial.hpp"

using namespace vu;

namespace {
cv::Mat makeRichPattern(int w, int h) {
  cv::Mat img(h, w, CV_32FC1);
  for (int y = 0; y < h; ++y) {
    float* r = img.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      float dx = static_cast<float>(x - w / 2);
      float dy = static_cast<float>(y - h / 2);
      float rad = std::sqrt(dx * dx + dy * dy);
      float base = 0.15f + 0.35f * (static_cast<float>(x) / w);
      if (x * 0.7f + y * 0.5f > w * 0.55f) base += 0.35f;                 // diagonal step edge
      if (rad > 18.f && rad < 42.f) base += 0.18f * std::sin(rad * 0.6f); // concentric chirp
      if (x > 20 && x < 70 && y > 20 && y < 70) base = 0.85f;             // crisp box
      r[x] = std::clamp(base, 0.f, 1.f);
    }
  }
  return img;
}

double psnr01(const cv::Mat& a, const cv::Mat& b) {
  CV_Assert(a.size() == b.size() && a.type() == CV_32FC1 && b.type() == CV_32FC1);
  double mse = 0.0;
  for (int y = 0; y < a.rows; ++y) {
    const float* pa = a.ptr<float>(y);
    const float* pb = b.ptr<float>(y);
    for (int x = 0; x < a.cols; ++x) {
      double d = static_cast<double>(pa[x]) - pb[x];
      mse += d * d;
    }
  }
  mse /= static_cast<double>(a.rows) * a.cols;
  if (mse <= 1e-12) return 100.0;
  return 10.0 * std::log10(1.0 / mse);
}
}  // namespace

TEST(AnalysisTest, NoiseEstimatorTracksGaussianSigma) {
  cv::Mat clean = makeRichPattern(192, 192);
  EXPECT_LT(analysis::estimateNoiseSigma(clean), 0.005f);

  cv::RNG rng(42);
  for (float sigma : {0.015f, 0.035f}) {
    cv::Mat noise(clean.size(), CV_32FC1);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, sigma);
    cv::Mat noisy;
    cv::add(clean, noise, noisy);
    cv::threshold(noisy, noisy, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(noisy, noisy, 0.0, 0.0, cv::THRESH_TOZERO);
    float est = analysis::estimateNoiseSigma(noisy);
    EXPECT_NEAR(est, sigma, 0.006f);
  }
}

TEST(AnalysisTest, BlurAndBlockinessMetricsRespondMonotonically) {
  cv::Mat sharp = makeRichPattern(160, 160);
  cv::Mat blurred;
  cv::GaussianBlur(sharp, blurred, cv::Size(0, 0), 2.0);
  EXPECT_GT(analysis::estimateBlurScore(blurred), analysis::estimateBlurScore(sharp) + 0.15f);

  // Create synthetic 8x8 blocking artifacts (quantized 8x8 block means on a smooth gradient)
  cv::Mat grad(128, 128, CV_32FC1);
  for (int y = 0; y < 128; ++y)
    for (int x = 0; x < 128; ++x)
      grad.at<float>(y, x) = 0.1f + 0.8f * (x + y) / 256.f;

  cv::Mat blocky = grad.clone();
  for (int by = 0; by < 128; by += 8) {
    for (int bx = 0; bx < 128; bx += 8) {
      cv::Rect r(bx, by, 8, 8);
      float m = static_cast<float>(cv::mean(grad(r))[0]);
      blocky(r).setTo(m);
    }
  }
  EXPECT_LT(analysis::estimateBlockiness(grad), 0.05f);
  EXPECT_GT(analysis::estimateBlockiness(blocky), 0.50f);

  // Scene cut detection
  auto sameRes = analysis::detectSceneCut(sharp, blurred, 0.35f);
  EXPECT_FALSE(sameRes.isCut);
  cv::Mat inverted = 1.0f - sharp;
  auto cutRes = analysis::detectSceneCut(sharp, inverted, 0.35f);
  EXPECT_TRUE(cutRes.isCut);
}

TEST(EnhanceTest, AdaptiveDenoiseImprovesPsnrOnNoisyImage) {
  cv::Mat clean = makeRichPattern(192, 192);
  cv::Mat noise(clean.size(), CV_32FC1);
  cv::RNG rng(7);
  rng.fill(noise, cv::RNG::NORMAL, 0.0, 0.035);
  cv::Mat noisy;
  cv::max(0.f, cv::min(1.f, clean + noise), noisy);

  float estSigma = analysis::estimateNoiseSigma(noisy);
  cv::Mat denoised;
  ASSERT_TRUE(enhance::adaptiveDenoise(noisy, denoised, estSigma, 1.0f));
  EXPECT_GT(psnr01(clean, denoised), psnr01(clean, noisy) + 2.5);
}

TEST(EnhanceTest, DeblockReducesBlockinessAndImprovesPsnr) {
  cv::Mat clean(128, 128, CV_32FC1);
  for (int y = 0; y < 128; ++y)
    for (int x = 0; x < 128; ++x)
      clean.at<float>(y, x) = 0.15f + 0.65f * (x + 0.5f * y) / 192.f;

  cv::Mat blocky = clean.clone();
  for (int by = 0; by < 128; by += 8) {
    for (int bx = 0; bx < 128; bx += 8) {
      cv::Rect r(bx, by, 8, 8);
      float m = static_cast<float>(cv::mean(clean(r))[0]);
      blocky(r).setTo(m);
    }
  }
  cv::Mat deblocked;
  ASSERT_TRUE(enhance::deblockPlane(blocky, deblocked, 1.2f));
  EXPECT_LT(analysis::estimateBlockiness(deblocked), analysis::estimateBlockiness(blocky) * 0.6f);
  EXPECT_GT(psnr01(clean, deblocked), psnr01(clean, blocky));
}

TEST(EnhanceTest, HaloSuppressedSharpeningClampsOvershoot) {
  cv::Mat src = makeRichPattern(128, 128);
  cv::Mat sharp;
  const float overshoot = 0.015f;
  enhance::adaptiveSharpen(src, sharp, 1.5f, 1.0f, true, overshoot);

  cv::Mat mn, mx;
  cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::erode(src, mn, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
  cv::dilate(src, mx, k, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      EXPECT_GE(sharp.at<float>(y, x), std::max(0.f, mn.at<float>(y, x) - overshoot - 1e-5f));
      EXPECT_LE(sharp.at<float>(y, x), std::min(1.f, mx.at<float>(y, x) + overshoot + 1e-5f));
    }
  }
}

TEST(ReconstructionTest, IbpConvergesAndBeatsBicubic) {
  cv::Mat hrTruth = makeRichPattern(256, 256);
  const int scale = 2;
  const std::vector<float> psfK = kern::gaussianKernel(0.45f * scale);
  cv::Mat lr, blur, tmp;
  recon::downsampleBoxPsf(hrTruth, lr, scale, psfK, blur, tmp);

  cv::Mat bicubic, ibpUntiled, ibpTiled;
  recon::upscalePlane(lr, bicubic, hrTruth.size(), Upscaler::Bicubic);

  ReconstructionParams rp;
  rp.iterations = 8;
  rp.psfSigma = 0.45f * scale;
  rp.btvLambda = 0.0004f;
  auto stUntiled = recon::iterativeBackProjection(lr, ibpUntiled, scale, rp, 512);
  auto stTiled = recon::iterativeBackProjection(lr, ibpTiled, scale, rp, 48);

  EXPECT_LT(stUntiled.finalRmse, stUntiled.initialRmse * 0.5);
  EXPECT_GT(psnr01(hrTruth, ibpUntiled), psnr01(hrTruth, bicubic) + 0.5);
  EXPECT_GT(stTiled.tilesProcessed, 4);
  // Tiled interior should match untiled closely (PSNR between tiled & untiled > 45 dB)
  EXPECT_GT(psnr01(ibpUntiled, ibpTiled), 45.0);
}
