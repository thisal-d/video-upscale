#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

#include <opencv2/imgproc.hpp>

#include "compute/kernels.hpp"
#include "core/frame.hpp"
#include "io/encoder.hpp"
#include "quality/benchmark_runner.hpp"
#include "quality/metrics.hpp"

namespace fs = std::filesystem;
using namespace vu;

TEST(QualityTest, PsnrAndSsimMatchExactAndDegradedIdentities) {
  cv::Mat a(64, 64, CV_32FC1);
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x)
      a.at<float>(y, x) = 0.2f + 0.6f * std::sin(x * 0.25f) * std::cos(y * 0.2f);

  EXPECT_DOUBLE_EQ(quality::computePsnr(a, a), 100.0);
  EXPECT_NEAR(quality::computeSsim(a, a), 1.0, 1e-6);

  cv::Mat b = a + 0.1f;
  EXPECT_NEAR(quality::computePsnr(a, b), 20.0, 0.05);
  EXPECT_LT(quality::computeSsim(a, b), 0.98);
  EXPECT_GT(quality::computeSsim(a, b), 0.5);
}

TEST(QualityTest, VideoPairEvaluationAndBenchmarkSuiteEndToEnd) {
  const fs::path dir = fs::temp_directory_path() / "vu_test_quality";
  fs::remove_all(dir);
  fs::create_directories(dir / "reference");
  fs::create_directories(dir / "degraded");

  const int rw = 128, rh = 96;
  const int dw = 64, dh = 48;
  const fs::path refVid = dir / "reference" / "clip1.mkv";
  const fs::path degVid = dir / "degraded" / "clip1.mkv";

  {
    VideoEncoder encRef, encDeg;
    EncoderSettings es;
    es.codec = "libx264";
    es.crf = 10;
    es.preset = "ultrafast";
    es.timeBase = Rational{1, 25};
    es.frameRate = Rational{25, 1};
    es.copyAudio = false;

    es.width = rw;
    es.height = rh;
    encRef.open(refVid.string(), es, nullptr, -1);

    es.width = dw;
    es.height = dh;
    encDeg.open(degVid.string(), es, nullptr, -1);

    VideoFrame fRef, fDeg;
    fRef.allocate(rw, rh);
    fDeg.allocate(dw, dh);
    for (int t = 0; t < 6; ++t) {
      cv::Mat hr(rh, rw, CV_32FC1);
      for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x)
          hr.at<float>(y, x) = std::clamp(0.45f + 0.35f * std::sin(0.15f * (x + t * 0.7f) + 0.1f * y), 0.f, 1.f);
      kern::floatToU8(hr, fRef.y);
      fRef.u.setTo(128);
      fRef.v.setTo(128);
      fRef.pts = t;
      fRef.duration = 1;
      encRef.write(fRef);

      cv::Mat lr;
      cv::resize(hr, lr, cv::Size(dw, dh), 0, 0, cv::INTER_AREA);
      kern::floatToU8(lr, fDeg.y);
      fDeg.u.setTo(128);
      fDeg.v.setTo(128);
      fDeg.pts = t;
      fDeg.duration = 1;
      encDeg.write(fDeg);
    }
    encRef.finish();
    encDeg.finish();
  }

  auto selfStats = quality::evaluateVideoPair(refVid.string(), refVid.string(), 4, 0);
  EXPECT_EQ(selfStats.framesCompared, 6);
  EXPECT_GT(selfStats.meanPsnr, 90.0);
  EXPECT_NEAR(selfStats.meanSsim, 1.0, 1e-4);
  EXPECT_NEAR(selfStats.temporalConsistencyError, 0.0, 1e-5);

  quality::BenchmarkOptions bo;
  bo.datasetDir = dir.string();
  bo.scale = 2;
  bo.methods = {"nearest", "bicubic", "mfsr"};
  bo.borderCrop = 4;
  auto report = quality::runBenchmarkSuite(bo);
  ASSERT_EQ(report.at("summary").size(), 3u);
  const double psnrNearest = report["summary"][0]["mean_psnr_db"].get<double>();
  const double psnrMfsr = report["summary"][2]["mean_psnr_db"].get<double>();
  EXPECT_GT(psnrMfsr, psnrNearest + 1.0);

  fs::remove_all(dir);
}
