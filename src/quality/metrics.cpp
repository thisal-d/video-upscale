#include "quality/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "compute/kernels.hpp"
#include "core/error.hpp"
#include "io/decoder.hpp"
#include "motion/optical_flow.hpp"

namespace vu::quality {

namespace {
cv::Mat cropBorder(const cv::Mat& m, int b) {
  if (b <= 0 || m.cols <= 2 * b || m.rows <= 2 * b) return m;
  return m(cv::Rect(b, b, m.cols - 2 * b, m.rows - 2 * b));
}
}  // namespace

double computePsnr(const cv::Mat& ref, const cv::Mat& test, int border) {
  CV_Assert(ref.type() == CV_32FC1 && test.type() == CV_32FC1 && ref.size() == test.size());
  const cv::Mat r = cropBorder(ref, border);
  const cv::Mat t = cropBorder(test, border);
  double mse = 0.0;
  for (int y = 0; y < r.rows; ++y) {
    const float* pr = r.ptr<float>(y);
    const float* pt = t.ptr<float>(y);
    for (int x = 0; x < r.cols; ++x) {
      const double d = static_cast<double>(pr[x]) - pt[x];
      mse += d * d;
    }
  }
  mse /= std::max<size_t>(1, r.total());
  if (mse <= 1e-12) return 100.0;
  return 10.0 * std::log10(1.0 / mse);
}

double computeSsim(const cv::Mat& ref, const cv::Mat& test, int border) {
  CV_Assert(ref.type() == CV_32FC1 && test.type() == CV_32FC1 && ref.size() == test.size());
  cv::Mat r, t;
  cropBorder(ref, border).convertTo(r, CV_64F);
  cropBorder(test, border).convertTo(t, CV_64F);

  constexpr double C1 = 0.01 * 0.01;
  constexpr double C2 = 0.03 * 0.03;

  cv::Mat mu1, mu2;
  cv::GaussianBlur(r, mu1, cv::Size(11, 11), 1.5, 0, cv::BORDER_REPLICATE);
  cv::GaussianBlur(t, mu2, cv::Size(11, 11), 1.5, 0, cv::BORDER_REPLICATE);

  cv::Mat mu1_2 = mu1.mul(mu1), mu2_2 = mu2.mul(mu2), mu1_mu2 = mu1.mul(mu2);
  cv::Mat r2 = r.mul(r), t2 = t.mul(t), rt = r.mul(t);

  cv::Mat sigma1_2, sigma2_2, sigma12;
  cv::GaussianBlur(r2, sigma1_2, cv::Size(11, 11), 1.5, 0, cv::BORDER_REPLICATE);
  cv::GaussianBlur(t2, sigma2_2, cv::Size(11, 11), 1.5, 0, cv::BORDER_REPLICATE);
  cv::GaussianBlur(rt, sigma12, cv::Size(11, 11), 1.5, 0, cv::BORDER_REPLICATE);

  sigma1_2 = cv::max(sigma1_2 - mu1_2, 0.0);
  sigma2_2 = cv::max(sigma2_2 - mu2_2, 0.0);
  sigma12 -= mu1_mu2;

  cv::Mat num = (2.0 * mu1_mu2 + C1).mul(2.0 * sigma12 + C2);
  cv::Mat den = (mu1_2 + mu2_2 + C1).mul(sigma1_2 + sigma2_2 + C2);
  cv::Mat ssimMap;
  cv::divide(num, den, ssimMap);
  return std::clamp(cv::mean(ssimMap)[0], -1.0, 1.0);
}

VideoQualityStats evaluateVideoPair(const std::string& referencePath, const std::string& testPath,
                                    int border, int maxFrames) {
  VideoDecoder decRef, decTest;
  decRef.open(referencePath);
  decTest.open(testPath);
  require(decRef.info().width == decTest.info().width && decRef.info().height == decTest.info().height,
          ErrorCode::InvalidArgument,
          "reference and test video dimensions do not match for quality evaluation");

  const int w = decRef.info().width, h = decRef.info().height;
  VideoFrame fRef, fTest;
  fRef.allocate(w, h);
  fTest.allocate(w, h);

  VideoQualityStats stats;
  stats.minPsnr = std::numeric_limits<double>::max();
  stats.minSsim = std::numeric_limits<double>::max();

  double sumPsnr = 0.0, sumSsim = 0.0;
  double sumWarpOut = 0.0, sumWarpRef = 0.0, sumTce = 0.0;
  int64_t temporalPairs = 0;

  cv::Mat refY, testY, prevRefY, prevTestY, refU8, testU8, prevRefU8, prevTestU8;
  auto flow = motion::createFlowEstimator("fast");
  TemporalParams tp;
  tp.flowPreset = "fast";

  while ((maxFrames <= 0 || stats.framesCompared < maxFrames) && decRef.read(fRef) && decTest.read(fTest)) {
    kern::u8ToFloat(fRef.y, refY);
    kern::u8ToFloat(fTest.y, testY);

    const double p = computePsnr(refY, testY, border);
    const double s = computeSsim(refY, testY, border);
    sumPsnr += p;
    sumSsim += s;
    stats.minPsnr = std::min(stats.minPsnr, p);
    stats.minSsim = std::min(stats.minSsim, s);

    if (!prevRefY.empty()) {
      fRef.y.copyTo(refU8);
      fTest.y.copyTo(testU8);
      // Temporal diff consistency: RMSE between frame-to-frame temporal derivatives
      cv::Mat dRef = refY - prevRefY;
      cv::Mat dTest = testY - prevTestY;
      cv::Mat dd = cropBorder(dTest - dRef, border);
      sumTce += std::sqrt(cv::mean(dd.mul(dd))[0]);

      // Warped temporal consistency using reference motion field (fair across methods)
      auto mrRef = motion::estimateMotion(refY, prevRefY, refU8, prevRefU8, tp, flow.get());
      cv::Mat warpedTest;
      motion::warpWithFlow(prevTestY, mrRef.flowFwd, warpedTest, cv::INTER_LINEAR);

      cv::Mat diffOut = cropBorder(testY - warpedTest, border);
      cv::Mat diffRef = cropBorder(refY - mrRef.alignedLr, border);
      cv::Mat mask = cropBorder(mrRef.occlusionMask, border);
      double validCnt = std::max(1.0, static_cast<double>(cv::countNonZero(mask)));

      cv::Mat maskF;
      mask.convertTo(maskF, CV_32F, 1.0 / 255.0);
      sumWarpOut += std::sqrt(cv::sum(diffOut.mul(diffOut).mul(maskF))[0] / validCnt);
      sumWarpRef += std::sqrt(cv::sum(diffRef.mul(diffRef).mul(maskF))[0] / validCnt);
      ++temporalPairs;
    }

    refY.copyTo(prevRefY);
    testY.copyTo(prevTestY);
    fRef.y.copyTo(prevRefU8);
    fTest.y.copyTo(prevTestU8);
    ++stats.framesCompared;
  }

  if (stats.framesCompared > 0) {
    stats.meanPsnr = sumPsnr / stats.framesCompared;
    stats.meanSsim = sumSsim / stats.framesCompared;
  } else {
    stats.minPsnr = 0.0;
    stats.minSsim = 0.0;
  }
  if (temporalPairs > 0) {
    stats.meanTemporalWarpedRmse = sumWarpOut / temporalPairs;
    stats.refTemporalWarpedRmse = sumWarpRef / temporalPairs;
    stats.temporalConsistencyError = sumTce / temporalPairs;
  }
  return stats;
}

}  // namespace vu::quality
