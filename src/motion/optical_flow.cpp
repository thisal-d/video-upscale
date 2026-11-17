#include "motion/optical_flow.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "compute/backend.hpp"

namespace vu::motion {

cv::Ptr<cv::DISOpticalFlow> createFlowEstimator(const std::string& preset) {
  int id = cv::DISOpticalFlow::PRESET_MEDIUM;
  if (preset == "ultrafast") id = cv::DISOpticalFlow::PRESET_ULTRAFAST;
  else if (preset == "fast") id = cv::DISOpticalFlow::PRESET_FAST;
  auto dis = cv::DISOpticalFlow::create(id);
  dis->setUseSpatialPropagation(true);
  dis->setUseMeanNormalization(true);
  return dis;
}

void warpWithFlow(const cv::Mat& src, const cv::Mat& flow, cv::Mat& dst, int interp) {
  CV_Assert(src.type() == CV_32FC1 && flow.type() == CV_32FC2 && src.size() == flow.size());
  const int w = src.cols, h = src.rows;
  cv::Mat mapX(h, w, CV_32FC1), mapY(h, w, CV_32FC1);
  const int nthr = compute::maxThreads();
#pragma omp parallel for schedule(static) num_threads(nthr) if (nthr > 1 && h >= 128)
  for (int y = 0; y < h; ++y) {
    const cv::Vec2f* fl = flow.ptr<cv::Vec2f>(y);
    float* mx = mapX.ptr<float>(y);
    float* my = mapY.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      mx[x] = static_cast<float>(x) + fl[x][0];
      my[x] = static_cast<float>(y) + fl[x][1];
    }
  }
  cv::remap(src, dst, mapX, mapY, interp, cv::BORDER_REPLICATE);
}

void upscaleFlow(const cv::Mat& lrFlow, cv::Mat& hrFlow, cv::Size hrSize, int scale) {
  CV_Assert(lrFlow.type() == CV_32FC2);
  cv::resize(lrFlow, hrFlow, hrSize, 0, 0, cv::INTER_LINEAR);
  hrFlow *= static_cast<float>(scale);
}

namespace {
inline cv::Vec2f sampleFlowBilinear(const cv::Mat& flow, float x, float y) {
  const int w = flow.cols, h = flow.rows;
  x = std::clamp(x, 0.f, static_cast<float>(w - 1));
  y = std::clamp(y, 0.f, static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const float fx = x - x0, fy = y - y0;
  const cv::Vec2f& v00 = flow.at<cv::Vec2f>(y0, x0);
  const cv::Vec2f& v10 = flow.at<cv::Vec2f>(y0, x1);
  const cv::Vec2f& v01 = flow.at<cv::Vec2f>(y1, x0);
  const cv::Vec2f& v11 = flow.at<cv::Vec2f>(y1, x1);
  const cv::Vec2f a = v00 + fx * (v10 - v00);
  const cv::Vec2f b = v01 + fx * (v11 - v01);
  return a + fy * (b - a);
}
}  // namespace

MotionResult estimateMotion(const cv::Mat& refY, const cv::Mat& nbrY, const cv::Mat& refU8,
                            const cv::Mat& nbrU8, const TemporalParams& params,
                            cv::DISOpticalFlow* flowEst) {
  CV_Assert(refY.type() == CV_32FC1 && nbrY.type() == CV_32FC1 && refY.size() == nbrY.size());
  CV_Assert(refU8.type() == CV_8UC1 && nbrU8.type() == CV_8UC1 && refU8.size() == refY.size());

  cv::Ptr<cv::DISOpticalFlow> owned;
  if (!flowEst) {
    owned = createFlowEstimator(params.flowPreset);
    flowEst = owned.get();
  }

  MotionResult res;
  flowEst->calc(refU8, nbrU8, res.flowFwd);
  flowEst->calc(nbrU8, refU8, res.flowBwd);

  warpWithFlow(nbrY, res.flowFwd, res.alignedLr, cv::INTER_CUBIC);

  const int w = refY.cols, h = refY.rows;
  res.fbError.create(h, w, CV_32FC1);
  res.occlusionMask.create(h, w, CV_8UC1);
  res.confidence.create(h, w, CV_32FC1);

  const float fbThresh = std::max(0.05f, params.fbThreshold);
  const float fbSigma = 0.5f * fbThresh;
  const float invFbVar2 = 1.f / (2.f * fbSigma * fbSigma);
  const float photoSigma = std::max(1e-3f, params.photometricSigma);
  const float invPhotoVar2 = 1.f / (2.f * photoSigma * photoSigma);

  double sumConf = 0.0, sumFb = 0.0, sumMag = 0.0;
  int64_t occCount = 0;

  const int nthr = compute::maxThreads();
#pragma omp parallel for reduction(+ : sumConf, sumFb, sumMag, occCount) schedule(static) num_threads(nthr) \
    if (nthr > 1 && h >= 128)
  for (int y = 0; y < h; ++y) {
    const cv::Vec2f* fwd = res.flowFwd.ptr<cv::Vec2f>(y);
    const float* rRow = refY.ptr<float>(y);
    const float* aRow = res.alignedLr.ptr<float>(y);
    float* fbRow = res.fbError.ptr<float>(y);
    uint8_t* occRow = res.occlusionMask.ptr<uint8_t>(y);
    float* confRow = res.confidence.ptr<float>(y);

    for (int x = 0; x < w; ++x) {
      const float ux = fwd[x][0], uy = fwd[x][1];
      const float qx = static_cast<float>(x) + ux;
      const float qy = static_cast<float>(y) + uy;
      const float mag = std::sqrt(ux * ux + uy * uy);
      sumMag += mag;

      if (qx < 0.f || qy < 0.f || qx > static_cast<float>(w - 1) || qy > static_cast<float>(h - 1)) {
        fbRow[x] = fbThresh * 4.f;
        occRow[x] = 0;
        confRow[x] = 0.f;
        ++occCount;
        continue;
      }

      const cv::Vec2f b = sampleFlowBilinear(res.flowBwd, qx, qy);
      const float ex = ux + b[0], ey = uy + b[1];
      const float fb = std::sqrt(ex * ex + ey * ey);
      fbRow[x] = fb;
      sumFb += fb;

      const float photoErr = std::abs(aRow[x] - rRow[x]);
      if (fb > fbThresh || photoErr > params.rejectThreshold) {
        occRow[x] = 0;
        confRow[x] = 0.f;
        ++occCount;
        continue;
      }

      const float wFb = std::exp(-fb * fb * invFbVar2);
      const float wPh = std::exp(-photoErr * photoErr * invPhotoVar2);
      float c = wFb * wPh;
      if (c < params.minConfidence) {
        occRow[x] = 0;
        confRow[x] = 0.f;
        ++occCount;
      } else {
        occRow[x] = 255;
        confRow[x] = c;
        sumConf += c;
      }
    }
  }

  const double total = static_cast<double>(w) * h;
  res.stats.meanConfidence = static_cast<float>(sumConf / std::max(1.0, total));
  res.stats.occludedFraction = static_cast<float>(occCount / std::max(1.0, total));
  res.stats.meanFbError = static_cast<float>(sumFb / std::max(1.0, total));
  res.stats.meanMotionMag = static_cast<float>(sumMag / std::max(1.0, total));
  return res;
}

}  // namespace vu::motion
