#include <benchmark/benchmark.h>
#include <opencv2/core.hpp>

#include "compute/backend.hpp"
#include "compute/kernels.hpp"
#include "reconstruct/ibp.hpp"
#include "reconstruct/spatial.hpp"

static void BM_SpatialUpscale(benchmark::State& state, vu::Upscaler method, int scale) {
  vu::compute::selectBackend(vu::Backend::Auto, 0);
  cv::Mat lr(360, 640, CV_32FC1);
  cv::randu(lr, 0.f, 1.f);
  cv::Mat hr;
  for (auto _ : state) {
    vu::recon::upscalePlane(lr, hr, cv::Size(lr.cols * scale, lr.rows * scale), method);
    benchmark::DoNotOptimize(hr.data);
  }
  state.SetItemsProcessed(state.iterations());
}

static void BM_SepFilter(benchmark::State& state, bool useScalar) {
  vu::compute::selectBackend(vu::Backend::Auto, 0);
  cv::Mat img(720, 1280, CV_32FC1);
  cv::randu(img, 0.f, 1.f);
  const auto k = vu::kern::gaussianKernel(0.9f);
  cv::Mat dst, tmp;
  for (auto _ : state) {
    if (useScalar) vu::kern::sepFilterScalar(img, dst, k, tmp);
    else vu::kern::sepFilter(img, dst, k, tmp);
    benchmark::DoNotOptimize(dst.data);
  }
  state.SetItemsProcessed(state.iterations());
}

static void BM_BtvGradient(benchmark::State& state, bool useScalar) {
  vu::compute::selectBackend(vu::Backend::Auto, 0);
  cv::Mat img(720, 1280, CV_32FC1);
  cv::randu(img, 0.f, 1.f);
  cv::Mat grad = cv::Mat::zeros(img.size(), CV_32FC1);
  for (auto _ : state) {
    if (useScalar) vu::kern::addBtvGradientScalar(img, grad, 1, 0.7f, 0.0004f);
    else vu::kern::addBtvGradient(img, grad, 1, 0.7f, 0.0004f);
    benchmark::DoNotOptimize(grad.data);
  }
  state.SetItemsProcessed(state.iterations());
}

static void BM_IbpTiled(benchmark::State& state, int scale) {
  vu::compute::selectBackend(vu::Backend::Auto, 0);
  cv::Mat lr(360, 640, CV_32FC1);
  cv::randu(lr, 0.f, 1.f);
  cv::Mat hr;
  vu::ReconstructionParams rp;
  for (auto _ : state) {
    vu::recon::iterativeBackProjection(lr, hr, scale, rp, 192);
    benchmark::DoNotOptimize(hr.data);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_CAPTURE(BM_SpatialUpscale, Bicubic_2x, vu::Upscaler::Bicubic, 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_SpatialUpscale, Lanczos_2x, vu::Upscaler::Lanczos, 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_SpatialUpscale, EdgeAware_2x, vu::Upscaler::EdgeAware, 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_IbpTiled, IBP_2x_640x360, 2)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_SepFilter, Scalar_720p, true)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_SepFilter, AVX2_720p, false)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_BtvGradient, Scalar_720p, true)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_BtvGradient, AVX2_720p, false)->Unit(benchmark::kMillisecond);
