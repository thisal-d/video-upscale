#include "quality/benchmark_runner.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "core/error.hpp"
#include "core/log.hpp"
#include "pipeline/pipeline.hpp"
#include "quality/metrics.hpp"

namespace fs = std::filesystem;

namespace vu::quality {

namespace {

struct DatasetItem {
  std::string name;
  std::string degradation;
  std::string refPath;
  std::string degPath;
};

std::vector<DatasetItem> loadDataset(const fs::path& dir) {
  require(fs::exists(dir) && fs::is_directory(dir), ErrorCode::InvalidArgument,
          "dataset directory does not exist: " + dir.string());
  std::vector<DatasetItem> items;
  const fs::path manifest = dir / "manifest.json";
  if (fs::exists(manifest)) {
    std::ifstream f(manifest);
    nlohmann::json j = nlohmann::json::parse(f);
    for (const auto& entry : j.at("items")) {
      DatasetItem it;
      it.name = entry.at("name").get<std::string>();
      it.degradation = entry.value("degradation", "unknown");
      it.refPath = (dir / entry.at("reference").get<std::string>()).string();
      it.degPath = (dir / entry.at("degraded").get<std::string>()).string();
      items.push_back(std::move(it));
    }
  } else {
    // Auto-discover pairs: <dir>/reference/<stem>.mp4 + <dir>/degraded/<stem>.mp4
    const fs::path refDir = dir / "reference";
    const fs::path degDir = dir / "degraded";
    if (fs::exists(refDir) && fs::exists(degDir)) {
      for (const auto& e : fs::directory_iterator(degDir)) {
        if (!e.is_regular_file()) continue;
        fs::path r = refDir / e.path().filename();
        if (fs::exists(r)) {
          items.push_back({e.path().stem().string(), "pair", r.string(), e.path().string()});
        }
      }
    }
  }
  require(!items.empty(), ErrorCode::InvalidArgument,
          "no dataset pairs found in '" + dir.string() + "' (expected manifest.json or reference/+degraded/)");
  return items;
}

struct MethodSpec {
  std::string label;
  Upscaler upscaler;
  bool progressive = false;
};

std::vector<MethodSpec> buildMethodSpecs(const BenchmarkOptions& opt) {
  std::vector<std::string> names = opt.methods;
  if (names.empty()) {
    names = {"nearest", "bilinear", "bicubic", "lanczos", "edge", "ibp", "mfsr"};
  }
  std::vector<MethodSpec> specs;
  for (const auto& n : names) {
    Upscaler u;
    if (!parseUpscaler(n, u)) fail(ErrorCode::InvalidArgument, "unknown benchmark method: " + n);
    specs.push_back({toString(u), u, false});
    if (opt.compareProgressive && opt.scale > 2 &&
        (u == Upscaler::EdgeAware || u == Upscaler::BackProjection || u == Upscaler::MultiFrame)) {
      specs.push_back({std::string(toString(u)) + "_prog", u, true});
    }
  }
  return specs;
}

}  // namespace

nlohmann::json runBenchmarkSuite(const BenchmarkOptions& opt) {
  const fs::path dsDir(opt.datasetDir);
  auto items = loadDataset(dsDir);
  auto methods = buildMethodSpecs(opt);

  const fs::path workDir =
      opt.tempDir.empty() ? (fs::temp_directory_path() / "vu_bench_runs") : fs::path(opt.tempDir);
  fs::create_directories(workDir);

  nlohmann::json perRun = nlohmann::json::array();
  struct Agg {
    int count = 0;
    double psnr = 0, ssim = 0, warpRmse = 0, tce = 0, fps = 0, peakMb = 0;
  };
  std::map<std::string, Agg> agg;
  std::vector<std::string> methodOrder;
  for (const auto& m : methods) methodOrder.push_back(m.label);

  for (const auto& item : items) {
    for (const auto& m : methods) {
      PipelineOptions po;
      po.input = item.degPath;
      const fs::path outFile = workDir / (item.name + "_" + m.label + "_x" + std::to_string(opt.scale) + ".mp4");
      po.output = outFile.string();
      po.maxFrames = opt.maxFrames;
      po.progress = false;
      po.validateOutput = true;
      po.overwrite = true;

      if (!opt.preset.empty()) po.cfg.applyPreset(opt.preset);
      po.cfg.scale = opt.scale;
      po.cfg.upscaler = m.upscaler;
      po.cfg.progressive = m.progressive;
      po.cfg.backend = opt.backend;
      po.cfg.threads = opt.threads;
      // Use near-lossless CRF for quality benchmark measurement so encoder compression doesn't mask SR differences
      po.cfg.encoder.crf = 10;
      po.cfg.encoder.preset = "veryfast";
      po.cfg.encoder.copyAudio = false;

      log::info("[bench] {} | method={} scale={}x", item.name, m.label, opt.scale);
      RunReport rep = runUpscale(po);
      VideoQualityStats qs = evaluateVideoPair(item.refPath, outFile.string(), opt.borderCrop, opt.maxFrames);

      const double peakMb = rep.peakMemoryBytes / (1024.0 * 1024.0);
      perRun.push_back({
          {"item", item.name},
          {"degradation", item.degradation},
          {"method", m.label},
          {"scale", opt.scale},
          {"progressive", m.progressive},
          {"frames", qs.framesCompared},
          {"mean_psnr_db", qs.meanPsnr},
          {"min_psnr_db", qs.minPsnr},
          {"mean_ssim", qs.meanSsim},
          {"min_ssim", qs.minSsim},
          {"temporal_warped_rmse", qs.meanTemporalWarpedRmse},
          {"temporal_consistency_error", qs.temporalConsistencyError},
          {"fps", rep.fps},
          {"wall_seconds", rep.wallSeconds},
          {"peak_memory_mb", peakMb},
      });

      auto& a = agg[m.label];
      ++a.count;
      a.psnr += qs.meanPsnr;
      a.ssim += qs.meanSsim;
      a.warpRmse += qs.meanTemporalWarpedRmse;
      a.tce += qs.temporalConsistencyError;
      a.fps += rep.fps;
      a.peakMb = std::max(a.peakMb, peakMb);

      if (!opt.keepOutputs) {
        std::error_code ec;
        fs::remove(outFile, ec);
      }
    }
  }

  nlohmann::json summary = nlohmann::json::array();
  std::cout << "\n=== VideoUpscale Benchmark Summary (scale=" << opt.scale << "x, items=" << items.size() << ") ===\n";
  std::cout << std::left << std::setw(14) << "Method" << std::right << std::setw(10) << "PSNR(dB)"
            << std::setw(9) << "SSIM" << std::setw(11) << "WarpRMSE" << std::setw(11) << "TempErr"
            << std::setw(9) << "FPS" << std::setw(10) << "PeakMB" << '\n';
  std::cout << std::string(74, '-') << '\n';

  for (const auto& label : methodOrder) {
    const auto& a = agg[label];
    const double n = std::max(1, a.count);
    const double mPsnr = a.psnr / n;
    const double mSsim = a.ssim / n;
    const double mWarp = a.warpRmse / n;
    const double mTce = a.tce / n;
    const double mFps = a.fps / n;

    summary.push_back({
        {"method", label},
        {"items", a.count},
        {"mean_psnr_db", mPsnr},
        {"mean_ssim", mSsim},
        {"mean_temporal_warped_rmse", mWarp},
        {"mean_temporal_consistency_error", mTce},
        {"mean_fps", mFps},
        {"max_peak_memory_mb", a.peakMb},
    });

    std::cout << std::left << std::setw(14) << label << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << mPsnr << std::setprecision(4) << std::setw(9) << mSsim
              << std::setprecision(5) << std::setw(11) << mWarp << std::setw(11) << mTce
              << std::setprecision(2) << std::setw(9) << mFps << std::setprecision(1) << std::setw(10)
              << a.peakMb << '\n';
  }
  std::cout << '\n';

  nlohmann::json result = {
      {"dataset", opt.datasetDir},
      {"scale", opt.scale},
      {"items_count", items.size()},
      {"summary", summary},
      {"runs", perRun},
  };
  if (!opt.outputJsonPath.empty()) {
    if (auto p = fs::path(opt.outputJsonPath).parent_path(); !p.empty()) fs::create_directories(p);
    std::ofstream out(opt.outputJsonPath);
    if (!out) fail(ErrorCode::OutputError, "cannot write benchmark report to '" + opt.outputJsonPath + "'");
    out << result.dump(2) << '\n';
  }
  return result;
}

}  // namespace vu::quality
