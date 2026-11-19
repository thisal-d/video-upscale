#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "analysis/frame_analysis.hpp"
#include "compute/backend.hpp"
#include "compute/kernels.hpp"
#include "core/config.hpp"
#include "core/error.hpp"
#include "core/log.hpp"
#include "core/system.hpp"
#include "io/decoder.hpp"
#include "pipeline/pipeline.hpp"
#include "quality/benchmark_runner.hpp"

namespace {

void printUsage() {
  std::cout <<
R"(VideoUpscale v0.1.1 - Classical Computer-Vision Video Enhancement Engine

Usage:
  videoupscale upscale <input> --output <path> [options]
  videoupscale analyze <input> [--report <path>] [--max-frames <N>] [--json]
  videoupscale benchmark --dataset <dir> [--scale 2|3|4|8] [--output <report.json>]
                         [--methods m1,m2,...] [--compare-progressive] [--max-frames N]

Upscale options:
  -o, --output <path>      Output video path (.mp4, .mkv, .mov, ...) [required]
  -s, --scale <2|3|4|8>    Upscaling factor (default: 2)
  -u, --upscaler <name>    nearest | bilinear | bicubic | lanczos | edge | ibp | mfsr |
                           tyranox_crazy | tyranox_edge | tyranox_smooth (default: mfsr)
      --chroma <name>      Chroma upscaler: nearest|bilinear|bicubic|lanczos|edge|tyranox_crazy (default: bicubic)
      --progressive        Apply successive 2x stages for 4x/8x
      --denoise <mode>     off | auto | on (default: auto)
      --deblock <mode>     off | auto | on (default: auto)
      --sharpen <float>    Adaptive sharpening amount 0..3 (default: 0)
      --preset <name>      fast | balanced | quality
  -c, --config <path>      Load JSON configuration file
  -b, --backend <name>     auto | cpu | openmp | gpu (default: auto)
  -t, --threads <N>        Worker thread count (0 = hardware threads)
      --tile-size <px>     LR tile size for tiled reconstruction (default: 192)
      --codec <name>       FFmpeg video encoder (default: libx264)
      --crf <N>            Encoder quality CRF 0..51 (default: 18)
      --enc-preset <name>  Encoder speed preset (default: medium)
      --no-audio           Do not copy audio streams
      --copy-subtitles     Copy compatible subtitle streams
      --max-frames <N>     Stop after N frames (0 = all)
      --no-validate        Skip post-encode output verification
  -y, --overwrite          Overwrite existing output file
      --report <path>      Write JSON run report to file
      --json               Print JSON report to stdout
  -q, --quiet              Suppress progress and info logs
  -v, --verbose            Enable debug logging
  -h, --help               Show this help
      --version            Print version and system capabilities
)";
}

std::string needVal(int& i, int argc, char** argv, const std::string& flag) {
  if (i + 1 >= argc) vu::fail(vu::ErrorCode::InvalidArgument, "missing value for " + flag);
  return argv[++i];
}

int parseIntArg(const std::string& s, const std::string& flag) {
  try {
    std::size_t pos = 0;
    int v = std::stoi(s, &pos);
    if (pos != s.size()) throw std::invalid_argument("trailing characters");
    return v;
  } catch (...) {
    vu::fail(vu::ErrorCode::InvalidArgument, "invalid integer for " + flag + ": '" + s + "'");
  }
}

int cmdUpscale(int argc, char** argv) {
  vu::PipelineOptions opt;
  bool printJson = false;
  std::string reportPath;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      printUsage();
      return 0;
    } else if (a == "-o" || a == "--output") {
      opt.output = needVal(i, argc, argv, a);
    } else if (a == "-s" || a == "--scale") {
      opt.cfg.scale = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "-u" || a == "--upscaler") {
      std::string v = needVal(i, argc, argv, a);
      if (!vu::parseUpscaler(v, opt.cfg.upscaler))
        vu::fail(vu::ErrorCode::InvalidArgument, "unknown upscaler: " + v);
    } else if (a == "--chroma") {
      std::string v = needVal(i, argc, argv, a);
      if (!vu::parseUpscaler(v, opt.cfg.chromaUpscaler))
        vu::fail(vu::ErrorCode::InvalidArgument, "unknown chroma upscaler: " + v);
    } else if (a == "--progressive") {
      opt.cfg.progressive = true;
    } else if (a == "--denoise") {
      std::string v = needVal(i, argc, argv, a);
      if (!vu::parseToggle(v, opt.cfg.enhance.denoise))
        vu::fail(vu::ErrorCode::InvalidArgument, "invalid --denoise value: " + v);
    } else if (a == "--deblock") {
      std::string v = needVal(i, argc, argv, a);
      if (!vu::parseToggle(v, opt.cfg.enhance.deblock))
        vu::fail(vu::ErrorCode::InvalidArgument, "invalid --deblock value: " + v);
    } else if (a == "--sharpen") {
      try {
        opt.cfg.enhance.sharpen = std::stof(needVal(i, argc, argv, a));
      } catch (...) {
        vu::fail(vu::ErrorCode::InvalidArgument, "invalid float for --sharpen");
      }
    } else if (a == "--preset") {
      opt.cfg.applyPreset(needVal(i, argc, argv, a));
    } else if (a == "-c" || a == "--config") {
      opt.cfg = vu::loadConfigFile(needVal(i, argc, argv, a), opt.cfg);
    } else if (a == "-b" || a == "--backend") {
      std::string v = needVal(i, argc, argv, a);
      if (!vu::parseBackend(v, opt.cfg.backend))
        vu::fail(vu::ErrorCode::InvalidArgument, "unknown backend: " + v);
    } else if (a == "-t" || a == "--threads") {
      opt.cfg.threads = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "--tile-size") {
      opt.cfg.tileSize = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "--codec") {
      opt.cfg.encoder.codec = needVal(i, argc, argv, a);
    } else if (a == "--crf") {
      opt.cfg.encoder.crf = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "--enc-preset") {
      opt.cfg.encoder.preset = needVal(i, argc, argv, a);
    } else if (a == "--no-audio") {
      opt.cfg.encoder.copyAudio = false;
    } else if (a == "--copy-subtitles") {
      opt.cfg.encoder.copySubtitles = true;
    } else if (a == "--max-frames") {
      opt.maxFrames = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "--no-validate") {
      opt.validateOutput = false;
    } else if (a == "-y" || a == "--overwrite") {
      opt.overwrite = true;
    } else if (a == "--report") {
      reportPath = needVal(i, argc, argv, a);
    } else if (a == "--json") {
      printJson = true;
    } else if (a == "-q" || a == "--quiet") {
      vu::log::setLevel(vu::log::Level::Error);
      opt.progress = false;
    } else if (a == "-v" || a == "--verbose") {
      vu::log::setLevel(vu::log::Level::Debug);
    } else if (!a.empty() && a[0] != '-' && opt.input.empty()) {
      opt.input = a;
    } else {
      vu::fail(vu::ErrorCode::InvalidArgument, "unexpected argument: " + a);
    }
  }

  if (opt.input.empty()) vu::fail(vu::ErrorCode::InvalidArgument, "missing <input> path");
  if (opt.output.empty()) vu::fail(vu::ErrorCode::InvalidArgument, "missing --output <path>");

  vu::sys::installInterruptHandler();
  vu::sys::clearInterrupt();
  vu::RunReport r = vu::runUpscale(opt);

  nlohmann::json jr = vu::toJson(r);
  if (!reportPath.empty()) {
    std::ofstream f(reportPath);
    if (!f) vu::fail(vu::ErrorCode::OutputError, "cannot write report to '" + reportPath + "'");
    f << jr.dump(2) << '\n';
  }
  if (printJson) {
    std::cout << jr.dump(2) << '\n';
  } else {
    vu::log::info("done: {} -> {} ({}x{} -> {}x{}, {} frames, {:.2f}s, {:.2f} fps, peak {:.1f} MB, pool allocs={})",
                  opt.input, opt.output, r.input.width, r.input.height, r.outWidth, r.outHeight, r.framesEncoded,
                  r.wallSeconds, r.fps, r.peakMemoryBytes / (1024.0 * 1024.0), r.framePoolAllocations);
    if (!r.encoder.copiedStreams.empty()) {
      vu::log::info("preserved {} non-video stream(s) ({} packets)", r.encoder.copiedStreams.size(),
                    r.encoder.copiedPackets);
    }
  }
  return r.cancelled ? static_cast<int>(vu::ErrorCode::Cancelled) : 0;
}

int cmdAnalyze(int argc, char** argv) {
  std::string input, reportPath;
  bool printJson = false;
  int maxFrames = 120;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--report" || a == "-o") reportPath = needVal(i, argc, argv, a);
    else if (a == "--max-frames") maxFrames = parseIntArg(needVal(i, argc, argv, a), a);
    else if (a == "--json") printJson = true;
    else if (a == "-q" || a == "--quiet") vu::log::setLevel(vu::log::Level::Error);
    else if (a == "-v" || a == "--verbose") vu::log::setLevel(vu::log::Level::Debug);
    else if (!a.empty() && a[0] != '-' && input.empty()) input = a;
    else vu::fail(vu::ErrorCode::InvalidArgument, "unexpected argument: " + a);
  }
  if (input.empty()) vu::fail(vu::ErrorCode::InvalidArgument, "missing <input> path");

  vu::VideoDecoder dec;
  dec.open(input);
  const vu::VideoInfo& info = dec.info();

  vu::VideoFrame f;
  f.allocate(info.width, info.height);
  cv::Mat yf, prevY;
  int64_t analyzed = 0;
  double sumNoise = 0, sumBlur = 0, sumBlock = 0, sumSharp = 0, sumEdge = 0, sumTex = 0;
  std::vector<int64_t> sceneCuts;

  while ((maxFrames <= 0 || analyzed < maxFrames) && dec.read(f)) {
    vu::kern::u8ToFloat(f.y, yf);
    if (analyzed % 2 == 0) {
      auto m = vu::analysis::analyzeFrame(yf);
      sumNoise += m.noiseSigma;
      sumBlur += m.blurScore;
      sumBlock += m.blockiness;
      sumSharp += m.sharpness;
      sumEdge += m.edgeDensity;
      sumTex += m.textureRatio;
    }
    if (!prevY.empty()) {
      auto sc = vu::analysis::detectSceneCut(prevY, yf, 0.35f);
      if (sc.isCut) sceneCuts.push_back(f.index);
    }
    yf.copyTo(prevY);
    ++analyzed;
  }

  const double mSamples = static_cast<double>(std::max<int64_t>(1, (analyzed + 1) / 2));
  const double avgNoise = sumNoise / mSamples;
  const double avgBlur = sumBlur / mSamples;
  const double avgBlock = sumBlock / mSamples;

  std::vector<std::string> advisories;
  if (info.width < 160 || info.height < 120)
    advisories.push_back("very low spatial resolution; high-factor upscaling (4x/8x) may look synthetic");
  if (avgNoise > 0.025)
    advisories.push_back("elevated source noise detected; adaptive denoising recommended");
  if (avgBlock > 0.15)
    advisories.push_back("noticeable 8x8 block artifacts detected; deblocking recommended before reconstruction");
  if (avgBlur > 0.65)
    advisories.push_back("source appears heavily blurred; classical SR detail recovery will be limited");

  nlohmann::json j = {
      {"video", vu::toJson(info)},
      {"analysis",
       {{"frames_analyzed", analyzed},
        {"mean_noise_sigma", avgNoise},
        {"mean_blur_score", avgBlur},
        {"mean_blockiness", avgBlock},
        {"mean_sharpness", sumSharp / mSamples},
        {"mean_edge_density", sumEdge / mSamples},
        {"mean_texture_ratio", sumTex / mSamples},
        {"scene_cut_frames", sceneCuts},
        {"advisories", advisories}}},
  };
  if (!reportPath.empty()) {
    std::ofstream out(reportPath);
    if (!out) vu::fail(vu::ErrorCode::OutputError, "cannot write report to '" + reportPath + "'");
    out << j.dump(2) << '\n';
  }
  if (printJson || reportPath.empty()) {
    std::cout << j.dump(2) << '\n';
  }
  return 0;
}

int cmdBenchmark(int argc, char** argv) {
  vu::quality::BenchmarkOptions bo;
  bool printJson = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--dataset" || a == "-d") {
      bo.datasetDir = needVal(i, argc, argv, a);
    } else if (a == "--output" || a == "-o" || a == "--report") {
      bo.outputJsonPath = needVal(i, argc, argv, a);
    } else if (a == "--scale" || a == "-s") {
      bo.scale = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "--preset") {
      bo.preset = needVal(i, argc, argv, a);
    } else if (a == "--methods") {
      std::string s = needVal(i, argc, argv, a);
      std::size_t start = 0;
      while (start < s.size()) {
        auto comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        std::string tok = s.substr(start, comma - start);
        if (!tok.empty()) bo.methods.push_back(tok);
        start = comma + 1;
      }
    } else if (a == "--compare-progressive") {
      bo.compareProgressive = true;
    } else if (a == "--max-frames") {
      bo.maxFrames = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "--backend" || a == "-b") {
      std::string v = needVal(i, argc, argv, a);
      if (!vu::parseBackend(v, bo.backend))
        vu::fail(vu::ErrorCode::InvalidArgument, "unknown backend: " + v);
    } else if (a == "--threads" || a == "-t") {
      bo.threads = parseIntArg(needVal(i, argc, argv, a), a);
    } else if (a == "--keep-outputs") {
      bo.keepOutputs = true;
    } else if (a == "--json") {
      printJson = true;
    } else if (a == "-q" || a == "--quiet") {
      vu::log::setLevel(vu::log::Level::Error);
    } else if (a == "-v" || a == "--verbose") {
      vu::log::setLevel(vu::log::Level::Debug);
    } else {
      vu::fail(vu::ErrorCode::InvalidArgument, "unexpected benchmark argument: " + a);
    }
  }
  if (bo.datasetDir.empty()) vu::fail(vu::ErrorCode::InvalidArgument, "missing --dataset <dir>");
  if (bo.scale != 2 && bo.scale != 3 && bo.scale != 4 && bo.scale != 8)
    vu::fail(vu::ErrorCode::InvalidArgument, "scale must be 2, 3, 4 or 8");

  nlohmann::json res = vu::quality::runBenchmarkSuite(bo);
  if (printJson) std::cout << res.dump(2) << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      printUsage();
      return static_cast<int>(vu::ErrorCode::InvalidArgument);
    }
    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
      printUsage();
      return 0;
    }
    if (cmd == "--version" || cmd == "version") {
      std::cout << "VideoUpscale 0.1.1\n"
                << "CPU: " << vu::sys::cpuName() << " (" << vu::sys::hardwareThreads() << " threads)\n"
                << "OpenMP: " << (vu::compute::openmpAvailable() ? "yes" : "no")
                << " | AVX2: " << (vu::compute::avx2KernelsAvailable() ? "yes" : "no") << '\n';
      return 0;
    }
    if (cmd == "upscale") return cmdUpscale(argc, argv);
    if (cmd == "analyze" || cmd == "probe") return cmdAnalyze(argc, argv);
    if (cmd == "benchmark") return cmdBenchmark(argc, argv);
    vu::fail(vu::ErrorCode::InvalidArgument, "unknown command '" + cmd + "' (run --help)");
  } catch (const vu::Error& e) {
    vu::log::endProgress();
    vu::log::error("[{}] {}", vu::errorCodeName(e.code()), e.what());
    return static_cast<int>(e.code());
  } catch (const std::exception& e) {
    vu::log::endProgress();
    vu::log::error("{}", e.what());
    return static_cast<int>(vu::ErrorCode::Generic);
  }
}
