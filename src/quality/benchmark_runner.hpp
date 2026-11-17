#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/config.hpp"

namespace vu::quality {

struct BenchmarkOptions {
  std::string datasetDir;
  std::string outputJsonPath;
  std::string tempDir;
  int scale = 2;
  std::string preset;                     // optional preset applied to ibp/mfsr
  std::vector<std::string> methods;       // e.g. nearest,bilinear,bicubic,lanczos,edge,ibp,mfsr
  bool compareProgressive = false;        // for scale > 2, also benchmark ibp+prog and mfsr+prog
  int maxFrames = 0;                      // 0 = all frames
  int borderCrop = 8;
  bool keepOutputs = false;
  Backend backend = Backend::Auto;
  int threads = 0;
};

nlohmann::json runBenchmarkSuite(const BenchmarkOptions& opt);

}  // namespace vu::quality
