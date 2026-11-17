#pragma once
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "compute/backend.hpp"
#include "core/config.hpp"
#include "core/frame.hpp"
#include "io/encoder.hpp"

namespace vu {

struct PipelineOptions {
  std::string input;
  std::string output;
  ProcessingConfig cfg;
  int64_t maxFrames = 0;     // 0 = all frames
  bool progress = true;
  bool validateOutput = true;
  bool overwrite = false;
};

struct RunReport {
  VideoInfo input;
  int outWidth = 0, outHeight = 0;
  int64_t framesDecoded = 0;
  int64_t framesEncoded = 0;
  int64_t timestampRepairs = 0;
  double wallSeconds = 0;
  double decodeSeconds = 0, processSeconds = 0, encodeSeconds = 0;
  double fps = 0;
  std::size_t peakMemoryBytes = 0;
  std::size_t framePoolAllocations = 0;
  bool cancelled = false;
  std::string processor;
  compute::BackendInfo backend;
  EncoderStats encoder;
  nlohmann::json processorStats;
  nlohmann::json validation;
  std::vector<std::string> warnings;
};

nlohmann::json toJson(const RunReport& r);
nlohmann::json toJson(const VideoInfo& v);

// Runs decode -> process -> encode with three threads connected by bounded queues.
// Throws vu::Error on failure (partial output is removed). On Ctrl+C the input is
// truncated, already-read frames are processed and the output is finalized
// (report.cancelled = true).
RunReport runUpscale(const PipelineOptions& opt);

// Validates an output file against expectations; returns a JSON object with "ok".
nlohmann::json validateOutputFile(const std::string& path, int expectW, int expectH, int64_t expectFrames,
                                  std::size_t expectAudioStreams);

}  // namespace vu
