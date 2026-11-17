#pragma once
#include <functional>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "core/config.hpp"
#include "core/frame.hpp"
#include "core/frame_pool.hpp"

namespace vu {

// Streaming frame processor. Frames are pushed in presentation order; output frames
// are emitted in the same order (possibly delayed by a temporal look-ahead).
// Every pushed frame produces exactly one emitted frame by the time flush() returns.
class FrameProcessor {
 public:
  using Emit = std::function<void(FramePtr)>;
  virtual ~FrameProcessor() = default;
  virtual void begin(const VideoInfo& in, FramePool& outPool) = 0;
  virtual void push(ConstFramePtr f, const Emit& emit) = 0;
  virtual void flush(const Emit& emit) = 0;
  virtual std::string describe() const = 0;
  virtual nlohmann::json statistics() const { return nlohmann::json::object(); }
  virtual int lookahead() const { return 0; }
};

std::unique_ptr<FrameProcessor> createProcessor(const ProcessingConfig& cfg);

}  // namespace vu
