#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "core/frame.hpp"

namespace vu {

// Recycles VideoFrame buffers of a fixed geometry to avoid per-frame heap traffic.
// acquire() returns a shared_ptr whose deleter returns the frame to the pool
// (up to maxRetained idle frames are kept; the rest are freed). The pool itself
// never blocks: memory is bounded by the pipeline's bounded queues, while the pool
// only removes allocation churn. The pool must outlive all frames it hands out
// (enforced through a shared control block).
class FramePool {
 public:
  FramePool(int width, int height, std::size_t maxRetained = 16);
  ~FramePool();
  FramePool(const FramePool&) = delete;
  FramePool& operator=(const FramePool&) = delete;

  FramePtr acquire();

  std::size_t allocations() const;  // total frames ever allocated
  std::size_t liveFrames() const;   // frames currently checked out
  int width() const { return w_; }
  int height() const { return h_; }

 private:
  struct State {
    std::mutex m;
    std::vector<std::unique_ptr<VideoFrame>> free;
    std::size_t maxRetained = 16;
    std::atomic<std::size_t> allocations{0};
    std::atomic<std::size_t> live{0};
  };
  int w_, h_;
  std::shared_ptr<State> st_;
};

}  // namespace vu
