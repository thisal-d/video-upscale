#include "core/frame_pool.hpp"

namespace vu {

FramePool::FramePool(int width, int height, std::size_t maxRetained)
    : w_(width), h_(height), st_(std::make_shared<State>()) {
  st_->maxRetained = maxRetained;
}

FramePool::~FramePool() = default;

FramePtr FramePool::acquire() {
  std::unique_ptr<VideoFrame> f;
  {
    std::lock_guard lk(st_->m);
    if (!st_->free.empty()) {
      f = std::move(st_->free.back());
      st_->free.pop_back();
    }
  }
  if (!f) {
    f = std::make_unique<VideoFrame>();
    f->allocate(w_, h_);
    st_->allocations.fetch_add(1, std::memory_order_relaxed);
  }
  f->pts = f->duration = f->index = 0;
  f->segment = 0;
  f->sceneCut = false;
  st_->live.fetch_add(1, std::memory_order_relaxed);
  std::weak_ptr<State> weak = st_;
  return FramePtr(f.release(), [weak](VideoFrame* p) {
    std::unique_ptr<VideoFrame> owned(p);
    if (auto s = weak.lock()) {
      s->live.fetch_sub(1, std::memory_order_relaxed);
      std::lock_guard lk(s->m);
      if (s->free.size() < s->maxRetained) s->free.push_back(std::move(owned));
    }
  });
}

std::size_t FramePool::allocations() const { return st_->allocations.load(); }
std::size_t FramePool::liveFrames() const { return st_->live.load(); }

}  // namespace vu
