#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace vu {

// Fixed-capacity blocking MPMC queue. push() blocks while full (back-pressure keeps
// memory bounded); pop() blocks while empty. close() wakes everybody: subsequent
// push() calls return false and pop() drains remaining items then returns nullopt.
template <class T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : cap_(capacity ? capacity : 1) {}

  bool push(T v) {
    std::unique_lock lk(m_);
    notFull_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    notEmpty_.notify_one();
    return true;
  }

  std::optional<T> pop() {
    std::unique_lock lk(m_);
    notEmpty_.wait(lk, [&] { return closed_ || !q_.empty(); });
    if (q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    notFull_.notify_one();
    return v;
  }

  void close() {
    std::lock_guard lk(m_);
    closed_ = true;
    notFull_.notify_all();
    notEmpty_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard lk(m_);
    return q_.size();
  }
  std::size_t capacity() const { return cap_; }

 private:
  const std::size_t cap_;
  mutable std::mutex m_;
  std::condition_variable notFull_, notEmpty_;
  std::deque<T> q_;
  bool closed_ = false;
};

}  // namespace vu
