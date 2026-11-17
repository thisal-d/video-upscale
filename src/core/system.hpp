#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <string>

namespace vu {

class Stopwatch {
 public:
  Stopwatch() : t0_(clock::now()) {}
  void reset() { t0_ = clock::now(); }
  double seconds() const { return std::chrono::duration<double>(clock::now() - t0_).count(); }
  double ms() const { return seconds() * 1000.0; }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point t0_;
};

// Accumulates time spent in a stage (thread-safe, nanosecond resolution).
class StageTimer {
 public:
  void add(double sec) { ns_.fetch_add(static_cast<long long>(sec * 1e9), std::memory_order_relaxed); }
  double seconds() const { return ns_.load() * 1e-9; }

 private:
  std::atomic<long long> ns_{0};
};

namespace sys {
std::size_t peakWorkingSetBytes();
std::size_t currentWorkingSetBytes();
int hardwareThreads();
std::string cpuName();
bool cpuHasAvx2();

// Graceful cancellation: installs a SIGINT/Ctrl+C handler that sets a flag.
void installInterruptHandler();
bool interruptRequested();
void requestInterrupt();
void clearInterrupt();
}  // namespace sys

}  // namespace vu
