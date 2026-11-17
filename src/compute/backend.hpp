#pragma once
#include <string>

#include "core/config.hpp"

namespace vu::compute {

// Describes the compute backend actually in use after selection/fallback.
struct BackendInfo {
  Backend requested = Backend::Auto;
  Backend active = Backend::Cpu;
  int threads = 1;
  bool openmp = false;
  bool avx2 = false;       // AVX2 kernels compiled AND supported by this CPU
  std::string note;
};

// Selects and configures the backend (OpenMP thread count, OpenCV thread count).
// Backend::Gpu throws vu::Error(Unsupported): no GPU backend is implemented in this build.
// Backend::Auto resolves to OpenMP when available, otherwise single-threaded CPU.
BackendInfo selectBackend(Backend requested, int threads);
const BackendInfo& activeBackend();

int maxThreads();          // threads available to engine parallel loops
bool openmpAvailable();
bool avx2KernelsAvailable();

}  // namespace vu::compute
