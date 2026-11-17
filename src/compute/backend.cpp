#include "compute/backend.hpp"

#include <opencv2/core.hpp>

#include "core/error.hpp"
#include "core/log.hpp"
#include "core/system.hpp"

#ifdef VU_HAVE_OPENMP
#include <omp.h>
#endif

namespace vu::compute {

namespace {
BackendInfo g_active;
}

bool openmpAvailable() {
#ifdef VU_HAVE_OPENMP
  return true;
#else
  return false;
#endif
}

bool avx2KernelsAvailable() {
#ifdef VU_HAVE_AVX2_KERNELS
  static const bool ok = sys::cpuHasAvx2();
  return ok;
#else
  return false;
#endif
}

int maxThreads() { return g_active.threads > 0 ? g_active.threads : 1; }

BackendInfo selectBackend(Backend requested, int threads) {
  BackendInfo bi;
  bi.requested = requested;
  const int hw = sys::hardwareThreads();
  const int n = threads > 0 ? threads : hw;
  switch (requested) {
    case Backend::Gpu:
      fail(ErrorCode::Unsupported,
           "GPU backend is not implemented in this build (CPU/OpenMP only). Use --backend cpu or auto.");
    case Backend::Cpu:
      bi.active = Backend::Cpu;
      bi.threads = 1;
      bi.note = "single-threaded engine kernels";
      break;
    case Backend::OpenMP:
      if (!openmpAvailable())
        fail(ErrorCode::Unsupported, "OpenMP backend requested but this build has no OpenMP support");
      bi.active = Backend::OpenMP;
      bi.threads = n;
      break;
    case Backend::Auto:
      bi.active = openmpAvailable() ? Backend::OpenMP : Backend::Cpu;
      bi.threads = openmpAvailable() ? n : 1;
      break;
  }
  bi.openmp = bi.active == Backend::OpenMP;
#ifdef VU_HAVE_OPENMP
  omp_set_num_threads(bi.threads);
  omp_set_nested(0);  // never nest OpenMP regions (avoid oversubscription)
#endif
  // OpenCV parallel_for_ is used only from serial engine code (never inside OpenMP regions).
  cv::setNumThreads(bi.active == Backend::Cpu ? 1 : bi.threads);
  bi.avx2 = avx2KernelsAvailable();
  g_active = bi;
  log::debug("backend: requested={} active={} threads={} avx2={}", toString(requested), toString(bi.active),
             bi.threads, bi.avx2);
  return bi;
}

const BackendInfo& activeBackend() { return g_active; }

}  // namespace vu::compute
