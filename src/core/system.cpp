#include "core/system.hpp"

#include <csignal>
#include <cstring>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <intrin.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace vu::sys {

std::size_t peakWorkingSetBytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.PeakWorkingSetSize;
#endif
  return 0;
}

std::size_t currentWorkingSetBytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.WorkingSetSize;
#endif
  return 0;
}

int hardwareThreads() {
  unsigned n = std::thread::hardware_concurrency();
  return n ? static_cast<int>(n) : 1;
}

std::string cpuName() {
#ifdef _WIN32
  int regs[4];
  char brand[49] = {};
  __cpuid(regs, 0x80000000);
  if (static_cast<unsigned>(regs[0]) >= 0x80000004u) {
    for (int i = 0; i < 3; ++i) {
      __cpuid(regs, 0x80000002 + i);
      std::memcpy(brand + i * 16, regs, 16);
    }
    std::string s(brand);
    auto b = s.find_first_not_of(' ');
    return b == std::string::npos ? s : s.substr(b);
  }
#endif
  return "unknown";
}

bool cpuHasAvx2() {
#ifdef _WIN32
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuid(r, 1);
  const bool osxsave = (r[2] & (1 << 27)) != 0;
  const bool fma = (r[2] & (1 << 12)) != 0;
  if (!osxsave || !fma) return false;
  const unsigned long long xcr0 = _xgetbv(0);
  if ((xcr0 & 0x6) != 0x6) return false;  // OS saves XMM+YMM state
  __cpuidex(r, 7, 0);
  return (r[1] & (1 << 5)) != 0;
#else
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

namespace {
std::atomic<bool> g_interrupt{false};
std::atomic<int> g_interruptCount{0};

void onSignal(int) {
  g_interrupt.store(true);
  if (g_interruptCount.fetch_add(1) >= 2) std::_Exit(130);  // third Ctrl+C: hard exit
  std::signal(SIGINT, onSignal);
}
}  // namespace

void installInterruptHandler() { std::signal(SIGINT, onSignal); }
bool interruptRequested() { return g_interrupt.load(std::memory_order_relaxed); }
void requestInterrupt() { g_interrupt.store(true); }
void clearInterrupt() {
  g_interrupt.store(false);
  g_interruptCount.store(0);
}

}  // namespace vu::sys
