#include "core/log.hpp"

#include <chrono>
#include <mutex>

#include "core/error.hpp"

namespace vu {
const char* errorCodeName(ErrorCode c) noexcept {
  switch (c) {
    case ErrorCode::Ok: return "ok";
    case ErrorCode::Generic: return "error";
    case ErrorCode::InvalidArgument: return "invalid-argument";
    case ErrorCode::InputError: return "input-error";
    case ErrorCode::OutputError: return "output-error";
    case ErrorCode::Unsupported: return "unsupported";
    case ErrorCode::Cancelled: return "cancelled";
  }
  return "unknown";
}
}  // namespace vu

namespace vu::log {
namespace {
std::atomic<int> g_level{static_cast<int>(Level::Info)};
std::mutex g_mutex;
bool g_progressActive = false;
const auto g_start = std::chrono::steady_clock::now();

const char* tag(Level l) {
  switch (l) {
    case Level::Trace: return "trace";
    case Level::Debug: return "debug";
    case Level::Info: return "info ";
    case Level::Warn: return "WARN ";
    case Level::Error: return "ERROR";
    default: return "";
  }
}
}  // namespace

void setLevel(Level l) { g_level.store(static_cast<int>(l)); }
Level level() { return static_cast<Level>(g_level.load(std::memory_order_relaxed)); }

bool parseLevel(std::string_view s, Level& out) {
  if (s == "trace") out = Level::Trace;
  else if (s == "debug") out = Level::Debug;
  else if (s == "info") out = Level::Info;
  else if (s == "warn" || s == "warning") out = Level::Warn;
  else if (s == "error") out = Level::Error;
  else if (s == "off" || s == "quiet") out = Level::Off;
  else return false;
  return true;
}

void write(Level l, std::string_view msg) {
  const double t =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
  std::lock_guard lk(g_mutex);
  if (g_progressActive) {
    std::fputs("\n", stderr);
    g_progressActive = false;
  }
  std::fprintf(stderr, "[%8.3f] %s %.*s\n", t, tag(l), static_cast<int>(msg.size()), msg.data());
  std::fflush(stderr);
}

void progress(std::string_view line) {
  if (level() > Level::Info) return;
  std::lock_guard lk(g_mutex);
  std::fprintf(stderr, "\r%.*s   ", static_cast<int>(line.size()), line.data());
  std::fflush(stderr);
  g_progressActive = true;
}

void endProgress() {
  std::lock_guard lk(g_mutex);
  if (g_progressActive) {
    std::fputs("\n", stderr);
    g_progressActive = false;
  }
}

}  // namespace vu::log
