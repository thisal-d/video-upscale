#include "io/ffmpeg_utils.hpp"

#include <cstdarg>
#include <cstdio>
#include <mutex>

#include "core/log.hpp"

namespace vu::ff {

std::string errStr(int err) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, buf, sizeof(buf));
  return buf;
}

namespace {
void logCallback(void* avcl, int lvl, const char* fmt, va_list vl) {
  if (lvl > av_log_get_level()) return;
  char line[1024];
  static int printPrefix = 1;
  static std::mutex m;
  std::lock_guard lk(m);
  av_log_format_line(avcl, lvl, fmt, vl, line, sizeof(line), &printPrefix);
  std::string s(line);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  if (s.empty()) return;
  if (lvl <= AV_LOG_ERROR) log::error("ffmpeg: {}", s);
  else if (lvl <= AV_LOG_WARNING) log::warn("ffmpeg: {}", s);
  else log::debug("ffmpeg: {}", s);
}
}  // namespace

void installLogBridge() {
  static std::once_flag once;
  std::call_once(once, [] {
    av_log_set_level(log::level() <= log::Level::Debug ? AV_LOG_VERBOSE : AV_LOG_WARNING);
    av_log_set_callback(logCallback);
  });
}

}  // namespace vu::ff
