#pragma once
#include <atomic>
#include <cstdio>
#include <format>
#include <string>
#include <string_view>

namespace vu::log {

enum class Level : int { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Off = 5 };

void setLevel(Level l);
Level level();
bool parseLevel(std::string_view s, Level& out);
void write(Level l, std::string_view msg);

template <class... Args>
void trace(std::format_string<Args...> f, Args&&... a) {
  if (level() <= Level::Trace) write(Level::Trace, std::format(f, std::forward<Args>(a)...));
}
template <class... Args>
void debug(std::format_string<Args...> f, Args&&... a) {
  if (level() <= Level::Debug) write(Level::Debug, std::format(f, std::forward<Args>(a)...));
}
template <class... Args>
void info(std::format_string<Args...> f, Args&&... a) {
  if (level() <= Level::Info) write(Level::Info, std::format(f, std::forward<Args>(a)...));
}
template <class... Args>
void warn(std::format_string<Args...> f, Args&&... a) {
  if (level() <= Level::Warn) write(Level::Warn, std::format(f, std::forward<Args>(a)...));
}
template <class... Args>
void error(std::format_string<Args...> f, Args&&... a) {
  if (level() <= Level::Error) write(Level::Error, std::format(f, std::forward<Args>(a)...));
}

// Writes a transient progress line (carriage-return terminated) to stderr.
void progress(std::string_view line);
void endProgress();

}  // namespace vu::log
