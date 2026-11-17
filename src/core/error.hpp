#pragma once
// Error handling primitives for VideoUpscale.
// The library reports failures via vu::Error exceptions carrying an ErrorCode.
// The CLI maps ErrorCode to process exit codes (see cli/main.cpp and README).

#include <stdexcept>
#include <string>

namespace vu {

enum class ErrorCode : int {
  Ok = 0,
  Generic = 1,
  InvalidArgument = 2,  // bad CLI arguments / configuration
  InputError = 3,       // cannot open / demux / decode input
  OutputError = 4,      // cannot encode / mux / write output
  Unsupported = 5,      // feature or backend not available in this build
  Cancelled = 130,      // user interrupt (SIGINT), output finalized gracefully
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& msg) : std::runtime_error(msg), code_(code) {}
  ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

[[noreturn]] inline void fail(ErrorCode code, const std::string& msg) { throw Error(code, msg); }

inline void require(bool cond, ErrorCode code, const std::string& msg) {
  if (!cond) throw Error(code, msg);
}

const char* errorCodeName(ErrorCode c) noexcept;

}  // namespace vu
