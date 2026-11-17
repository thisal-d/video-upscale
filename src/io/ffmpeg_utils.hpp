#pragma once
// Internal FFmpeg helpers (not part of the public API: includes FFmpeg headers).
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/display.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <memory>
#include <string>

#include "core/error.hpp"
#include "core/frame.hpp"

namespace vu::ff {

std::string errStr(int err);
inline void check(int ret, ErrorCode code, const std::string& what) {
  if (ret < 0) fail(code, what + ": " + errStr(ret));
}

inline Rational toRational(AVRational r) { return Rational{r.num, r.den}; }
inline AVRational toAV(Rational r) { return AVRational{r.num, r.den}; }

struct FormatInputDeleter {
  void operator()(AVFormatContext* p) const { avformat_close_input(&p); }
};
struct CodecCtxDeleter {
  void operator()(AVCodecContext* p) const { avcodec_free_context(&p); }
};
struct FrameDeleter {
  void operator()(AVFrame* p) const { av_frame_free(&p); }
};
struct PacketDeleter {
  void operator()(AVPacket* p) const { av_packet_free(&p); }
};
struct SwsDeleter {
  void operator()(SwsContext* p) const { sws_freeContext(p); }
};

using FormatInputPtr = std::unique_ptr<AVFormatContext, FormatInputDeleter>;
using CodecCtxPtr = std::unique_ptr<AVCodecContext, CodecCtxDeleter>;
using AVFramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using AVPacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using SwsPtr = std::unique_ptr<SwsContext, SwsDeleter>;

inline AVFramePtr makeFrame() {
  AVFrame* f = av_frame_alloc();
  if (!f) fail(ErrorCode::Generic, "av_frame_alloc failed");
  return AVFramePtr(f);
}
inline AVPacketPtr makePacket() {
  AVPacket* p = av_packet_alloc();
  if (!p) fail(ErrorCode::Generic, "av_packet_alloc failed");
  return AVPacketPtr(p);
}

// Routes FFmpeg's internal log to vu::log (warnings and errors only by default).
void installLogBridge();

}  // namespace vu::ff
