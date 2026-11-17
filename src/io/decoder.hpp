#pragma once
#include <functional>
#include <memory>
#include <string>

#include "core/frame.hpp"

struct AVPacket;
struct AVFormatContext;

namespace vu {

struct DecoderOptions {
  int threads = 0;  // 0 -> automatic (bounded)
};

// Demuxes and decodes the best video stream of a media file into planar 8-bit
// YUV 4:2:0 VideoFrames, in presentation order, with original timestamps.
// Non-video packets can be forwarded (e.g. to a muxer for stream copy) through
// the packet callback, which receives packets in demux order.
class VideoDecoder {
 public:
  using PacketCallback = std::function<void(AVPacket* pkt, int streamIndex)>;

  VideoDecoder();
  ~VideoDecoder();
  VideoDecoder(const VideoDecoder&) = delete;
  VideoDecoder& operator=(const VideoDecoder&) = delete;

  void open(const std::string& path, const DecoderOptions& opt = {});
  const VideoInfo& info() const;

  // Optional: called for every packet that does not belong to the decoded video stream.
  void setPacketCallback(PacketCallback cb);

  // Decodes the next frame into `out` (planes must already be allocated with the
  // input geometry; FramePool does this). Returns false at end of stream.
  bool read(VideoFrame& out);

  // Underlying format context (for muxers that copy streams). Owned by the decoder.
  const AVFormatContext* formatContext() const;

  int64_t framesDecoded() const;
  int64_t timestampRepairs() const;  // number of frames whose pts had to be synthesized/fixed

 private:
  struct Impl;
  std::unique_ptr<Impl> d_;
};

// Probes a file without decoding frames (fast metadata read).
VideoInfo probeVideo(const std::string& path);

}  // namespace vu
