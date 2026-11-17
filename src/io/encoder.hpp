#pragma once
#include <memory>
#include <string>
#include <vector>

#include "core/frame.hpp"

struct AVPacket;
struct AVFormatContext;

namespace vu {

struct EncoderSettings {
  std::string codec = "libx264";
  int crf = 18;
  std::string preset = "medium";
  std::string pixFmt = "yuv420p";
  int threads = 0;            // 0 -> automatic
  int width = 0, height = 0;
  Rational timeBase;          // pts units of frames passed to write() (input stream time base)
  Rational frameRate;         // nominal frame rate (rate-control hint only; pts are authoritative)
  Rational sampleAspect{1, 1};
  ColorInfo color;            // copied from input; unspecified values are inferred from sourceHeight
  int sourceHeight = 0;
  int rotationDeg = 0;        // display rotation copied to output metadata (pixels are not rotated)
  bool copyAudio = true;
  bool copySubtitles = false;
  std::string comment;        // stored in container metadata
};

struct EncoderStats {
  int64_t framesWritten = 0;
  int64_t videoBytes = 0;
  int64_t copiedPackets = 0;
  std::vector<int> copiedStreams;   // input stream indices copied to output
  std::vector<int> droppedStreams;  // input streams that could not be stored
  std::string codecUsed;
};

// Encodes VideoFrames and muxes them, together with stream-copied audio (and
// optionally subtitle) packets from the input, into the output container.
// Video pts are passed through unchanged (rescaled only between time bases),
// which preserves variable frame rate timing and A/V sync.
// write() and copyPacket() may be called from different threads.
class VideoEncoder {
 public:
  VideoEncoder();
  ~VideoEncoder();
  VideoEncoder(const VideoEncoder&) = delete;
  VideoEncoder& operator=(const VideoEncoder&) = delete;

  // `input` (nullable) is the demuxer whose non-video streams are copied.
  void open(const std::string& path, const EncoderSettings& s, const AVFormatContext* input, int inputVideoStream);
  void write(const VideoFrame& f);
  void copyPacket(AVPacket* pkt, int inputStreamIndex);
  void finish();  // flushes encoder and writes trailer; output is valid afterwards
  bool isOpen() const;
  EncoderStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> d_;
};

}  // namespace vu
