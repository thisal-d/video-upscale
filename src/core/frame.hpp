#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace vu {

// Rational number (FFmpeg-compatible semantics, no FFmpeg dependency in headers).
struct Rational {
  int num = 0;
  int den = 1;
  double toDouble() const { return den ? static_cast<double>(num) / den : 0.0; }
  bool valid() const { return num > 0 && den > 0; }
};

// Color metadata carried from input to output. Values are FFmpeg enum integers
// (AVColorPrimaries, AVColorTransferCharacteristic, AVColorSpace, AVColorRange,
// AVChromaLocation) kept as ints so that public headers stay FFmpeg-free.
struct ColorInfo {
  int primaries = 2;   // AVCOL_PRI_UNSPECIFIED
  int transfer = 2;    // AVCOL_TRC_UNSPECIFIED
  int space = 2;       // AVCOL_SPC_UNSPECIFIED
  int range = 0;       // AVCOL_RANGE_UNSPECIFIED
  int chromaLocation = 0;
};

struct AudioStreamInfo {
  int index = -1;
  std::string codec;
  int sampleRate = 0;
  int channels = 0;
  int64_t bitRate = 0;
};

// Metadata describing the input video stream.
struct VideoInfo {
  std::string path;
  std::string container;
  std::string codec;
  std::string pixelFormat;  // source pixel format name
  int width = 0;
  int height = 0;
  int bitDepth = 8;
  Rational timeBase;        // stream time base: pts units
  Rational avgFrameRate;
  Rational realFrameRate;   // r_frame_rate (lowest common rate)
  Rational sampleAspect{1, 1};
  int64_t frameCountEstimate = 0;  // 0 if unknown
  double durationSec = 0.0;
  int64_t startPts = 0;
  int64_t bitRate = 0;
  bool likelyVfr = false;
  int rotationDeg = 0;
  ColorInfo color;
  int videoStreamIndex = -1;
  std::vector<AudioStreamInfo> audio;
  int subtitleStreams = 0;
};

// Planar 8-bit YUV 4:2:0 frame (the engine's I/O representation).
// y is H x W, u/v are ceil(H/2) x ceil(W/2). All planes CV_8UC1.
struct VideoFrame {
  cv::Mat y, u, v;
  int64_t pts = 0;         // presentation timestamp in VideoInfo::timeBase units
  int64_t duration = 0;    // in timeBase units (0 if unknown)
  int64_t index = 0;       // decode-order index of the frame in presentation order
  int segment = 0;         // scene segment id (incremented at each detected scene cut)
  bool sceneCut = false;   // true if a scene cut starts at this frame

  int width() const { return y.cols; }
  int height() const { return y.rows; }
  void allocate(int w, int h) {
    y.create(h, w, CV_8UC1);
    u.create((h + 1) / 2, (w + 1) / 2, CV_8UC1);
    v.create((h + 1) / 2, (w + 1) / 2, CV_8UC1);
  }
  std::size_t bytes() const { return y.total() + u.total() + v.total(); }
};

using FramePtr = std::shared_ptr<VideoFrame>;
using ConstFramePtr = std::shared_ptr<const VideoFrame>;

}  // namespace vu
