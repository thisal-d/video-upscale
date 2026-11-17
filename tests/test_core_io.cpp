#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "compute/backend.hpp"
#include "compute/kernels.hpp"
#include "core/bounded_queue.hpp"
#include "core/config.hpp"
#include "core/error.hpp"
#include "core/frame_pool.hpp"
#include "io/decoder.hpp"
#include "io/encoder.hpp"
#include "pipeline/pipeline.hpp"
#include "reconstruct/spatial.hpp"

namespace fs = std::filesystem;
using namespace vu;

namespace {
fs::path makeTempDir(const std::string& name) {
  fs::path p = fs::temp_directory_path() / ("vu_test_" + name);
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

void fillPatternFrame(VideoFrame& f, int frameIdx, int w, int h) {
  for (int y = 0; y < h; ++y) {
    uint8_t* row = f.y.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      int v = ((x + frameIdx * 3) * 255) / std::max(1, w - 1);
      if (((x / 8 + y / 8 + frameIdx) % 2) == 0) v = (v + 40) % 255;
      row[x] = static_cast<uint8_t>(v);
    }
  }
  f.u.setTo(110);
  f.v.setTo(145);
}
}  // namespace

TEST(CoreTest, FramePoolReusesBuffers) {
  FramePool pool(64, 48, 4);
  const void* firstPtr = nullptr;
  {
    auto f1 = pool.acquire();
    EXPECT_EQ(f1->width(), 64);
    EXPECT_EQ(f1->height(), 48);
    EXPECT_EQ(pool.liveFrames(), 1u);
    firstPtr = f1->y.data;
  }
  EXPECT_EQ(pool.liveFrames(), 0u);
  auto f2 = pool.acquire();
  EXPECT_EQ(f2->y.data, firstPtr);
  EXPECT_EQ(pool.allocations(), 1u);
}

TEST(CoreTest, BoundedQueuePreservesOrderAndCloses) {
  BoundedQueue<int> q(2);
  EXPECT_TRUE(q.push(10));
  EXPECT_TRUE(q.push(20));
  q.close();
  EXPECT_FALSE(q.push(30));
  EXPECT_EQ(q.pop(), 10);
  EXPECT_EQ(q.pop(), 20);
  EXPECT_EQ(q.pop(), std::nullopt);
}

TEST(CoreTest, ConfigValidationAndJson) {
  ProcessingConfig c;
  EXPECT_NO_THROW(c.validate());
  c.scale = 3;
  EXPECT_THROW(c.validate(), Error);
  c.scale = 4;
  c.applyPreset("quality");
  EXPECT_EQ(c.recon.iterations, 8);
}

TEST(SpatialTest, AllBaselineUpscalersAndAllScales) {
  cv::Mat lr(32, 48, CV_32FC1);
  for (int y = 0; y < lr.rows; ++y)
    for (int x = 0; x < lr.cols; ++x)
      lr.at<float>(y, x) = (x > y) ? 0.85f : 0.15f;

  for (int scale : {2, 4, 8}) {
    for (Upscaler m : {Upscaler::Nearest, Upscaler::Bilinear, Upscaler::Bicubic, Upscaler::Lanczos,
                       Upscaler::EdgeAware}) {
      cv::Mat hr;
      recon::upscalePlane(lr, hr, cv::Size(lr.cols * scale, lr.rows * scale), m);
      EXPECT_EQ(hr.cols, lr.cols * scale);
      EXPECT_EQ(hr.rows, lr.rows * scale);
      double minV = 0, maxV = 0;
      cv::minMaxLoc(hr, &minV, &maxV);
      EXPECT_GE(minV, -0.25);
      EXPECT_LE(maxV, 1.25);
    }
  }
  // Progressive 4x test
  cv::Mat prog;
  recon::upscaleProgressive(lr, prog, 4, Upscaler::EdgeAware);
  EXPECT_EQ(prog.cols, lr.cols * 4);
  EXPECT_EQ(prog.rows, lr.rows * 4);
}

TEST(IOTest, SyntheticEncodeDecodePreservesVfrTimestamps) {
  const fs::path dir = makeTempDir("vfr");
  const fs::path srcPath = dir / "vfr_src.mkv";
  const fs::path outPath = dir / "vfr_out.mkv";

  const int w = 64, h = 48;
  const std::vector<int64_t> ptsList = {0, 33, 66, 150, 183, 250, 283, 400};
  {
    VideoEncoder enc;
    EncoderSettings es;
    es.codec = "libx264";
    es.crf = 12;
    es.preset = "ultrafast";
    es.width = w;
    es.height = h;
    es.timeBase = Rational{1, 1000};
    es.frameRate = Rational{30, 1};
    es.copyAudio = false;
    enc.open(srcPath.string(), es, nullptr, -1);

    VideoFrame f;
    f.allocate(w, h);
    for (size_t i = 0; i < ptsList.size(); ++i) {
      fillPatternFrame(f, static_cast<int>(i), w, h);
      f.pts = ptsList[i];
      f.duration = (i + 1 < ptsList.size()) ? (ptsList[i + 1] - ptsList[i]) : 33;
      enc.write(f);
    }
    enc.finish();
  }

  PipelineOptions opt;
  opt.input = srcPath.string();
  opt.output = outPath.string();
  opt.cfg.scale = 2;
  opt.cfg.upscaler = Upscaler::Bicubic;
  opt.cfg.encoder.preset = "ultrafast";
  opt.progress = false;
  opt.overwrite = true;

  RunReport rep = runUpscale(opt);
  EXPECT_EQ(rep.framesDecoded, static_cast<int64_t>(ptsList.size()));
  EXPECT_EQ(rep.framesEncoded, static_cast<int64_t>(ptsList.size()));
  EXPECT_EQ(rep.outWidth, w * 2);
  EXPECT_EQ(rep.outHeight, h * 2);
  EXPECT_TRUE(rep.validation.value("ok", false));

  // Verify exact PTS preservation through the pipeline
  {
    VideoDecoder dec;
    dec.open(outPath.string());
    VideoFrame outF;
    outF.allocate(dec.info().width, dec.info().height);
    std::vector<int64_t> gotPts;
    while (dec.read(outF)) {
      // Rescale back to ms if muxer changed timebase
      int64_t ms = (outF.pts * 1000LL * dec.info().timeBase.num) / dec.info().timeBase.den;
      gotPts.push_back(ms);
    }
    ASSERT_EQ(gotPts.size(), ptsList.size());
    for (size_t i = 0; i < ptsList.size(); ++i) {
      EXPECT_NEAR(static_cast<double>(gotPts[i]), static_cast<double>(ptsList[i]), 1.0);
    }
  }
  fs::remove_all(dir);
}

TEST(IOTest, RealVideoWithAudioPreservationAndValidation) {
  const fs::path sample = fs::path(VU_SOURCE_DIR) / "example videos" / "White-throated Kingfisher Video Animation.mp4";
  if (!fs::exists(sample)) GTEST_SKIP() << "Sample video not present";

  const fs::path dir = makeTempDir("real_audio");
  const fs::path outPath = dir / "kingfisher_2x.mp4";

  PipelineOptions opt;
  opt.input = sample.string();
  opt.output = outPath.string();
  opt.cfg.scale = 2;
  opt.cfg.upscaler = Upscaler::Bicubic;
  opt.cfg.encoder.preset = "ultrafast";
  opt.cfg.encoder.crf = 23;
  opt.maxFrames = 24;  // 1 second subset for fast CI test
  opt.progress = false;
  opt.overwrite = true;

  RunReport rep = runUpscale(opt);
  EXPECT_EQ(rep.framesDecoded, 24);
  EXPECT_EQ(rep.framesEncoded, 24);
  EXPECT_EQ(rep.outWidth, 1440);
  EXPECT_EQ(rep.outHeight, 2560);
  EXPECT_EQ(rep.encoder.copiedStreams.size(), 1u);
  EXPECT_GT(rep.encoder.copiedPackets, 0);
  EXPECT_TRUE(rep.validation.value("ok", false));

  VideoInfo outInfo = probeVideo(outPath.string());
  ASSERT_EQ(outInfo.audio.size(), 1u);
  EXPECT_EQ(outInfo.audio[0].codec, "aac");
  EXPECT_EQ(outInfo.audio[0].sampleRate, 44100);
  EXPECT_EQ(outInfo.audio[0].channels, 2);

  fs::remove_all(dir);
}

TEST(CoreTest, ErrorCodesAndUnsupportedGpuBackend) {
  try {
    compute::selectBackend(Backend::Gpu, 0);
    FAIL() << "Expected ErrorCode::Unsupported for Backend::Gpu";
  } catch (const Error& e) {
    EXPECT_EQ(e.code(), ErrorCode::Unsupported);
  }
  try {
    probeVideo("non_existent_input_video_xyz.mp4");
    FAIL() << "Expected ErrorCode::InputError for missing input";
  } catch (const Error& e) {
    EXPECT_EQ(e.code(), ErrorCode::InputError);
  }
}

TEST(IOTest, RealVideoMfsrAndCpuBackendEndToEnd) {
  const fs::path sample = fs::path(VU_SOURCE_DIR) / "example videos" / "bcb0d8f0-c8f9-4f5f-9705-57e887ace9aa.mp4";
  if (!fs::exists(sample)) GTEST_SKIP() << "Second sample video not present";

  const fs::path dir = makeTempDir("real_mfsr_cpu");
  const fs::path outPath = dir / "sample_mfsr_2x.mp4";

  PipelineOptions opt;
  opt.input = sample.string();
  opt.output = outPath.string();
  opt.cfg.scale = 2;
  opt.cfg.upscaler = Upscaler::MultiFrame;
  opt.cfg.temporal.radius = 2;
  opt.cfg.temporal.flowPreset = "fast";
  opt.cfg.recon.iterations = 3;
  opt.cfg.backend = Backend::Cpu;
  opt.cfg.encoder.preset = "ultrafast";
  opt.cfg.encoder.crf = 23;
  opt.maxFrames = 10;
  opt.progress = false;
  opt.overwrite = true;

  RunReport rep = runUpscale(opt);
  EXPECT_EQ(rep.backend.active, Backend::Cpu);
  EXPECT_EQ(rep.backend.threads, 1);
  EXPECT_EQ(rep.framesDecoded, 10);
  EXPECT_EQ(rep.framesEncoded, 10);
  EXPECT_EQ(rep.outWidth, 1440);
  EXPECT_EQ(rep.outHeight, 2560);
  EXPECT_EQ(rep.encoder.copiedStreams.size(), 1u);
  EXPECT_TRUE(rep.validation.value("ok", false));

  fs::remove_all(dir);
}
