#include "pipeline/pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <mutex>
#include <thread>

#include "core/bounded_queue.hpp"
#include "core/error.hpp"
#include "core/frame_pool.hpp"
#include "core/log.hpp"
#include "core/system.hpp"
#include "io/decoder.hpp"
#include "pipeline/processor.hpp"

namespace fs = std::filesystem;

namespace vu {

nlohmann::json toJson(const VideoInfo& v) {
  nlohmann::json audio = nlohmann::json::array();
  for (const auto& a : v.audio) {
    audio.push_back({{"index", a.index},
                     {"codec", a.codec},
                     {"sample_rate", a.sampleRate},
                     {"channels", a.channels},
                     {"bit_rate", a.bitRate}});
  }
  return {
      {"path", v.path},
      {"container", v.container},
      {"codec", v.codec},
      {"pixel_format", v.pixelFormat},
      {"width", v.width},
      {"height", v.height},
      {"bit_depth", v.bitDepth},
      {"time_base", {v.timeBase.num, v.timeBase.den}},
      {"avg_frame_rate", {v.avgFrameRate.num, v.avgFrameRate.den}},
      {"real_frame_rate", {v.realFrameRate.num, v.realFrameRate.den}},
      {"fps", v.avgFrameRate.toDouble()},
      {"sample_aspect", {v.sampleAspect.num, v.sampleAspect.den}},
      {"frame_count_estimate", v.frameCountEstimate},
      {"duration_sec", v.durationSec},
      {"bit_rate", v.bitRate},
      {"likely_vfr", v.likelyVfr},
      {"rotation_deg", v.rotationDeg},
      {"color",
       {{"primaries", v.color.primaries},
        {"transfer", v.color.transfer},
        {"space", v.color.space},
        {"range", v.color.range},
        {"chroma_location", v.color.chromaLocation}}},
      {"audio_streams", audio},
      {"subtitle_streams", v.subtitleStreams},
  };
}

nlohmann::json toJson(const RunReport& r) {
  return {
      {"input", toJson(r.input)},
      {"output_width", r.outWidth},
      {"output_height", r.outHeight},
      {"frames_decoded", r.framesDecoded},
      {"frames_encoded", r.framesEncoded},
      {"timestamp_repairs", r.timestampRepairs},
      {"wall_seconds", r.wallSeconds},
      {"decode_seconds", r.decodeSeconds},
      {"process_seconds", r.processSeconds},
      {"encode_seconds", r.encodeSeconds},
      {"fps", r.fps},
      {"peak_memory_bytes", r.peakMemoryBytes},
      {"frame_pool_allocations", r.framePoolAllocations},
      {"cancelled", r.cancelled},
      {"processor", r.processor},
      {"backend",
       {{"requested", toString(r.backend.requested)},
        {"active", toString(r.backend.active)},
        {"threads", r.backend.threads},
        {"openmp", r.backend.openmp},
        {"avx2", r.backend.avx2}}},
      {"encoder",
       {{"codec", r.encoder.codecUsed},
        {"frames_written", r.encoder.framesWritten},
        {"video_bytes", r.encoder.videoBytes},
        {"copied_packets", r.encoder.copiedPackets},
        {"copied_streams", r.encoder.copiedStreams},
        {"dropped_streams", r.encoder.droppedStreams}}},
      {"processor_stats", r.processorStats},
      {"validation", r.validation},
      {"warnings", r.warnings},
  };
}

nlohmann::json validateOutputFile(const std::string& path, int expectW, int expectH, int64_t expectFrames,
                                  std::size_t expectAudioStreams) {
  nlohmann::json res = {{"ok", false}, {"path", path}};
  if (!fs::exists(path)) {
    res["error"] = "output file does not exist";
    return res;
  }
  const auto fileBytes = fs::file_size(path);
  res["file_bytes"] = fileBytes;
  if (fileBytes == 0) {
    res["error"] = "output file is empty";
    return res;
  }

  VideoDecoder dec;
  dec.open(path);
  const VideoInfo& info = dec.info();
  res["width"] = info.width;
  res["height"] = info.height;
  res["codec"] = info.codec;
  res["audio_streams"] = info.audio.size();

  if (info.width != expectW || info.height != expectH) {
    res["error"] = std::format("dimension mismatch: expected {}x{}, got {}x{}", expectW, expectH, info.width,
                               info.height);
    return res;
  }
  if (info.audio.size() != expectAudioStreams) {
    res["error"] = std::format("audio stream count mismatch: expected {}, got {}", expectAudioStreams,
                               info.audio.size());
    return res;
  }

  VideoFrame f;
  f.allocate(info.width, info.height);
  int64_t count = 0;
  int64_t prevPts = -1;
  bool monotonic = true;
  while (dec.read(f)) {
    if (count > 0 && f.pts <= prevPts) monotonic = false;
    prevPts = f.pts;
    ++count;
  }
  res["decoded_frames"] = count;
  res["monotonic_pts"] = monotonic;
  res["duration_sec"] = info.durationSec;

  if (count != expectFrames) {
    res["error"] = std::format("frame count mismatch: expected {}, got {}", expectFrames, count);
    return res;
  }
  if (!monotonic) {
    res["error"] = "output timestamps are not strictly monotonic";
    return res;
  }
  res["ok"] = true;
  return res;
}

RunReport runUpscale(const PipelineOptions& opt) {
  Stopwatch wall;
  opt.cfg.validate();
  if (opt.input.empty()) fail(ErrorCode::InvalidArgument, "input path is empty");
  if (opt.output.empty()) fail(ErrorCode::InvalidArgument, "output path is empty");
  if (!opt.overwrite && fs::exists(opt.output)) {
    fail(ErrorCode::OutputError,
         "output file already exists: '" + opt.output + "' (pass --overwrite or -y to replace)");
  }
  if (fs::path(opt.input) == fs::path(opt.output)) {
    fail(ErrorCode::InvalidArgument, "input and output paths must differ");
  }
  if (auto parent = fs::path(opt.output).parent_path(); !parent.empty()) {
    fs::create_directories(parent);
  }

  RunReport report;
  report.backend = compute::selectBackend(opt.cfg.backend, opt.cfg.threads);
  auto proc = createProcessor(opt.cfg);
  report.processor = proc->describe();

  VideoDecoder dec;
  dec.open(opt.input, DecoderOptions{opt.cfg.decoderThreads});
  const VideoInfo& inInfo = dec.info();
  report.input = inInfo;
  report.outWidth = inInfo.width * opt.cfg.scale;
  report.outHeight = inInfo.height * opt.cfg.scale;
  if (opt.cfg.encoder.pixFmt == "yuv420p" && (report.outWidth % 2 != 0 || report.outHeight % 2 != 0)) {
    fail(ErrorCode::InvalidArgument,
         std::format("yuv420p requires even output dimensions, but {}x{} at {}x scale produces {}x{} "
                     "(use an even-sized input or an even scale factor)",
                     inInfo.width, inInfo.height, opt.cfg.scale, report.outWidth, report.outHeight));
  }

  auto addWarn = [&](const std::string& w) {
    log::warn("{}", w);
    report.warnings.push_back(w);
  };

  if (inInfo.bitDepth > 8) {
    addWarn(std::format("source is {}-bit ({}); converting to 8-bit yuv420p", inInfo.bitDepth, inInfo.pixelFormat));
  }
  if (inInfo.width < 160 || inInfo.height < 120) {
    addWarn(std::format("very low source resolution ({}x{}); {}x upscaling may amplify compression artifacts",
                        inInfo.width, inInfo.height, opt.cfg.scale));
  } else if (opt.cfg.scale == 8) {
    addWarn("8x upscaling is experimental; classical reconstruction cannot hallucinate missing high-frequency detail");
  }
  if (inInfo.bitRate > 0 && inInfo.width > 0 && inInfo.height > 0 && inInfo.avgFrameRate.valid()) {
    const double bpp =
        inInfo.bitRate / (static_cast<double>(inInfo.width) * inInfo.height * inInfo.avgFrameRate.toDouble());
    if (bpp < 0.04) {
      addWarn(std::format("very low source bitrate ({:.3f} bits/pixel); heavy compression artifacts likely", bpp));
    }
  }
  if (inInfo.likelyVfr) {
    log::info("VFR source detected (avg {:.3f} fps vs r_frame_rate {:.3f} fps); preserving per-frame PTS",
              inInfo.avgFrameRate.toDouble(), inInfo.realFrameRate.toDouble());
  }

  VideoEncoder enc;
  EncoderSettings es;
  es.codec = opt.cfg.encoder.codec;
  es.crf = opt.cfg.encoder.crf;
  es.preset = opt.cfg.encoder.preset;
  es.pixFmt = opt.cfg.encoder.pixFmt;
  es.threads = opt.cfg.encoder.threads;
  es.width = report.outWidth;
  es.height = report.outHeight;
  es.timeBase = inInfo.timeBase;
  es.frameRate = inInfo.avgFrameRate;
  es.sampleAspect = inInfo.sampleAspect;
  es.color = inInfo.color;
  es.sourceHeight = inInfo.height;
  es.rotationDeg = inInfo.rotationDeg;
  es.copyAudio = opt.cfg.encoder.copyAudio;
  es.copySubtitles = opt.cfg.encoder.copySubtitles;
  es.comment = "Processed by VideoUpscale (" + report.processor + ")";
  enc.open(opt.output, es, dec.formatContext(), inInfo.videoStreamIndex);

  const std::size_t qCap = static_cast<std::size_t>(opt.cfg.queueDepth);
  const std::size_t poolKeep = qCap * 2 + static_cast<std::size_t>(proc->lookahead()) + 4;
  FramePool inPool(inInfo.width, inInfo.height, poolKeep);
  FramePool outPool(report.outWidth, report.outHeight, qCap + 4);
  proc->begin(inInfo, outPool);

  BoundedQueue<FramePtr> decQ(qCap);
  BoundedQueue<FramePtr> encQ(qCap);

  std::atomic<bool> abortFlag{false};
  std::atomic<int64_t> encodedCount{0};
  std::exception_ptr firstErr;
  std::mutex errMutex;
  auto recordError = [&](std::exception_ptr ep) {
    std::lock_guard lk(errMutex);
    if (!firstErr) firstErr = ep;
    abortFlag.store(true);
    decQ.close();
    encQ.close();
  };

  if (opt.cfg.encoder.copyAudio || opt.cfg.encoder.copySubtitles) {
    dec.setPacketCallback([&](AVPacket* pkt, int idx) {
      if (!abortFlag.load(std::memory_order_relaxed)) enc.copyPacket(pkt, idx);
    });
  }

  StageTimer tDec, tProc, tEnc;
  std::atomic<bool> cancelled{false};

  std::jthread decThread([&] {
    try {
      while (!abortFlag.load(std::memory_order_relaxed)) {
        if (sys::interruptRequested()) {
          cancelled.store(true);
          break;
        }
        if (opt.maxFrames > 0 && dec.framesDecoded() >= opt.maxFrames) break;
        FramePtr f = inPool.acquire();
        Stopwatch s;
        const bool got = dec.read(*f);
        tDec.add(s.seconds());
        if (!got) break;
        if (!decQ.push(std::move(f))) break;
      }
      decQ.close();
    } catch (...) {
      recordError(std::current_exception());
    }
  });

  std::jthread procThread([&] {
    try {
      auto emit = [&](FramePtr out) {
        if (!abortFlag.load(std::memory_order_relaxed)) encQ.push(std::move(out));
      };
      while (auto item = decQ.pop()) {
        if (abortFlag.load(std::memory_order_relaxed)) break;
        Stopwatch s;
        proc->push(*item, emit);
        tProc.add(s.seconds());
      }
      if (!abortFlag.load(std::memory_order_relaxed)) {
        Stopwatch s;
        proc->flush(emit);
        tProc.add(s.seconds());
      }
      encQ.close();
    } catch (...) {
      recordError(std::current_exception());
    }
  });

  std::jthread encThread([&] {
    try {
      auto lastReport = std::chrono::steady_clock::now();
      while (auto item = encQ.pop()) {
        if (abortFlag.load(std::memory_order_relaxed)) break;
        Stopwatch s;
        enc.write(**item);
        tEnc.add(s.seconds());
        const int64_t n = encodedCount.fetch_add(1) + 1;
        if (opt.progress) {
          const auto now = std::chrono::steady_clock::now();
          if (now - lastReport >= std::chrono::milliseconds(250)) {
            lastReport = now;
            const double elapsed = wall.seconds();
            const double fps = elapsed > 0 ? n / elapsed : 0.0;
            if (inInfo.frameCountEstimate > 0 && opt.maxFrames == 0) {
              const double pct = std::min(100.0, 100.0 * n / inInfo.frameCountEstimate);
              log::progress(std::format("frame {}/{} ({:5.1f}%) | {:5.2f} fps", n, inInfo.frameCountEstimate, pct,
                                        fps));
            } else {
              log::progress(std::format("frame {} | {:5.2f} fps", n, fps));
            }
          }
        }
      }
    } catch (...) {
      recordError(std::current_exception());
    }
  });

  decThread.join();
  procThread.join();
  encThread.join();
  if (opt.progress) log::endProgress();

  if (firstErr) {
    // Best-effort cleanup of incomplete output file on error.
    std::error_code ec;
    fs::remove(opt.output, ec);
    std::rethrow_exception(firstErr);
  }

  {
    Stopwatch s;
    enc.finish();
    tEnc.add(s.seconds());
  }

  report.framesDecoded = dec.framesDecoded();
  report.timestampRepairs = dec.timestampRepairs();
  report.encoder = enc.stats();
  report.framesEncoded = report.encoder.framesWritten;
  report.wallSeconds = wall.seconds();
  report.decodeSeconds = tDec.seconds();
  report.processSeconds = tProc.seconds();
  report.encodeSeconds = tEnc.seconds();
  report.fps = report.wallSeconds > 0 ? report.framesEncoded / report.wallSeconds : 0.0;
  report.peakMemoryBytes = sys::peakWorkingSetBytes();
  report.framePoolAllocations = inPool.allocations() + outPool.allocations();
  report.cancelled = cancelled.load();
  report.processorStats = proc->statistics();

  if (opt.validateOutput) {
    const std::size_t expectedAudio = report.encoder.copiedStreams.size();
    report.validation =
        validateOutputFile(opt.output, report.outWidth, report.outHeight, report.framesEncoded, expectedAudio);
    if (!report.validation.value("ok", false)) {
      fail(ErrorCode::OutputError,
           "output validation failed: " + report.validation.value("error", std::string("unknown")));
    }
  }

  if (report.cancelled) {
    log::warn("cancelled by user after {} frames; partial output finalized at '{}'", report.framesEncoded, opt.output);
  }
  return report;
}

}  // namespace vu
