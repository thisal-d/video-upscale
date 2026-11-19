#include "core/config.hpp"

#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

#include "core/error.hpp"

namespace vu {

using nlohmann::json;

const char* toString(Upscaler u) {
  switch (u) {
    case Upscaler::Nearest: return "nearest";
    case Upscaler::Bilinear: return "bilinear";
    case Upscaler::Bicubic: return "bicubic";
    case Upscaler::Lanczos: return "lanczos";
    case Upscaler::EdgeAware: return "edge";
    case Upscaler::BackProjection: return "ibp";
    case Upscaler::MultiFrame: return "mfsr";
    case Upscaler::TyranoxCrazy: return "tyranox_crazy";
    case Upscaler::TyranoxEdge: return "tyranox_edge";
    case Upscaler::TyranoxSmooth: return "tyranox_smooth";
  }
  return "?";
}
const char* toString(Toggle t) {
  switch (t) {
    case Toggle::Off: return "off";
    case Toggle::Auto: return "auto";
    case Toggle::On: return "on";
  }
  return "?";
}
const char* toString(Backend b) {
  switch (b) {
    case Backend::Auto: return "auto";
    case Backend::Cpu: return "cpu";
    case Backend::OpenMP: return "openmp";
    case Backend::Gpu: return "gpu";
  }
  return "?";
}

bool parseUpscaler(const std::string& s, Upscaler& out) {
  for (Upscaler u : {Upscaler::Nearest, Upscaler::Bilinear, Upscaler::Bicubic, Upscaler::Lanczos,
                     Upscaler::EdgeAware, Upscaler::BackProjection, Upscaler::MultiFrame,
                     Upscaler::TyranoxCrazy, Upscaler::TyranoxEdge, Upscaler::TyranoxSmooth}) {
    if (s == toString(u)) {
      out = u;
      return true;
    }
  }
  if (s == "temporal" || s == "multiframe") { out = Upscaler::MultiFrame; return true; }
  if (s == "edge-aware" || s == "edgeaware") { out = Upscaler::EdgeAware; return true; }
  if (s == "backprojection") { out = Upscaler::BackProjection; return true; }
  if (s == "tyranox" || s == "tyranox-crazy" || s == "tyranox crazy" || s == "Tyranox - crazy") {
    out = Upscaler::TyranoxCrazy;
    return true;
  }
  if (s == "tyranox-edge") { out = Upscaler::TyranoxEdge; return true; }
  if (s == "tyranox-smooth") { out = Upscaler::TyranoxSmooth; return true; }
  return false;
}
bool parseToggle(const std::string& s, Toggle& out) {
  if (s == "off" || s == "false" || s == "0") out = Toggle::Off;
  else if (s == "auto") out = Toggle::Auto;
  else if (s == "on" || s == "true" || s == "1") out = Toggle::On;
  else return false;
  return true;
}
bool parseBackend(const std::string& s, Backend& out) {
  if (s == "auto") out = Backend::Auto;
  else if (s == "cpu") out = Backend::Cpu;
  else if (s == "openmp" || s == "omp") out = Backend::OpenMP;
  else if (s == "gpu" || s == "cuda" || s == "opencl") out = Backend::Gpu;
  else return false;
  return true;
}

void ProcessingConfig::applyPreset(const std::string& preset) {
  if (preset == "fast") {
    upscaler = Upscaler::EdgeAware;
    temporal.radius = 1;
    temporal.flowPreset = "ultrafast";
    recon.iterations = 3;
    enhance.denoise = Toggle::Auto;
    enhance.deblock = Toggle::Auto;
    enhance.sharpen = 0.0f;
    encoder.preset = "veryfast";
  } else if (preset == "balanced") {
    upscaler = Upscaler::MultiFrame;
    temporal.radius = 1;
    temporal.flowPreset = "fast";
    recon.iterations = 5;
    encoder.preset = "medium";
  } else if (preset == "quality") {
    upscaler = Upscaler::MultiFrame;
    temporal.radius = 2;
    temporal.flowPreset = "medium";
    recon.iterations = 8;
    encoder.preset = "slow";
    encoder.crf = 16;
  } else {
    fail(ErrorCode::InvalidArgument, "unknown preset '" + preset + "' (expected fast|balanced|quality)");
  }
}

void ProcessingConfig::validate() const {
  auto bad = [](const std::string& m) { fail(ErrorCode::InvalidArgument, m); };
  if (scale != 2 && scale != 3 && scale != 4 && scale != 8) bad("scale must be 2, 3, 4 or 8");
  if (threads < 0) bad("threads must be >= 0");
  if (tileSize < 32 || tileSize > 4096) bad("tile_size must be in [32, 4096]");
  if (queueDepth < 1 || queueDepth > 64) bad("queue_depth must be in [1, 64]");
  if (sceneCutThreshold <= 0.f || sceneCutThreshold > 1.f) bad("scene_cut_threshold must be in (0, 1]");
  if (recon.iterations < 0 || recon.iterations > 100) bad("recon.iterations must be in [0, 100]");
  if (recon.huberDelta <= 0.f) bad("recon.huber_delta must be > 0");
  if (recon.step <= 0.f || recon.step > 4.f) bad("recon.step must be in (0, 4]");
  if (recon.btvLambda < 0.f) bad("recon.btv_lambda must be >= 0");
  if (recon.btvRadius < 1 || recon.btvRadius > 3) bad("recon.btv_radius must be in [1, 3]");
  if (recon.edgeBlend < 0.f || recon.edgeBlend > 1.f) bad("recon.edge_blend must be in [0, 1]");
  if (temporal.radius < 0 || temporal.radius > 3) bad("temporal.radius must be in [0, 3]");
  if (temporal.flowPreset != "ultrafast" && temporal.flowPreset != "fast" && temporal.flowPreset != "medium")
    bad("temporal.flow_preset must be ultrafast|fast|medium");
  if (temporal.temporalRefine < 0.f || temporal.temporalRefine > 0.9f) bad("temporal.temporal_refine must be in [0, 0.9]");
  if (enhance.sharpen < 0.f || enhance.sharpen > 3.f) bad("enhance.sharpen must be in [0, 3]");
  if (enhance.denoiseStrength < 0.f || enhance.denoiseStrength > 4.f) bad("enhance.denoise_strength must be in [0, 4]");
  if (encoder.crf < 0 || encoder.crf > 51) bad("encoder.crf must be in [0, 51]");
  if (chromaUpscaler == Upscaler::MultiFrame || chromaUpscaler == Upscaler::BackProjection)
    bad("chroma_upscaler must be a spatial interpolator (nearest|bilinear|bicubic|lanczos|edge|tyranox_crazy|tyranox_edge|tyranox_smooth)");
}

void to_json(json& j, const ProcessingConfig& c) {
  j = json{
      {"scale", c.scale},
      {"upscaler", toString(c.upscaler)},
      {"chroma_upscaler", toString(c.chromaUpscaler)},
      {"progressive", c.progressive},
      {"backend", toString(c.backend)},
      {"threads", c.threads},
      {"tile_size", c.tileSize},
      {"queue_depth", c.queueDepth},
      {"scene_cut_threshold", c.sceneCutThreshold},
      {"decoder_threads", c.decoderThreads},
      {"recon",
       {{"iterations", c.recon.iterations},
        {"psf_sigma", c.recon.psfSigma},
        {"huber_delta", c.recon.huberDelta},
        {"step", c.recon.step},
        {"btv_lambda", c.recon.btvLambda},
        {"btv_radius", c.recon.btvRadius},
        {"btv_alpha", c.recon.btvAlpha},
        {"edge_blend", c.recon.edgeBlend}}},
      {"temporal",
       {{"radius", c.temporal.radius},
        {"flow_preset", c.temporal.flowPreset},
        {"fb_threshold", c.temporal.fbThreshold},
        {"photometric_sigma", c.temporal.photometricSigma},
        {"reject_threshold", c.temporal.rejectThreshold},
        {"min_confidence", c.temporal.minConfidence},
        {"temporal_refine", c.temporal.temporalRefine}}},
      {"enhance",
       {{"denoise", toString(c.enhance.denoise)},
        {"denoise_strength", c.enhance.denoiseStrength},
        {"deblock", toString(c.enhance.deblock)},
        {"deblock_strength", c.enhance.deblockStrength},
        {"sharpen", c.enhance.sharpen},
        {"sharpen_radius", c.enhance.sharpenRadius},
        {"halo_suppression", c.enhance.haloSuppression},
        {"halo_overshoot", c.enhance.haloOvershoot}}},
      {"encoder",
       {{"codec", c.encoder.codec},
        {"crf", c.encoder.crf},
        {"preset", c.encoder.preset},
        {"pix_fmt", c.encoder.pixFmt},
        {"copy_audio", c.encoder.copyAudio},
        {"copy_subtitles", c.encoder.copySubtitles},
        {"threads", c.encoder.threads}}},
  };
}

namespace {
template <class T>
void getIf(const json& j, const char* key, T& out, std::set<std::string>& seen) {
  if (auto it = j.find(key); it != j.end()) {
    seen.insert(key);
    try {
      out = it->get<T>();
    } catch (const std::exception& e) {
      fail(ErrorCode::InvalidArgument, std::string("config key '") + key + "': " + e.what());
    }
  }
}
void checkUnknown(const json& j, const std::set<std::string>& seen, const std::string& section) {
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!seen.count(it.key()))
      fail(ErrorCode::InvalidArgument, "unknown config key '" + section + it.key() + "'");
}
template <class E, class P>
void getEnum(const json& j, const char* key, E& out, P parse, std::set<std::string>& seen) {
  std::string s;
  bool present = j.contains(key);
  getIf(j, key, s, seen);
  if (present && !parse(s, out)) fail(ErrorCode::InvalidArgument, std::string("invalid value for '") + key + "': " + s);
}
}  // namespace

void applyJson(const json& j, ProcessingConfig& c) {
  if (!j.is_object()) fail(ErrorCode::InvalidArgument, "config root must be a JSON object");
  std::set<std::string> seen;
  if (j.contains("preset")) {
    seen.insert("preset");
    c.applyPreset(j.at("preset").get<std::string>());
  }
  getIf(j, "scale", c.scale, seen);
  getEnum(j, "upscaler", c.upscaler, parseUpscaler, seen);
  getEnum(j, "chroma_upscaler", c.chromaUpscaler, parseUpscaler, seen);
  getIf(j, "progressive", c.progressive, seen);
  getEnum(j, "backend", c.backend, parseBackend, seen);
  getIf(j, "threads", c.threads, seen);
  getIf(j, "tile_size", c.tileSize, seen);
  getIf(j, "queue_depth", c.queueDepth, seen);
  getIf(j, "scene_cut_threshold", c.sceneCutThreshold, seen);
  getIf(j, "decoder_threads", c.decoderThreads, seen);
  if (j.contains("$comment")) seen.insert("$comment");

  if (auto it = j.find("recon"); it != j.end()) {
    seen.insert("recon");
    std::set<std::string> s;
    const json& r = *it;
    getIf(r, "iterations", c.recon.iterations, s);
    getIf(r, "psf_sigma", c.recon.psfSigma, s);
    getIf(r, "huber_delta", c.recon.huberDelta, s);
    getIf(r, "step", c.recon.step, s);
    getIf(r, "btv_lambda", c.recon.btvLambda, s);
    getIf(r, "btv_radius", c.recon.btvRadius, s);
    getIf(r, "btv_alpha", c.recon.btvAlpha, s);
    getIf(r, "edge_blend", c.recon.edgeBlend, s);
    checkUnknown(r, s, "recon.");
  }
  if (auto it = j.find("temporal"); it != j.end()) {
    seen.insert("temporal");
    std::set<std::string> s;
    const json& t = *it;
    getIf(t, "radius", c.temporal.radius, s);
    getIf(t, "flow_preset", c.temporal.flowPreset, s);
    getIf(t, "fb_threshold", c.temporal.fbThreshold, s);
    getIf(t, "photometric_sigma", c.temporal.photometricSigma, s);
    getIf(t, "reject_threshold", c.temporal.rejectThreshold, s);
    getIf(t, "min_confidence", c.temporal.minConfidence, s);
    getIf(t, "temporal_refine", c.temporal.temporalRefine, s);
    checkUnknown(t, s, "temporal.");
  }
  if (auto it = j.find("enhance"); it != j.end()) {
    seen.insert("enhance");
    std::set<std::string> s;
    const json& e = *it;
    getEnum(e, "denoise", c.enhance.denoise, parseToggle, s);
    getIf(e, "denoise_strength", c.enhance.denoiseStrength, s);
    getEnum(e, "deblock", c.enhance.deblock, parseToggle, s);
    getIf(e, "deblock_strength", c.enhance.deblockStrength, s);
    getIf(e, "sharpen", c.enhance.sharpen, s);
    getIf(e, "sharpen_radius", c.enhance.sharpenRadius, s);
    getIf(e, "halo_suppression", c.enhance.haloSuppression, s);
    getIf(e, "halo_overshoot", c.enhance.haloOvershoot, s);
    checkUnknown(e, s, "enhance.");
  }
  if (auto it = j.find("encoder"); it != j.end()) {
    seen.insert("encoder");
    std::set<std::string> s;
    const json& e = *it;
    getIf(e, "codec", c.encoder.codec, s);
    getIf(e, "crf", c.encoder.crf, s);
    getIf(e, "preset", c.encoder.preset, s);
    getIf(e, "pix_fmt", c.encoder.pixFmt, s);
    getIf(e, "copy_audio", c.encoder.copyAudio, s);
    getIf(e, "copy_subtitles", c.encoder.copySubtitles, s);
    getIf(e, "threads", c.encoder.threads, s);
    checkUnknown(e, s, "encoder.");
  }
  checkUnknown(j, seen, "");
}

ProcessingConfig loadConfigFile(const std::string& path, ProcessingConfig base) {
  std::ifstream f(path);
  if (!f) fail(ErrorCode::InvalidArgument, "cannot open config file: " + path);
  json j;
  try {
    j = json::parse(f, nullptr, true, /*ignore_comments=*/true);
  } catch (const std::exception& e) {
    fail(ErrorCode::InvalidArgument, "config parse error in " + path + ": " + e.what());
  }
  applyJson(j, base);
  return base;
}

}  // namespace vu
