#pragma once
#include <string>

#include <nlohmann/json.hpp>

namespace vu {

enum class Upscaler { Nearest, Bilinear, Bicubic, Lanczos, EdgeAware, BackProjection, MultiFrame };
enum class Toggle { Off, Auto, On };
enum class Backend { Auto, Cpu, OpenMP, Gpu };

const char* toString(Upscaler u);
const char* toString(Toggle t);
const char* toString(Backend b);
bool parseUpscaler(const std::string& s, Upscaler& out);
bool parseToggle(const std::string& s, Toggle& out);
bool parseBackend(const std::string& s, Backend& out);

// Single-frame back-projection / reconstruction parameters.
struct ReconstructionParams {
  int iterations = 6;          // gradient iterations of the robust reconstruction
  float psfSigma = 0.0f;       // Gaussian PSF sigma in HR pixels; <=0 -> auto (0.28*scale)
  float huberDelta = 0.050f;   // Huber outlier threshold on [0,1] intensities
  float step = 1.0f;           // preconditioned gradient step
  float btvLambda = 0.0010f;   // bilateral total-variation regularization weight
  int btvRadius = 1;           // BTV neighbourhood radius (HR pixels)
  float btvAlpha = 0.7f;       // BTV spatial decay
  float edgeBlend = 0.75f;     // edge-directed interpolation strength for the initial estimate
};

// Multi-frame (temporal) parameters.
struct TemporalParams {
  int radius = 2;                    // frames on each side of the reference (2 -> 5-frame window)
  std::string flowPreset = "medium"; // DIS optical flow preset: ultrafast|fast|medium
  float fbThreshold = 0.5f;          // forward-backward error (LR px) above which observations are rejected
  float photometricSigma = 0.04f;    // scale of the residual-based confidence (0..1 intensities)
  float rejectThreshold = 0.12f;     // residual above which an observation is discarded (outlier)
  float minConfidence = 0.15f;       // observations below this confidence are dropped
  float temporalRefine = 0.0f;       // EXPERIMENTAL: blend weight with motion-compensated previous output
};

struct EnhanceParams {
  Toggle denoise = Toggle::Auto;
  float denoiseStrength = 1.0f;  // multiplier on the estimated noise sigma
  Toggle deblock = Toggle::Auto;
  float deblockStrength = 1.0f;
  float sharpen = 0.0f;          // adaptive sharpening amount (0 disables)
  float sharpenRadius = 1.0f;    // Gaussian sigma of the detail layer (HR px)
  bool haloSuppression = true;
  float haloOvershoot = 0.02f;   // allowed overshoot over local min/max (0..1)
};

struct EncoderParams {
  std::string codec = "libx264";
  int crf = 18;
  std::string preset = "medium";
  std::string pixFmt = "yuv420p";
  bool copyAudio = true;
  bool copySubtitles = false;
  int threads = 0;               // 0 -> automatic (bounded to avoid oversubscription)
};

struct ProcessingConfig {
  int scale = 2;
  Upscaler upscaler = Upscaler::MultiFrame;
  Upscaler chromaUpscaler = Upscaler::Bicubic;
  bool progressive = false;      // apply as successive 2x stages instead of direct
  Backend backend = Backend::Auto;
  int threads = 0;               // 0 -> all hardware threads
  int tileSize = 192;            // LR tile size for tiled reconstruction
  int queueDepth = 4;            // bounded queue capacity between pipeline stages
  float sceneCutThreshold = 0.35f;
  int decoderThreads = 0;        // 0 -> automatic
  ReconstructionParams recon;
  TemporalParams temporal;
  EnhanceParams enhance;
  EncoderParams encoder;

  void applyPreset(const std::string& preset);  // fast | balanced | quality
  void validate() const;                         // throws vu::Error(InvalidArgument)
};

void to_json(nlohmann::json& j, const ProcessingConfig& c);
// Applies keys present in j onto c; throws vu::Error on unknown keys / bad values.
void applyJson(const nlohmann::json& j, ProcessingConfig& c);
ProcessingConfig loadConfigFile(const std::string& path, ProcessingConfig base);

}  // namespace vu
