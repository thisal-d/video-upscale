# VideoUpscale Progress Log

## 1. Requirements Checklist

### Milestone 1: Foundation & Video I/O
- [x] CMake 3.24+ / Ninja / MSVC C++20 project structure
- [x] Automated dependency fetch script (`scripts/fetch_deps.ps1`) for FFmpeg 8.1, OpenCV 4.14.0, GoogleTest 1.18.0, Google Benchmark 1.10.0, nlohmann/json 3.12.0
- [x] Core error codes (`ErrorCode`, `vu::Error`), thread-safe logger with progress line, `FramePool`, `BoundedQueue`, `ProcessingConfig` (presets + JSON + validation), Windows memory/CPUID/`SIGINT` utilities
- [x] FFmpeg streaming decoder (`VideoDecoder`) with planar YUV420p output, VFR PTS preservation/repair, color metadata & rotation extraction
- [x] FFmpeg streaming encoder (`VideoEncoder`) with `libx264`/fallback, color tagging, rotation copy, and audio/subtitle stream copy
- [x] 3-thread bounded streaming pipeline (`decode -> process -> encode`) + post-encode output validation

### Milestone 2: Baseline Upscaling
- [x] Nearest Neighbor, Bilinear, Bicubic, Lanczos, and structure-tensor Edge-Directed (`edge`) spatial upscalers
- [x] Support `2x`, `4x`, `8x` scaling factors (direct and progressive `2x` stages)

### Milestone 3: Analysis, Enhancement & Single-Frame IBP
- [x] Noise variance estimation (Immerkær MAD), blur estimation (Crete re-blur), 8×8 blockiness, Sobel edge density, texture ratio, scene-cut detection (`src/analysis/`)
- [x] Adaptive bilateral denoising, 4-tap 8×8 deblocking loop filter, adaptive sharpening with 3×3 min/max halo suppression (`src/enhance/`)
- [x] Tiled OpenMP Huber + BTV Iterative Back-Projection (`ibp`, `src/reconstruct/ibp.cpp`)

### Milestone 4: Motion Analysis
- [x] Bidirectional DIS optical flow (`ultrafast`/`fast`/`medium`)
- [x] Forward–backward cycle consistency, occlusion masking, photometric & geometric confidence weighting, sub-pixel warping (`src/motion/`)

### Milestone 5: Temporal Reconstruction
- [x] 5-frame initial temporal window (`I(t-2)..I(t+2)`), scene-cut isolation
- [x] Sub-pixel HR alignment, confidence-weighted temporal fusion, multi-frame Huber+BTV IBP, observation consistency check with single-frame IBP fallback, optional temporal refinement (`src/reconstruct/mfsr.cpp`)

### Milestone 6: Performance & SIMD Optimization
- [x] OpenMP non-nested tile/row parallelism & backend selection (`auto`, `cpu`, `openmp`, `gpu` rejected as unsupported)
- [x] Runtime-dispatched AVX2+FMA kernels (`src/compute/kernels_avx2.cpp`) + precomputed `CV_16SC2` remap tables
- [x] Google Benchmark micro-benchmarks (`bench/bench_kernels.cpp`)

### Milestone 7: Quality Metrics, Dataset & Validation
- [x] PSNR, double-precision SSIM, motion-compensated temporal warped RMSE, temporal consistency error (`src/quality/metrics.cpp`)
- [x] Controlled `dev` and `eval` degraded datasets (`scripts/generate_dataset.py`) with `clean`, `compressed`, `noisy`, `motionblur` at `2x`, `4x`, `8x`
- [x] Dataset benchmark runner (`videoupscale benchmark`) comparing all 7 methods and direct vs. progressive `4x`/`8x`

### Milestone 8: Release Preparation & Final Verification
- [x] Complete documentation (`README.md`, `ARCHITECTURE.md`, `ALGORITHMS.md`, `BUILDING.md`, `TESTING.md`, `PERFORMANCE.md`, `ROADMAP.md`, `CHANGELOG.md`, `LICENSE-NOTES.md`)
- [x] One-command clean build script (`scripts/build.ps1`) + CPack portable Windows ZIP (`VideoUpscale-0.1.1-win64.zip`)
- [x] Extra integration tests for `mfsr` + `--backend cpu` + error codes (`tests/test_core_io.cpp`)
- [x] Clean from-scratch build verification (`scripts/build.ps1 -Clean -Test -Package`) — 21/21 CTest tests passed in 16.82s
- [x] Final CLI command smoke verification (`2x`, `4x --preset quality`, `--backend cpu`, `analyze --report`, `--backend gpu` exit code `5`, `ffprobe` stream checks)
- [x] Git initialization and commits
- [x] `FINAL_REPORT.md`

---

## 2. Decisions & Assumptions Log
1. **Dependencies Strategy**: Pinned `FFmpeg n8.1` (BtbN GPL shared build), `OpenCV 4.14.0` (official Windows MSVC x64 release), `GoogleTest v1.18.0`, `Google Benchmark v1.10.0`, and `nlohmann/json v3.12.0` via `scripts/fetch_deps.ps1` rather than requiring a global `vcpkg` registry setup, keeping builds deterministic and self-contained inside `third_party/`.
2. **Internal Pixel Format**: Processing operates on planar 8-bit `yuv420p` (luma reconstructed with `edge`/`ibp`/`mfsr`, chroma upscaled with fast spatial `bicubic` by default); 10-bit+ or non-4:2:0 inputs are converted via `libswscale` on decode with a logged warning.
3. **Conservative Enhancement Defaults**: Pure spatial baselines (`nearest`, `bilinear`, `bicubic`, `lanczos`) leave `auto` denoise/deblock off unless explicitly enabled (`on`), keeping baseline benchmarks uncontaminated; `edge`/`ibp`/`mfsr` enable `auto` analysis-gated denoising and deblocking.
4. **Adaptive PSF & Regularization**: When `recon.psf_sigma <= 0` (`auto`), PSF width adapts to measured frame blur (`(0.26 + 0.16 * t) * scale`) and `btv_lambda` scales with measured noise/blockiness to avoid ringing on sharp area-downscaled footage while restoring blurred sources.
5. **No GPU Acceleration**: No CUDA/OpenCL backend is implemented; `--backend gpu` fails explicitly with `ErrorCode::Unsupported` (exit code `5`).
6. **Dataset Separation**: `benchmarks/datasets/dev` (synth_dev + Kingfisher crop) and `benchmarks/datasets/eval` (synth_eval + bcb0d8f0 crop) are kept strictly separate.
7. **Git Ignore Policy**: `example videos/`, `third_party/`, `build*/`, `benchmarks/datasets/`, and `test_outputs/` are excluded in `.gitignore` to keep the git history clean and compact.

---

## 3. Errors Encountered & Root-Cause Fixes
1. **`config.hpp(4): Cannot open include file 'nlohmann/json_fwd.hpp'`**: Single-header `nlohmann/json.hpp` does not ship `json_fwd.hpp`. Fixed by `#include <nlohmann/json.hpp>`.
2. **`IOTest.SyntheticEncodeDecodePreservesVfrTimestamps` Windows `remove_all` file lock**: `VideoDecoder` was still in scope holding `outPath` open during `fs::remove_all(dir)`. Scoped `VideoDecoder` in `{ ... }` before cleanup.
3. **Initial IBP (`huberDelta=0.010`, `btvLambda=0.0015`) underperformed Bicubic**: `0.010` clipped genuine edge residuals while 8-neighbor BTV summed to `~0.012`, overpowering the data term. Retuned `huberDelta=0.050`, `btvLambda=0.0010`, and `edge` gate `[0.035, 0.10]`.
4. **Initial MFSR ran at `0.35 fps` on 720p**: `mfsr.cpp` was calling single-threaded `sepFilter`/`addBtvGradient` on 3.7 MP frames 32–40 times per frame and rebuilding float `mapX`/`mapY` every iteration. Fixed by adding OpenMP-parallel `sepFilterParallel`/`addBtvGradientParallel`, AVX2+FMA `kernels_avx2.cpp` (`5.3×`–`6.4×` faster), and precomputed `CV_16SC2` fixed-point `remap` tables (`4×` end-to-end speedup to `~1.30–1.45 fps`).
5. **Periodic sine wave test ambiguity in `MfsrTest`**: Global high-frequency sine wave caused DIS optical flow aperture ambiguity. Replaced synthetic test scene with a multi-scale radial chirp + smooth envelope + sharp edges.
6. **`CV_32F` catastrophic cancellation in `computeSsim`**: On `synth_eval_noisy` (`edge`), `E[X^2] - E[X]^2` in `CV_32F` produced negative variance estimates and SSIM = `1.625`. Fixed by converting inputs to `CV_64F` and clamping variances `>= 0.0` before computing SSIM.
