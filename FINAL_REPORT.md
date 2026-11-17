# VideoUpscale — Final Implementation Report

## 1. Summary of What Was Built

**VideoUpscale (`v0.1.0`)** is a CPU-first classical computer-vision video super-resolution and enhancement engine implemented in **C++20** with **CMake + Ninja**, **FFmpeg n8.1**, **OpenCV 4.14.0**, **OpenMP 2.0**, **AVX2+FMA**, **GoogleTest v1.18.0**, **Google Benchmark v1.10.0**, and **nlohmann/json v3.12.0**.

### Subsystems (`src/`)
- **Core (`src/core/`)**: `ErrorCode`/`vu::Error`, thread-safe logger with stderr progress reporting (`log.cpp`), `FramePool` buffer recycling (`frame_pool.cpp`), fixed-capacity MPMC `BoundedQueue` (`bounded_queue.hpp`), JSON/preset `ProcessingConfig` with validation (`config.cpp`), Windows working-set memory & CPUID/AVX2 detection + `SIGINT` handler (`system.cpp`).
- **Video I/O (`src/io/`)**: Streaming `VideoDecoder` (demux/decode to planar 8-bit `yuv420p`, VFR PTS preservation & monotonic repair, color/rotation metadata extraction) and `VideoEncoder` (`libx264`/fallback codecs, explicit color tagging, rotation matrix copy, thread-safe audio & subtitle stream copy, safe trailer finalization).
- **Analysis (`src/analysis/`)**: Immerkær MAD noise variance estimator, Crete re-blurring blur score, 8×8 DCT boundary blockiness metric, Sobel edge density, texture ratio, and histogram+MAD scene-cut detection (`frame_analysis.cpp`).
- **Enhancement (`src/enhance/`)**: Edge-preserving adaptive bilateral denoising, 4-tap 8×8 deblocking loop filter, and gradient-gated adaptive sharpening with 3×3 min/max halo suppression (`spatial_enhance.cpp`).
- **Motion (`src/motion/`)**: Bidirectional DIS optical flow (`ultrafast`/`fast`/`medium`), forward–backward cycle consistency error, occlusion masking, photometric + geometric confidence weighting, and sub-pixel warping (`optical_flow.cpp`).
- **Reconstruction (`src/reconstruct/`)**:
  - Spatial baselines (`spatial.cpp`): `nearest`, `bilinear`, `bicubic`, `lanczos`, and structure-tensor anisotropic `edge` interpolation at `2×`, `4×`, and `8×` (direct and progressive).
  - Single-frame `ibp` (`ibp.cpp`): Tiled OpenMP Huber + Bilateral Total Variation (BTV) Iterative Back-Projection.
  - Multi-frame `mfsr` (`mfsr.cpp`): 5-frame sliding window (`I(t-2)..I(t+2)`), scene-cut isolation, precomputed `CV_16SC2` sub-pixel warping, confidence-weighted temporal fusion, multi-frame Huber+BTV IBP, observation consistency check with single-frame IBP fallback, and optional experimental `temporal_refine`.
- **Compute & SIMD (`src/compute/`)**: `auto`/`cpu`/`openmp` backend selection with `omp_set_nested(0)` (`backend.cpp`), plus runtime-dispatched AVX2+FMA kernels for separable Gaussian convolution, BTV gradients, and u8↔float conversion (`kernels_avx2.cpp`).
- **Quality & Benchmarking (`src/quality/`)**: PSNR, double-precision Wang et al. SSIM, motion-compensated temporal warped RMSE, temporal consistency error (`metrics.cpp`), and dataset benchmark runner (`benchmark_runner.cpp`).
- **Pipeline & CLI (`src/pipeline/`, `src/cli/`)**: 3-thread bounded streaming pipeline (`decode → process → encode`) with post-encode validation (`pipeline.cpp`) and CLI supporting `upscale`, `analyze`, `benchmark`, JSON reporting, presets, config files, and exit codes (`main.cpp`).

---

### Constraints Followed
- **Zero neural networks / pretrained AI / cloud APIs**: All reconstruction and enhancement algorithms are strictly mathematical and classical signal-processing methods.
- **CPU-first**: Works fully without a GPU (`--backend gpu` explicitly returns exit code `5`).

---

## 2. How to Build and Run

### Fetch Dependencies & Clean Build + Test + Package (Windows PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_deps.ps1
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Clean -Test -Package
```
Artifacts produced:
- CLI: `build/src/videoupscale.exe`
- Tests: `build/tests/vu_tests.exe`
- Micro-benchmarks: `build/bench/vu_bench.exe`
- Portable release ZIP: `build/packages/VideoUpscale-0.1.0-win64.zip`

### CLI Examples
```powershell
.\build\src\videoupscale.exe upscale input.mp4 --output out_2x.mp4 --scale 2
.\build\src\videoupscale.exe upscale input.mp4 --output out_4x.mp4 --scale 4 --preset quality
.\build\src\videoupscale.exe upscale input.mp4 --output out_cpu.mp4 --backend cpu
.\build\src\videoupscale.exe analyze input.mp4 --report analysis.json
.\build\src\videoupscale.exe benchmark --dataset benchmarks/datasets/eval --scale 2 --output benchmarks/results/eval_2x.json
```

---

## 3. Observed Test & Benchmark Results

### Automated Tests (`ctest --test-dir build --output-on-failure`)
- **21 / 21 tests passed (100%) in 16.82 s** after a clean from-scratch build (`scripts/build.ps1 -Clean -Test -Package`).
- Covers `CoreTest` (4 tests), `SpatialTest` (1 test), `IOTest` (3 tests including VFR MKV PTS round-trip and real HEVC+AAC & H.264+AAC MP4 end-to-end upscales), `AnalysisTest` (2 tests), `EnhanceTest` (3 tests), `ReconstructionTest` (1 test), `MotionTest` (2 tests), `MfsrTest` (2 tests), `SimdTest` (1 test), and `QualityTest` (2 tests).

### CLI End-to-End Verification (`White-throated Kingfisher Video Animation.mp4`, 720×1280 HEVC + 44.1 kHz AAC)
- `upscale --scale 2 --max-frames 12`: exited `0`; `ffprobe` confirmed `h264 1440×2560` (`12` frames) + preserved `aac 44100 Hz stereo` (`20` frames).
- `upscale --scale 4 --preset quality --max-frames 6`: exited `0`; `ffprobe` confirmed `h264 2880×5120` (`6` frames) + preserved `aac 44100 Hz stereo` (`10` frames).
- `upscale --backend cpu --max-frames 6`: exited `0`.
- `analyze --report analysis.json --max-frames 24`: exited `0`.
- `upscale --backend gpu`: exited `5` (`ERROR [unsupported]`).

### Kernel Micro-Benchmarks (`vu_bench.exe`, 11th Gen Intel i5-1135G7)
- `BM_SepFilter` (1280×720): `5.58 ms` scalar → `1.06 ms` AVX2+FMA (**5.3× speedup**)
- `BM_BtvGradient` (1280×720): `33.0 ms` scalar → `5.14 ms` AVX2+FMA (**6.4× speedup**)
- End-to-end 720×1280 → 1440×2560 `balanced` MFSR throughput improved from `0.35 fps` to `1.30–1.45 fps`.

### Evaluation Dataset Results (`benchmarks/datasets/eval`, separated from `dev`)
- **2× (8 clips: `clean`, `compressed`, `noisy`, `motionblur` across synthetic & real sources)**:
  - `nearest`: `28.837 dB` / `0.8145` SSIM (`315.37 fps`)
  - `bilinear`: `30.167 dB` / `0.8381` SSIM (`322.93 fps`)
  - `bicubic`: `32.131 dB` / `0.8696` SSIM (`301.18 fps`)
  - `lanczos`: `32.169 dB` / `0.8698` SSIM (`225.72 fps`)
  - `edge`: `31.588 dB` / `0.8610` SSIM, lowest temporal consistency error `0.01259` (`141.18 fps`)
  - `ibp`: `31.364 dB` / `0.8595` SSIM (`39.64 fps`)
  - `mfsr`: `31.463 dB` / `0.8606` SSIM (`12.81 fps`; beats `bicubic`/`lanczos` on `motionblur` clips and `real_eval_clean`)
- **4× & 8× Direct vs. Progressive**:
  - `4×`: Direct `ibp` (`26.177 dB` / `0.6660` SSIM) and `mfsr` (`26.173 dB` / `0.6656` SSIM) beat progressive `ibp_prog` (`25.324 dB`) and `mfsr_prog` (`25.353 dB`).
  - `8×`: Direct `mfsr` (`22.953 dB` / `0.4471` SSIM) beat `mfsr_prog` (`22.672 dB` / `0.4213` SSIM) and `bicubic` (`22.856 dB` / `0.4296` SSIM).

---

## 4. Assumptions Made
- Windows x64 MSVC 2022+ environment with pinned third-party libraries fetched into `third_party/`.
- Internal planar 8-bit `yuv420p` pipeline (higher bit-depth / 4:2:2 / 4:4:4 sources converted via `libswscale` on decode).
- Pure spatial baselines keep `auto` denoise/deblock off unless explicitly enabled so baseline measurements stay uncontaminated.
- Auto PSF width adapts to measured frame blur `(0.26 + 0.16 * t) * scale` when `psf_sigma <= 0`.

---

## 5. Incomplete / Experimental Features & Limitations
- **GPU Acceleration**: Not implemented; `--backend gpu` returns exit code `5` (`Unsupported`). CPU/OpenMP/AVX2 paths are fully functional.
- **8× Upscaling & `temporal_refine`**: Experimental; classical signal processing cannot hallucinate missing high-frequency detail destroyed by extreme downsampling or quantization.
- **Chroma Planes**: `U`/`V` channels use fast spatial interpolation (`bicubic` by default) rather than multi-frame IBP.
