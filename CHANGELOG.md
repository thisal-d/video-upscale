# Changelog & Development Log

All notable engineering decisions, milestone builds, and verified test results are recorded here.

## [0.1.1] - Release Update
- **Version Bump**: Updated project, CLI, Desktop GUI, installer, and release documentation to `0.1.1` (`VideoUpscale-0.1.1-Setup.exe`, `VideoUpscale-0.1.1-win64.zip`).
- **Tyranox - crazy & 3× Scaling**: Added `tyranox_crazy` (`"Tyranox - crazy"`, with `tyranox` alias) pure `1 → s×s` exact luma & chroma pixel replication alongside `tyranox_edge` (2×/3× diagonal anti-staircasing) and `tyranox_smooth`; added `3×` integer scaling (`3×3 = 9` pixels per original pixel, e.g. 360p → 1080p) with clean `yuv420p` even-dimension validation.

## [0.1.0] - Milestones 1–7

### Milestone 1 & 2: Foundation & Spatial Baselines
- **Build & Dependencies**: CMake 3.24+ / Ninja / MSVC 19.44 C++20; `scripts/fetch_deps.ps1` pinning FFmpeg `n8.1`, OpenCV `4.14.0`, GoogleTest `v1.18.0`, Google Benchmark `v1.10.0`, and nlohmann/json `v3.12.0`.
- **Core & Video I/O**: `FramePool`, `BoundedQueue`, `ProcessingConfig` (JSON/presets), `VideoDecoder` (planar YUV420p, VFR PTS repair/preservation, color metadata), `VideoEncoder` (libx264/fallback, color tagging, rotation matrix, audio/subtitle stream copy), 3-thread streaming pipeline, post-encode validation.
- **Spatial Baselines**: Nearest, Bilinear, Bicubic, Lanczos, and structure-tensor Edge-Directed interpolation (`edge`) for `2x`, `4x`, `8x` (direct & progressive).

### Milestone 3: Analysis, Spatial Enhancement & Single-Frame IBP
- **Analysis (`src/analysis/`)**: Immerkær MAD noise estimation, Crete re-blurring blur score, 8×8 blockiness metric, Sobel edge density, texture ratio, histogram+MAD scene-cut detection.
- **Enhancement (`src/enhance/`)**: Adaptive edge-preserving bilateral denoising, 4-tap 8×8 deblocking loop filter, edge-masked adaptive sharpening with 3×3 min/max halo suppression.
- **IBP (`src/reconstruct/ibp.cpp`)**: Tiled OpenMP Huber + Bilateral Total Variation (BTV) Iterative Back-Projection.

### Milestone 4 & 5: Motion Analysis & 5-Frame MFSR
- **Motion (`src/motion/optical_flow.cpp`)**: Bidirectional DIS optical flow, forward–backward cycle consistency, occlusion masking, photometric+geometric confidence weighting.
- **MFSR (`src/reconstruct/mfsr.cpp`)**: 5-frame sliding window (`I(t-2)..I(t+2)`), scene-cut buffer reset, sub-pixel HR warping (`INTER_CUBIC` + precomputed `CV_16SC2` maps), confidence-weighted fusion, multi-frame Huber+BTV IBP, observation consistency check with single-frame IBP fallback, optional temporal refinement.

### Milestone 6: Performance & SIMD Optimization
- **AVX2 + FMA (`src/compute/kernels_avx2.cpp`)**: Runtime-dispatched (`CPUID` + `XGETBV`) AVX2+FMA kernels for separable Gaussian convolution (`5.3×` speedup on 720p), BTV gradient (`6.4×` speedup on 720p), and u8↔float conversion.
- **Pipeline Speedup**: Precomputed `CV_16SC2` remap tables + OpenMP kernel parallelization improved 720×1280 → 1440×2560 `balanced` MFSR from `0.35 fps` to `~1.30–1.45 fps`.

### Milestone 7: Quality Metrics, Dataset & Benchmark Tuning
- **Quality (`src/quality/`)**: PSNR, SSIM (Wang et al. 2004), motion-compensated temporal warped RMSE, temporal consistency error, and `videoupscale benchmark` runner.
- **Controlled Datasets (`scripts/generate_dataset.py`)**: Strictly separated `dev` and `eval` splits covering `clean`, `compressed`, `noisy`, and `motionblur` degradations across `2x`, `4x`, and `8x`.
- **Empirical Tuning on `dev`**: Restricted `edge` steering gate to strong edges (`[0.035, 0.10]`) and tuned default `psfSigma` from `0.45*scale` to `0.28*scale` (`btvLambda = 0.0010`) to prevent over-sharpening ringing on area-downscaled sources.
