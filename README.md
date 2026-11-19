<div align="center">
  <img src="assets/logo.png" alt="VideoUpscale Studio Logo" width="140" />
  <h1>VideoUpscale Studio</h1>
  <p><strong>CPU-first classical computer-vision video super-resolution, enhancement &amp; watermark removal suite</strong></p>
  <p>Built in C++20 with FFmpeg, OpenCV, OpenMP &amp; AVX2 — zero neural networks, full audio &amp; VFR timestamp preservation.</p>
</div>

---

## Features

- **Scaling Factors**: `2x`, `3x`, `4x`, `8x` (direct reconstruction and progressive `2x` stage modes for `4x`/`8x`)
- **Reconstruction & Spatial Baselines**:
  - `nearest`, `bilinear`, `bicubic`, `lanczos`
  - `edge`: Structure-tensor anisotropic edge-directed interpolation
  - `ibp`: Tiled OpenMP single-frame Huber + Bilateral Total Variation (BTV) Iterative Back-Projection
  - `mfsr`: 5-frame sliding-window (`I(t-2)..I(t+2)`) DIS optical flow, forward–backward occlusion gating, confidence-weighted temporal fusion, multi-frame Huber+BTV IBP, consistency checks, and scene-cut buffer resets
- **Pre/Post Enhancement & Analysis**:
  - `analyze`: Immerkær MAD noise estimation, Crete re-blurring blur score, 8×8 blockiness, Sobel edge density, texture ratio, scene-cut detection, and quality advisories
  - `enhance`: Adaptive bilateral denoising (`off`/`auto`/`on`), 8×8 deblocking loop filter (`off`/`auto`/`on`), adaptive detail sharpening with 3×3 min/max halo suppression
- **Video I/O & Streaming Pipeline**:
  - FFmpeg (`libavformat`/`libavcodec`/`libswscale`) streaming demuxer/decoder and muxer/encoder (`libx264`, `libx265`, fallback codecs)
  - 3-thread bounded pipeline (`decode → process → encode`) with `FramePool` buffer recycling
  - Audio & subtitle stream copying, VFR PTS preservation, color-space tagging, display rotation matrix copy, graceful `SIGINT` cancellation, and post-encode validation
- **Compute Backends**: `auto`, `cpu` (single-threaded), `openmp` (non-nested multi-core), plus runtime-dispatched **AVX2+FMA** kernels (`gpu` is explicitly rejected as unimplemented)

## CLI Usage

```powershell
# Upscale 2x using 5-frame MFSR (default)
videoupscale upscale input.mp4 --output output_2x.mp4 --scale 2

# Upscale 4x with quality preset
videoupscale upscale input.mp4 --output output_4x.mp4 --scale 4 --preset quality

# Upscale with explicit CPU backend or config file
videoupscale upscale input.mp4 --output output.mp4 --backend cpu
videoupscale upscale input.mp4 --output output.mp4 --config configs/balanced.json

# Analyze video degradations and scene cuts
videoupscale analyze input.mp4 --report analysis.json

# Benchmark all methods on a controlled dataset
videoupscale benchmark --dataset benchmarks/datasets/eval --scale 2 --output benchmarks/results/eval_2x.json
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Generic runtime error |
| `2` | Invalid CLI arguments or configuration |
| `3` | Input open / demux / decode error |
| `4` | Output open / encode / mux / validation error |
| `5` | Unsupported feature or backend (`--backend gpu`) |
| `130` | Cancelled by `SIGINT` (output finalized cleanly) |

## Known Limitations

- **CPU-only**: No GPU (CUDA/OpenCL) backend is implemented (`--backend gpu` returns exit code `5`).
- **8-bit 4:2:0 Internal Representation**: 10-bit+ and non-4:2:0 sources are converted to `yuv420p` on decode.
- **8× Upscaling**: Experimental; classical signal processing cannot invent missing high-frequency textures destroyed by downsampling or heavy quantization.
- **Chroma Upscaling**: `U`/`V` planes use fast spatial interpolation (`bicubic` by default) rather than multi-frame IBP.

## Build & Dependencies (Windows)

- **Dependencies**: Visual Studio 2022+ (MSVC C++20), CMake ≥ 3.24, Ninja, FFmpeg `n8.1`, OpenCV `4.14.0`, OpenMP, `nlohmann/json` `v3.12.0`, GoogleTest `v1.18.0`, Google Benchmark `v1.10.0`.
- Fetch pinned third-party dependencies and run a clean build + test + package in two commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_deps.ps1
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Clean -Test -Package
```

## Desktop UI & Windows Installer

Launch the GUI directly in development:

```powershell
python ui\videoupscale_gui.py
```

Or build the standalone **VideoUpscale Studio** GUI + CLI + Inno Setup installer (`build/installer/VideoUpscale-0.1.1-Setup.exe`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

The GUI supports:
- Multi-video input queue + video quality inspection (`analyze`)
- Optional **Dola AI** & **Gemini Sparkle** watermark removal pre-passes (`inpaint` / `delogo`, custom ROI, enable/disable)
- Multi-select **Output Scales** (`2×`, `3×`, `4×`, `8×`) and **Algorithms** (`mfsr`, `ibp`, `edge`, `tyranox_crazy` ("Tyranox - crazy"), `tyranox_edge`, `tyranox_smooth`, `lanczos`, `bicubic`, `bilinear`, `nearest`) to generate multiple comparison outputs in one click
- Presets (`quality`, `balanced`, `fast`), denoise/deblock/sharpen/halo suppression, CRF/codec/backend settings, 24-frame preview mode, live progress bars, and cancellation

## Documentation

- [BUILDING.md](BUILDING.md) — Windows build & dependency instructions
- [ARCHITECTURE.md](ARCHITECTURE.md) — Subsystem layout, threading & memory model
- [ALGORITHMS.md](ALGORITHMS.md) — Mathematical formulations, parameters & assumptions
- [TESTING.md](TESTING.md) — Automated tests, datasets & evaluation benchmark results
- [PERFORMANCE.md](PERFORMANCE.md) — Micro-benchmarks, SIMD speedups & pipeline FPS
- [ROADMAP.md](ROADMAP.md) & [CHANGELOG.md](CHANGELOG.md) — Milestone status & development log
