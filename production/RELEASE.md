# VideoUpscale Studio v0.1.1 — Release

## Overview
**VideoUpscale Studio v0.1.1** is the Windows x64 release of the CPU-first classical computer-vision video super-resolution, enhancement, and watermark-removal suite.

---

## Included Production Artifacts
- **`VideoUpscale-0.1.1-Setup.exe`** — Standalone Windows Inno Setup installer (installs **VideoUpscale Studio** GUI, `videoupscale.exe` CLI, FFmpeg `n8.1`, OpenCV `4.14.0` runtime DLLs, and preset configs; no Python installation required).
- **`VideoUpscale-0.1.1-win64.zip`** — Portable CLI ZIP archive (`videoupscale.exe`, runtime DLLs, presets, and documentation).

---

## Key Features in v0.1.1
- **Desktop GUI (`VideoUpscaleUI.exe`)**:
  - Multi-video input queue + **Inspect Selected** video quality analyzer (`analyze`)
  - Toggleable built-in **Dola AI** and **Gemini Sparkle** watermark removers (`inpaint` / `delogo`, optional custom ROI)
  - Multi-select **Scales** (`2×`, `3×`, `4×`, `8×`) and **Algorithms** (`mfsr`, `ibp`, `edge`, `tyranox_crazy` ("Tyranox - crazy"), `tyranox_edge`, `tyranox_smooth`, `lanczos`, `bicubic`, `bilinear`, `nearest`) for batch side-by-side comparison outputs
  - Presets (`quality`, `balanced`, `fast`), adaptive denoising/deblocking/sharpening with halo suppression, CRF/codec (`libx264`/`libx265`), audio preservation, 24-frame preview mode, live progress bars, and cancellation
- **Classical CV Engine (`videoupscale.exe`)**:
  - 5-frame Multi-Frame Super-Resolution (`mfsr`) with DIS optical flow, forward–backward occlusion gating, confidence-weighted fusion, Huber+BTV Iterative Back-Projection, and scene-cut buffer resets
  - Single-frame tiled OpenMP `ibp` and structure-tensor `edge` interpolation
  - Runtime-dispatched **AVX2+FMA** kernels (`5.3×` separable filter, `6.4×` BTV gradient speedups on 720p)
  - Full AAC/audio stream copying & VFR timestamp preservation
- **Verification**:
  - **100% automated CTest unit & integration test suite passing** (including exact `2×` `1→4` and `3×` `1→9` luma & chroma pixel replication tests)
  - Controlled `dev` vs. `eval` dataset benchmarks verified across `2×`, `4×`, and `8×`

---

## Quick Start After Installing
1. Run **`VideoUpscale-0.1.1-Setup.exe`** and launch **VideoUpscale Studio** from the Start Menu or Desktop shortcut.
2. Add your input video(s), enable/disable watermark removers if needed, select your scale(s) and algorithm(s), and click **Start Processing**.

---

## System Requirements & Known Limitations
- **OS**: Windows 10 / 11 (x64)
- **CPU**: x86-64 (AVX2+FMA supported & auto-detected; falls back cleanly to scalar/OpenMP)
- **Limitations**: CPU-only (`--backend gpu` is unsupported); internal processing uses 8-bit `yuv420p`; `8×` upscaling and `temporal_refine` are experimental.
