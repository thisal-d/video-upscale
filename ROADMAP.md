# VideoUpscale Implementation Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M1: Foundation** | CMake + MSVC C++20 + FFmpeg 8.1 + OpenCV 4.14 + CLI + streaming pipeline + audio/PTS preservation + GoogleTest | Complete & Verified |
| **M2: Baseline Upscaling** | Nearest, Bilinear, Bicubic, Lanczos, Edge-Directed (`2x`, `4x`, `8x`, direct & progressive) | Complete & Verified |
| **M3: Spatial Enhancement & IBP** | Noise/blur/blockiness/texture analysis, adaptive bilateral denoising, 8×8 deblocking, halo-suppressed sharpening, tiled Huber+BTV IBP | Complete & Verified |
| **M4: Motion Analysis** | DIS optical flow, forward-backward consistency, occlusion masks, photometric & geometric motion confidence, histogram+MAD scene-cut detection | Complete & Verified |
| **M5: Temporal Reconstruction** | 5-frame window MFSR (`I(t-2)..I(t+2)`), confidence-weighted fusion, multi-frame Huber+BTV IBP, observation consistency check, optional temporal refinement | Complete & Verified |
| **M6: Optimization** | OpenMP tile/row parallelism, runtime-dispatched AVX2+FMA SIMD kernels, `CV_16SC2` precomputed maps, Google Benchmark | Complete & Verified |
| **M7: Validation & Dataset** | Strictly separated `dev` & `eval` degraded datasets (`benchmarks/datasets`), `videoupscale benchmark` CLI, PSNR/SSIM/temporal consistency across `2x`, `4x`, `8x` | Complete & Verified |
| **M8: Release Preparation** | Full documentation (`README`, `ARCHITECTURE`, `ALGORITHMS`, `BUILDING`, `TESTING`, `PERFORMANCE`, `ROADMAP`, `CHANGELOG`), CPack Windows portable ZIP | Complete & Verified |
