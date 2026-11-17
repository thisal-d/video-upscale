# Testing & Validation Guide

## 1. Automated Unit & Integration Tests (GoogleTest)

Run the full C++ test suite (`21` tests across `10` suites):

```powershell
.\build\tests\vu_tests.exe
```

### Coverage Summary
- **Core (`CoreTest`)**: `FramePool` buffer reuse, `BoundedQueue` ordering & close semantics, `ProcessingConfig` validation & presets, `ErrorCode::Unsupported`/`InputError` handling.
- **Spatial Baselines (`SpatialTest`)**: Nearest, Bilinear, Bicubic, Lanczos, Edge-Directed at `2x`, `4x`, `8x` (direct & progressive).
- **Video I/O & Audio/Timestamps (`IOTest`)**: Synthetic VFR MKV PTS round-trip preservation; real HEVC + AAC MP4 (`White-throated Kingfisher Video Animation.mp4`) and H.264 + AAC MP4 (`bcb0d8f0...mp4`) end-to-end upscaling (`bicubic` & `mfsr` + `cpu` backend) with AAC stream copy and post-encode validation.
- **Analysis (`AnalysisTest`)**: Immerkær MAD noise sigma accuracy, Crete re-blur monotonicity, 8×8 blockiness detection, histogram+MAD scene-cut detection.
- **Enhancement (`EnhanceTest`)**: Adaptive denoising PSNR improvement, deblocking blockiness reduction & PSNR gain, halo-suppressed sharpening overshoot clamping.
- **Reconstruction (`ReconstructionTest`)**: Huber+BTV IBP convergence & tiled-vs-untiled parity.
- **Motion & MFSR (`MotionTest`, `MfsrTest`)**: Sub-pixel DIS alignment, occlusion detection, 5-frame sub-pixel MFSR beating Bicubic & single-frame IBP, scene-cut temporal buffer reset.
- **SIMD (`SimdTest`)**: AVX2+FMA kernel equivalence against scalar implementations (`< 1e-5` max difference).
- **Quality (`QualityTest`)**: PSNR, SSIM, temporal consistency, and `runBenchmarkSuite` end-to-end execution.

## 2. Controlled Benchmark Dataset Generation

Development (`benchmarks/datasets/dev`) and final evaluation (`benchmarks/datasets/eval`) datasets are strictly separated (distinct synthetic scenes and distinct source video clips):

```powershell
python scripts/generate_dataset.py --frames 20
```

Degradations per reference: `clean`, `compressed` (H.264 CRF 29), `noisy` (additive temporal noise), and `motionblur`, at `2x`, `4x`, and `8x`.

## 3. Running Dataset Benchmarks

```powershell
.\build\src\videoupscale.exe benchmark --dataset benchmarks/datasets/eval --scale 2 --output benchmarks/results/eval_2x.json
.\build\src\videoupscale.exe benchmark --dataset benchmarks/datasets/eval/scale4 --scale 4 --compare-progressive --output benchmarks/results/eval_4x.json
.\build\src\videoupscale.exe benchmark --dataset benchmarks/datasets/eval/scale8 --scale 8 --compare-progressive --output benchmarks/results/eval_8x.json
```

## 4. Measured Evaluation Results (`benchmarks/datasets/eval`)

### 2× Upscaling (8 degraded clips: `clean`, `compressed`, `noisy`, `motionblur` across synthetic & real sources)

| Method | Mean PSNR (dB) | Mean SSIM | Temporal Warp RMSE | Temporal Consistency Error | Mean FPS | Peak RAM (MB) |
|---|---|---|---|---|---|---|
| `nearest` | 28.837 | 0.8145 | 0.01820 | 0.01581 | 315.37 | 82.1 |
| `bilinear` | 30.167 | 0.8381 | 0.00813 | 0.01336 | 322.93 | 82.1 |
| `bicubic` | 32.131 | 0.8696 | 0.01022 | 0.01362 | 301.18 | 82.1 |
| `lanczos` | 32.169 | 0.8698 | 0.01045 | 0.01388 | 225.72 | 82.1 |
| `edge` | 31.588 | 0.8610 | 0.00872 | **0.01259** | 141.18 | 82.1 |
| `ibp` | 31.364 | 0.8595 | 0.01102 | 0.01469 | 39.64 | 82.1 |
| `mfsr` | 31.463 | 0.8606 | 0.01028 | 0.01433 | 12.81 | 82.1 |

**Key Observations**:
- **By Degradation Type** (from `eval_2x.json`): On `motionblur` clips, `mfsr` and `ibp` outperform `bicubic` and `lanczos` on both PSNR and SSIM (`28.78 dB` vs `28.60 dB` synthetic; `25.45 dB` vs `25.25 dB` real). On `real_eval_clean`, `mfsr` (`31.19 dB`) outperforms `bicubic` (`31.09 dB`). On pure area-downscaled `synth_eval_clean` and `noisy` clips, `lanczos`/`bicubic` achieve higher mean PSNR while `edge` delivers the lowest temporal consistency error (`0.01259`).
- On controlled sub-pixel multi-frame sequences (`MfsrTest` & `QualityTest`), 5-frame `mfsr` achieves `45.95 dB` vs `45.08 dB` (`bicubic`) and outperforms single-frame `ibp`.

### 4× Direct vs. Progressive Upscaling (`benchmarks/datasets/eval/scale4`, 2 clips)

| Method | Mode | Mean PSNR (dB) | Mean SSIM | Temporal Warp RMSE | Mean FPS | Peak RAM (MB) |
|---|---|---|---|---|---|---|
| `bicubic` | Direct | 27.241 | 0.7300 | 0.00614 | 330.95 | 59.8 |
| `edge` | Direct | 25.496 | 0.6264 | 0.00466 | 224.05 | 68.3 |
| `edge_prog` | Progressive (2×→2×) | 25.490 | 0.6263 | 0.00464 | 118.28 | 68.3 |
| `ibp` | Direct | 26.177 | 0.6660 | 0.00826 | 49.86 | 70.5 |
| `ibp_prog` | Progressive (2×→2×) | 25.324 | 0.6014 | 0.01066 | 35.32 | 70.5 |
| `mfsr` | Direct | 26.173 | 0.6656 | 0.00769 | 17.91 | 71.5 |
| `mfsr_prog` | Progressive (2×→2×) | 25.353 | 0.6022 | 0.01017 | 19.95 | 71.8 |

### 8× Direct vs. Progressive Upscaling (Experimental, `benchmarks/datasets/eval/scale8`, 2 clips)

| Method | Mode | Mean PSNR (dB) | Mean SSIM | Temporal Warp RMSE | Mean FPS | Peak RAM (MB) |
|---|---|---|---|---|---|---|
| `bicubic` | Direct | 22.856 | 0.4296 | 0.00623 | 349.10 | 55.9 |
| `ibp` | Direct | 22.844 | 0.4425 | 0.01247 | 47.90 | 68.2 |
| `ibp_prog` | Progressive (2×→2×→2×) | 22.540 | 0.4173 | 0.01711 | 32.86 | 68.2 |
| `mfsr` | Direct | **22.953** | **0.4471** | 0.01084 | 15.16 | 68.3 |
| `mfsr_prog` | Progressive (2×→2×→2×) | 22.672 | 0.4213 | 0.01590 | 24.37 | 68.7 |

**Direct vs. Progressive Conclusion**: Across both `4x` and `8x` evaluations, direct reconstruction (`ibp`, `mfsr`) consistently outperformed progressive `2x` chaining in PSNR (`+0.82 dB` at `4x`, `+0.28 dB` at `8x`), SSIM (`+0.063` at `4x`, `+0.026` at `8x`), and temporal warped RMSE. Progressive chaining compounds intermediate resampling blur and should not be assumed superior.
