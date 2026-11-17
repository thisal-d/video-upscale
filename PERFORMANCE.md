# Performance Engineering & Measurements

All measurements below were collected on **11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40 GHz (4 cores / 8 logical threads, AVX2+FMA)**, Release build (`MSVC 19.44 /O2`, OpenMP 2.0, OpenCV 4.14.0, FFmpeg 8.1).

## 1. Kernel Micro-Benchmarks (`vu_bench.exe`)

| Benchmark | Resolution | Time (ms) | Throughput | Notes |
|---|---|---|---|---|
| `BM_SpatialUpscale/Bicubic_2x` | 640×360 → 1280×720 | 0.312 ms | 3,373 frames/s | OpenCV SIMD |
| `BM_SpatialUpscale/Lanczos_2x` | 640×360 → 1280×720 | 1.04 ms | 964 frames/s | OpenCV SIMD |
| `BM_SpatialUpscale/EdgeAware_2x` | 640×360 → 1280×720 | 34.9 ms | 28.8 frames/s | OpenMP row-parallel |
| `BM_IbpTiled/IBP_2x_640x360` | 640×360 → 1280×720 | 82.5 ms | 12.8 frames/s | OpenMP tiled (192×192) + AVX2 |
| `BM_SepFilter/Scalar_720p` | 1280×720 | 5.58 ms | 177.2 frames/s | Single-thread scalar |
| `BM_SepFilter/AVX2_720p` | 1280×720 | **1.06 ms** | **1,007 frames/s** | **5.3× speedup** (AVX2+FMA) |
| `BM_BtvGradient/Scalar_720p` | 1280×720 | 33.0 ms | 30.3 frames/s | Single-thread scalar |
| `BM_BtvGradient/AVX2_720p` | 1280×720 | **5.14 ms** | **192.0 frames/s** | **6.4× speedup** (AVX2+FMA) |

## 2. Architecture Optimizations

- **Bounded Streaming Memory**: 3-stage threaded pipeline with `BoundedQueue` (`queue_depth=4`) and `FramePool` recycling. On 720×1280 → 1440×2560 `mfsr`, `FramePool` allocates only **9 frames total** across the entire run.
- **Zero Nested Parallelism**: `omp_set_nested(0)` enforced; tiled `ibp` runs strictly serial kernels per OpenMP tile thread, while `mfsr` uses OpenMP row-parallel `sepFilterParallel` / `addBtvGradientParallel` and precomputed `CV_16SC2` fixed-point `remap` tables.
- **End-to-End Pipeline Speedup (720×1280 → 1440×2560 `balanced` MFSR)**: Improved from `0.35 fps` (initial unoptimized implementation) to `1.30–1.45 fps` (~4× faster) via precomputed `CV_16SC2` maps, OpenMP kernel parallelization, and AVX2+FMA vectorization.

## 3. End-to-End Evaluation Throughput & Memory (`benchmarks/datasets/eval`, 384×384 output)

| Scale | Method | Mean FPS | Peak Working Set (MB) |
|---|---|---|---|
| 2× | `bicubic` | 301.18 | 82.1 |
| 2× | `lanczos` | 225.72 | 82.1 |
| 2× | `edge` | 141.18 | 82.1 |
| 2× | `ibp` | 39.64 | 82.1 |
| 2× | `mfsr` | 12.81 | 82.1 |
| 4× | `ibp` (direct) | 49.86 | 70.5 |
| 4× | `mfsr` (direct) | 17.91 | 71.5 |
| 8× | `mfsr` (direct) | 15.16 | 68.3 |
