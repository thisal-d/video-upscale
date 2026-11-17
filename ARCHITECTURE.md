# VideoUpscale Architecture

VideoUpscale is a CPU-first, classical computer-vision video super-resolution and enhancement engine built in C++20.

## Subsystem Layout (`src/`)

- **`core/`**:
  - `error.hpp`: Structured `ErrorCode` (`InvalidArgument`, `InputError`, `OutputError`, `Unsupported`, `Cancelled`) and `vu::Error`.
  - `frame.hpp`: FFmpeg-free `VideoFrame` (planar 8-bit YUV 4:2:0 `cv::Mat` planes, PTS, duration, scene segment ID) and `VideoInfo`.
  - `frame_pool.hpp/.cpp`: Lock-guarded buffer recycling pool (`FramePool`) with automatic `shared_ptr` return deleter.
  - `bounded_queue.hpp`: Fixed-capacity MPMC `BoundedQueue<T>` providing back-pressure between pipeline stages.
  - `config.hpp/.cpp`: `ProcessingConfig` with preset profiles (`fast`, `balanced`, `quality`), JSON serialization/loading, and strict validation.
  - `system.hpp/.cpp`: Windows working-set memory inspection, CPUID/AVX2 feature probing, and `SIGINT` graceful cancellation.
  - `log.hpp/.cpp`: Thread-safe leveled logger and stderr progress reporting.

- **`io/`**:
  - `decoder.hpp/.cpp`: FFmpeg (`libavformat`/`libavcodec`/`libswscale`) demuxer & decoder converting to planar YUV420p, repairing non-monotonic/missing PTS, detecting VFR and color metadata, and forwarding non-video packets via callback.
  - `encoder.hpp/.cpp`: FFmpeg muxer & encoder (`libx264`, `libx265`, fallback encoders), explicit color tagging when source metadata is unspecified, display rotation preservation, and thread-safe stream-copy of audio/subtitles.

- **`compute/`**:
  - `backend.hpp/.cpp`: Backend selection (`auto`, `cpu`, `openmp`, `gpu`), OpenMP non-nested thread policy (`omp_set_nested(0)`), and OpenCV thread control to prevent oversubscription.
  - `kernels.hpp/.cpp`: Serial and row-parallel float image kernels (separable Gaussian filtering, bilinear sampling/splatting, BTV gradient, u8/float conversion) with runtime dispatch.
  - `kernels_avx2.hpp/.cpp`: Dedicated AVX2+FMA SIMD translation unit compiled with `/arch:AVX2` (`-mavx2 -mfma`), gated by `CPUID` + `XGETBV`.

- **`analysis/`**:
  - `frame_analysis.hpp/.cpp`: Immerkær MAD noise estimation, Crete re-blurring blur estimation, Sobel edge magnitude/density, 8×8 blockiness metric, texture ratio, and histogram+MAD scene-cut detection.

- **`enhance/`**:
  - `spatial_enhance.hpp/.cpp`: Adaptive edge-preserving bilateral denoising, 4-tap 8×8 block-boundary deblocking, adaptive detail sharpening with 3×3 min/max halo suppression.

- **`motion/`**:
  - `optical_flow.hpp/.cpp`: Bidirectional DIS optical flow (`ultrafast`/`fast`/`medium`), forward–backward consistency, occlusion masking, photometric+geometric confidence weighting, and sub-pixel flow warping.

- **`reconstruct/`**:
  - `spatial.hpp/.cpp`: Nearest, Bilinear, Bicubic, Lanczos, structure-tensor Edge-Directed interpolation (`edge`), progressive vs. direct `2x`/`4x`/`8x` scaling.
  - `ibp.hpp/.cpp`: Tiled OpenMP single-frame Huber + BTV Iterative Back-Projection (`ibp`).
  - `mfsr.hpp/.cpp`: 5-frame sliding-window Multi-Frame Super-Resolution (`mfsr`) with scene-cut isolation, `CV_16SC2` sub-pixel alignment, confidence-weighted fusion, multi-frame Huber+BTV IBP, consistency fallback, and optional temporal refinement.

- **`quality/`**:
  - `metrics.hpp/.cpp`: PSNR, Wang et al. SSIM, motion-compensated temporal warped RMSE, and temporal consistency error.
  - `benchmark_runner.hpp/.cpp`: Multi-method dataset benchmark suite (`videoupscale benchmark`) supporting `2x`/`4x`/`8x` and direct-vs-progressive comparisons.

- **`pipeline/`**:
  - `processor.hpp/.cpp`: Streaming `FrameProcessor` abstraction (supports temporal lookahead and clean flush).
  - `pipeline.hpp/.cpp`: 3-thread bounded streaming pipeline (decode → process → encode), `FramePool` memory bounding, stage timing, and post-encode validation.

- **`cli/`**:
  - `main.cpp`: CLI entrypoint (`upscale`, `analyze`, `benchmark`, JSON reporting, exit codes).

## Threading & Memory Model

```
[Decode Thread] --(BoundedQueue, cap Q)--> [Process Thread] --(BoundedQueue, cap Q)--> [Encode Thread]
       |                                                                                       ^
       +-----------------------(non-video packets: audio / subtitles)--------------------------+
```

- **Bounded Memory**: Only `~2*Q + lookahead` input frames and `~Q` output frames live in memory at once; large videos stream without loading into RAM.
- **No Oversubscription**: OpenMP regions are never nested; `VideoDecoder` and `VideoEncoder` cap internal codec threads.
