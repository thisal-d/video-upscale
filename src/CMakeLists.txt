set(VU_LIB_SOURCES
  core/config.cpp
  core/frame_pool.cpp
  core/log.cpp
  core/system.cpp
  io/decoder.cpp
  io/encoder.cpp
  io/ffmpeg_utils.cpp
  compute/backend.cpp
  compute/kernels.cpp
  analysis/frame_analysis.cpp
  enhance/spatial_enhance.cpp
  motion/optical_flow.cpp
  reconstruct/spatial.cpp
  reconstruct/ibp.cpp
  reconstruct/mfsr.cpp
  quality/metrics.cpp
  quality/benchmark_runner.cpp
  pipeline/processor.cpp
  pipeline/pipeline.cpp
)

if(VU_ENABLE_AVX2)
  list(APPEND VU_LIB_SOURCES compute/kernels_avx2.cpp)
  set_source_files_properties(compute/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "${VU_AVX2_FLAGS}")
endif()

add_library(videoupscale_lib STATIC ${VU_LIB_SOURCES})
if(VU_ENABLE_AVX2)
  target_compile_definitions(videoupscale_lib PRIVATE VU_HAVE_AVX2_KERNELS=1)
endif()
target_include_directories(videoupscale_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(videoupscale_lib
  PUBLIC
    ${FFMPEG_TARGETS}
    ${OpenCV_LIBS}
    vu_json
    vu_compile_options
)

add_executable(videoupscale cli/main.cpp)
target_link_libraries(videoupscale PRIVATE videoupscale_lib vu_compile_options)
vu_copy_runtime_dlls(videoupscale)
