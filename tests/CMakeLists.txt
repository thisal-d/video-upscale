include(GoogleTest)

add_executable(vu_tests
  test_core_io.cpp
  test_analysis_enhance_ibp.cpp
  test_motion_mfsr.cpp
  test_quality.cpp
)
target_link_libraries(vu_tests PRIVATE videoupscale_lib GTest::gtest_main vu_compile_options)
target_compile_definitions(vu_tests PRIVATE
  VU_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  VU_FFMPEG_EXE="${VU_FFMPEG_EXE}"
  VU_FFPROBE_EXE="${VU_FFPROBE_EXE}"
)
vu_copy_runtime_dlls(vu_tests)
gtest_discover_tests(vu_tests DISCOVERY_TIMEOUT 30)
