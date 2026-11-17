# Locates the pinned third-party dependencies fetched by scripts/fetch_deps.ps1.
# Every dependency can be overridden with standard CMake variables
# (FFMPEG_ROOT, OpenCV_DIR, VU_GTEST_SOURCE_DIR, VU_BENCHMARK_SOURCE_DIR, VU_JSON_INCLUDE_DIR).

# ---------------- FFmpeg (libavformat/libavcodec/libavutil/libswscale) ----------------
set(FFMPEG_ROOT "${VU_THIRD_PARTY_DIR}/ffmpeg" CACHE PATH "FFmpeg dev root (include/, lib/, bin/)")
set(_ff_libs avformat avcodec avutil swscale swresample)
set(FFMPEG_TARGETS "")
foreach(_l IN LISTS _ff_libs)
  find_library(FFMPEG_${_l}_LIB NAMES ${_l} lib${_l} PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
  if(NOT FFMPEG_${_l}_LIB)
    message(FATAL_ERROR "FFmpeg library '${_l}' not found under ${FFMPEG_ROOT}/lib. Run scripts/fetch_deps.ps1 or set FFMPEG_ROOT.")
  endif()
  add_library(FFmpeg::${_l} UNKNOWN IMPORTED)
  set_target_properties(FFmpeg::${_l} PROPERTIES
    IMPORTED_LOCATION "${FFMPEG_${_l}_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_ROOT}/include")
  list(APPEND FFMPEG_TARGETS FFmpeg::${_l})
endforeach()
if(NOT EXISTS "${FFMPEG_ROOT}/include/libavformat/avformat.h")
  message(FATAL_ERROR "FFmpeg headers not found under ${FFMPEG_ROOT}/include")
endif()
file(GLOB VU_FFMPEG_DLLS "${FFMPEG_ROOT}/bin/*.dll")
set(VU_FFMPEG_EXE "${FFMPEG_ROOT}/bin/ffmpeg.exe")
set(VU_FFPROBE_EXE "${FFMPEG_ROOT}/bin/ffprobe.exe")

# ---------------- OpenCV ----------------
if(NOT OpenCV_DIR AND EXISTS "${VU_THIRD_PARTY_DIR}/opencv/build")
  set(OpenCV_DIR "${VU_THIRD_PARTY_DIR}/opencv/build")
endif()
set(OpenCV_STATIC OFF)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc video photo)
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")
set(VU_OPENCV_DLLS "")
if(WIN32)
  # opencv_world DLL from the official Windows distribution.
  file(GLOB VU_OPENCV_DLLS "${OpenCV_DIR}/x64/vc16/bin/opencv_world*.dll")
  list(FILTER VU_OPENCV_DLLS EXCLUDE REGEX "d\\.dll$")
  file(GLOB VU_OPENCV_DLLS_DEBUG "${OpenCV_DIR}/x64/vc16/bin/opencv_world*d.dll")
endif()

# ---------------- OpenMP ----------------
if(VU_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ---------------- nlohmann/json (header only) ----------------
set(VU_JSON_INCLUDE_DIR "${VU_THIRD_PARTY_DIR}/nlohmann" CACHE PATH "Directory containing nlohmann/json.hpp")
if(NOT EXISTS "${VU_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
  message(FATAL_ERROR "nlohmann/json.hpp not found in ${VU_JSON_INCLUDE_DIR}/nlohmann")
endif()
add_library(vu_json INTERFACE)
target_include_directories(vu_json SYSTEM INTERFACE "${VU_JSON_INCLUDE_DIR}")

# ---------------- GoogleTest (optional, for tests) ----------------
if(VU_BUILD_TESTS)
  set(VU_GTEST_SOURCE_DIR "${VU_THIRD_PARTY_DIR}/googletest" CACHE PATH "GoogleTest source dir")
  if(NOT EXISTS "${VU_GTEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GoogleTest not found in ${VU_GTEST_SOURCE_DIR}. Run scripts/fetch_deps.ps1.")
  endif()
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  add_subdirectory("${VU_GTEST_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/googletest" EXCLUDE_FROM_ALL)
endif()

# ---------------- Google Benchmark (optional, for bench) ----------------
if(VU_BUILD_BENCHMARKS)
  set(VU_BENCHMARK_SOURCE_DIR "${VU_THIRD_PARTY_DIR}/benchmark" CACHE PATH "Google Benchmark source dir")
  if(NOT EXISTS "${VU_BENCHMARK_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Google Benchmark not found in ${VU_BENCHMARK_SOURCE_DIR}. Run scripts/fetch_deps.ps1.")
  endif()
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  add_subdirectory("${VU_BENCHMARK_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/benchmark" EXCLUDE_FROM_ALL)
endif()

set(VU_RUNTIME_DLLS ${VU_FFMPEG_DLLS} ${VU_OPENCV_DLLS})

# Copies runtime DLLs next to a target's executable after build.
function(vu_copy_runtime_dlls target)
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VU_FFMPEG_DLLS} $<TARGET_FILE_DIR:${target}>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<IF:$<CONFIG:Debug>,${VU_OPENCV_DLLS_DEBUG},${VU_OPENCV_DLLS}>" $<TARGET_FILE_DIR:${target}>
      COMMAND_EXPAND_LISTS)
  endif()
endfunction()
