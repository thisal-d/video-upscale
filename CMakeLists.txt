cmake_minimum_required(VERSION 3.24)
project(VideoUpscale VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VU_BUILD_TESTS      "Build GoogleTest unit/integration tests" ON)
option(VU_BUILD_BENCHMARKS "Build Google Benchmark micro-benchmarks" ON)
option(VU_ENABLE_OPENMP    "Enable OpenMP parallelization" ON)
option(VU_ENABLE_AVX2      "Compile SIMD kernels with AVX2 (runtime-dispatched)" ON)

set(VU_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH "Directory with fetched dependencies")

include(cmake/Dependencies.cmake)
include(cmake/CompilerOptions.cmake)

add_subdirectory(src)

if(VU_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VU_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

# ---- Install / packaging ---------------------------------------------------
include(GNUInstallDirs)
install(TARGETS videoupscale RUNTIME DESTINATION bin)
install(FILES ${VU_RUNTIME_DLLS} DESTINATION bin)
install(DIRECTORY configs/ DESTINATION share/videoupscale/configs)
install(FILES README.md LICENSE-NOTES.md DESTINATION share/doc/videoupscale OPTIONAL)
install(DIRECTORY docs/ DESTINATION share/doc/videoupscale/docs OPTIONAL)

set(CPACK_PACKAGE_NAME "VideoUpscale")
set(CPACK_PACKAGE_VENDOR "VideoUpscale")
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_FILE_NAME "VideoUpscale-${PROJECT_VERSION}-win64")
include(CPack)
