add_library(vu_compile_options INTERFACE)

if(MSVC)
  target_compile_options(vu_compile_options INTERFACE
    /W4 /permissive- /utf-8 /Zc:__cplusplus /EHsc /MP
    /wd4251  # dll-interface warnings from third-party headers
    $<$<CONFIG:Release,RelWithDebInfo>:/O2 /Oi /Ot /GL- /fp:fast>)
  target_compile_definitions(vu_compile_options INTERFACE
    NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(vu_compile_options INTERFACE -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release,RelWithDebInfo>:-O3>)
endif()

if(VU_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(vu_compile_options INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(vu_compile_options INTERFACE VU_HAVE_OPENMP=1)
endif()

# Flags for translation units that contain AVX2 kernels (dispatched at runtime).
if(MSVC)
  set(VU_AVX2_FLAGS /arch:AVX2)
else()
  set(VU_AVX2_FLAGS -mavx2 -mfma)
endif()
