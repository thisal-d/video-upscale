add_executable(vu_bench
  bench_kernels.cpp
)
target_link_libraries(vu_bench PRIVATE videoupscale_lib benchmark::benchmark_main vu_compile_options)
vu_copy_runtime_dlls(vu_bench)
