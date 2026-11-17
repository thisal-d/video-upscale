# Building VideoUpscale on Windows

## Toolchain & Pinned Dependencies

- **Compiler**: Visual Studio 2022 Community (MSVC 19.44 / toolset 14.44, C++20)
- **Build System**: CMake (>= 3.24) + Ninja (both bundled with VS 2022)
- **FFmpeg**: `n8.1` (`BtbN` win64 GPL shared build, `avcodec` 62 / `avformat` 62 / `avutil` 60 / `swscale` 9 / `swresample` 6, includes `libx264` / `libx265`)
- **OpenCV**: `4.14.0` (official Windows MSVC x64 `opencv_world4140`)
- **JSON**: `nlohmann/json` `v3.12.0`
- **Tests**: GoogleTest `v1.18.0`
- **Micro-benchmarks**: Google Benchmark `v1.10.0`

## 1. Fetch Dependencies

Run from PowerShell (requires internet access):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_deps.ps1
```

This populates `third_party/` with FFmpeg, OpenCV, GoogleTest, Google Benchmark, and `nlohmann/json.hpp`.

## 2. Configure & Build (One-Command, x64 Release)

From PowerShell (auto-detects Visual Studio via `vswhere.exe`, initializes `vcvars64.bat`, configures Ninja, builds, runs CTest, and builds the portable ZIP):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Clean -Test -Package
```

Or manually from an **x64 Native Tools Command Prompt for VS 2022**:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## 3. Run Automated Tests

```bat
ctest --test-dir build --output-on-failure
```

## 4. Packaging (Portable ZIP)

```bat
cpack --config build/CPackConfig.cmake -B build/packages
```
