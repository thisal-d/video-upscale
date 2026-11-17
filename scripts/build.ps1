<#
.SYNOPSIS
  One-command configure, build, and test runner for VideoUpscale on Windows (MSVC + Ninja).

.DESCRIPTION
  Locates Visual Studio 2022+ via vswhere.exe, initializes vcvars64.bat, configures CMake
  with Ninja in Release mode, builds all targets, and optionally runs CTest.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Clean -Test
#>
param(
  [switch]$Clean,
  [switch]$Test,
  [switch]$Package,
  [string]$BuildDir = "build"
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$bdir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "Visual Studio with C++ x64 build tools not found" }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

if ($Clean -and (Test-Path $bdir)) {
  Write-Host "[build] cleaning $bdir"
  Remove-Item -Recurse -Force $bdir
}

$cmd = "call `"$vcvars`" && cmake -S `"$root`" -B `"$bdir`" -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build `"$bdir`" --parallel"
if ($Test) {
  $cmd += " && ctest --test-dir `"$bdir`" --output-on-failure"
}
if ($Package) {
  $cmd += " && cpack --config `"$bdir\CPackConfig.cmake`" -B `"$bdir\packages`""
}

Write-Host "[build] running via $vsPath"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }
