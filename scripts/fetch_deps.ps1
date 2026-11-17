<#
.SYNOPSIS
  Downloads and extracts pinned third-party dependencies for VideoUpscale into ./third_party.

.DESCRIPTION
  Pinned versions (see BUILDING.md):
    FFmpeg      n8.1  (BtbN win64 gpl shared build: headers + import libs + DLLs, includes libx264)
    OpenCV      4.14.0 (official Windows release, MSVC vc16 x64 opencv_world)
    GoogleTest  v1.18.0 (source, built by CMake)
    Benchmark   v1.10.0 (source, built by CMake)

  Use -DownloadOnly to only fetch archives, -ExtractOnly to only extract already-downloaded archives.
#>
param(
  [switch]$DownloadOnly,
  [switch]$ExtractOnly
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$root = Split-Path -Parent $PSScriptRoot
$tp = Join-Path $root 'third_party'
$dl = Join-Path $tp '_downloads'
New-Item -ItemType Directory -Force -Path $dl | Out-Null

$deps = @(
  @{ Name='ffmpeg';    File='ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip';
     Url='https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip' },
  @{ Name='opencv';    File='opencv-4.14.0-windows.exe';
     Url='https://github.com/opencv/opencv/releases/download/4.14.0/opencv-4.14.0-windows.exe' },
  @{ Name='googletest';File='googletest-1.18.0.zip';
     Url='https://github.com/google/googletest/archive/refs/tags/v1.18.0.zip' },
  @{ Name='benchmark'; File='benchmark-1.10.0.zip';
     Url='https://github.com/google/benchmark/archive/refs/tags/v1.10.0.zip' },
  @{ Name='json';      File='json-3.12.0.hpp';
     Url='https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp' }
)

if (-not $ExtractOnly) {
  foreach ($d in $deps) {
    $out = Join-Path $dl $d.File
    if (Test-Path $out) { Write-Host "[fetch] cached $($d.File)"; continue }
    Write-Host "[fetch] downloading $($d.Url)"
    Invoke-WebRequest -Uri $d.Url -OutFile "$out.part"
    Move-Item "$out.part" $out
  }
}
if ($DownloadOnly) { return }

function Expand-Zip-Flatten([string]$zip, [string]$dest) {
  if (Test-Path $dest) { Write-Host "[extract] exists $dest"; return }
  $tmp = "$dest.tmp"
  if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
  Expand-Archive -Path $zip -DestinationPath $tmp
  $inner = Get-ChildItem $tmp | Select-Object -First 1
  Move-Item $inner.FullName $dest
  Remove-Item -Recurse -Force $tmp
}

Expand-Zip-Flatten (Join-Path $dl 'ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip') (Join-Path $tp 'ffmpeg')
Expand-Zip-Flatten (Join-Path $dl 'googletest-1.18.0.zip') (Join-Path $tp 'googletest')
Expand-Zip-Flatten (Join-Path $dl 'benchmark-1.10.0.zip') (Join-Path $tp 'benchmark')

$jsonDest = Join-Path $tp 'nlohmann\nlohmann'
New-Item -ItemType Directory -Force -Path $jsonDest | Out-Null
Copy-Item (Join-Path $dl 'json-3.12.0.hpp') (Join-Path $jsonDest 'json.hpp') -Force

$ocvDest = Join-Path $tp 'opencv'
if (-not (Test-Path $ocvDest)) {
  # The OpenCV Windows release is a 7-Zip self-extracting archive containing an 'opencv' folder.
  Write-Host "[extract] opencv (7z SFX)"
  $p = Start-Process -FilePath (Join-Path $dl 'opencv-4.14.0-windows.exe') -ArgumentList "-o`"$tp`"", '-y' -Wait -PassThru -NoNewWindow
  if ($p.ExitCode -ne 0) { throw "OpenCV extraction failed ($($p.ExitCode))" }
}
Write-Host "[deps] done -> $tp"
