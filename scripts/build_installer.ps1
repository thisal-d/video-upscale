<#
.SYNOPSIS
  Packages VideoUpscale Studio (UI + CLI + FFmpeg + OpenCV + Watermark Removers)
  and builds the Windows Inno Setup installer (.exe).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/build_installer.ps1
#>
param(
  [switch]$RebuildCpp
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root 'build'
$cliExe = Join-Path $buildDir 'src\videoupscale.exe'

if ($RebuildCpp -or -not (Test-Path $cliExe)) {
  Write-Host "[installer] Building C++ engine..."
  & (Join-Path $PSScriptRoot 'build.ps1')
}

# Locate Python + PyInstaller
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py -or $py -like '*WindowsApps*') {
  $py = "$env:LOCALAPPDATA\Python\bin\python.exe"
}
if (-not (Test-Path $py)) { throw "Python not found at $py" }

$pyiWork = Join-Path $buildDir 'pyi_work'
$pyiDist = Join-Path $buildDir 'pyi_dist'
$stageDir = Join-Path $buildDir 'dist_stage'

foreach ($d in @($pyiWork, $pyiDist, $stageDir)) {
  if (Test-Path $d) { Remove-Item -Recurse -Force $d }
}

Write-Host "[installer] Bundling VideoUpscaleUI.exe with PyInstaller..."
& $py -m PyInstaller `
  --noconfirm `
  --onedir `
  --windowed `
  --name VideoUpscaleUI `
  --icon (Join-Path $root 'assets\logo.ico') `
  --add-data "$(Join-Path $root 'assets\logo.ico');." `
  --distpath $pyiDist `
  --workpath $pyiWork `
  --specpath $pyiWork `
  --paths (Join-Path $root 'ui') `
  --hidden-import cv2 `
  --hidden-import numpy `
  --hidden-import imageio_ffmpeg `
  (Join-Path $root 'ui\videoupscale_gui.py')

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

Write-Host "[installer] Staging combined application directory -> $stageDir"
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
Copy-Item -Recurse -Force (Join-Path $pyiDist 'VideoUpscaleUI\*') $stageDir

# Copy C++ CLI + all FFmpeg/OpenCV runtime DLLs + ffmpeg.exe / ffprobe.exe
Copy-Item -Force (Join-Path $buildDir 'src\videoupscale.exe') $stageDir
Copy-Item -Force (Join-Path $buildDir 'src\*.dll') $stageDir
$ffBin = Join-Path $root 'third_party\ffmpeg\bin'
if (Test-Path (Join-Path $ffBin 'ffmpeg.exe')) { Copy-Item -Force (Join-Path $ffBin 'ffmpeg.exe') $stageDir }
if (Test-Path (Join-Path $ffBin 'ffprobe.exe')) { Copy-Item -Force (Join-Path $ffBin 'ffprobe.exe') $stageDir }

# Copy configs and docs (no loose .py files in installed app)
Copy-Item -Recurse -Force (Join-Path $root 'configs') (Join-Path $stageDir 'configs')
foreach ($doc in @('README.md', 'LICENSE-NOTES.md')) {
  $dp = Join-Path $root $doc
  if (Test-Path $dp) { Copy-Item -Force $dp $stageDir }
}

# Locate Inno Setup 6 ISCC.exe
$isccCandidates = @(
  "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
  'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
  'C:\Program Files\Inno Setup 6\ISCC.exe'
)
$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found." }

Write-Host "[installer] Compiling Inno Setup installer via $iscc..."
& $iscc (Join-Path $root 'installer\VideoUpscale.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }

$setupExe = Join-Path $buildDir 'installer\VideoUpscale-0.1.0-Setup.exe'
if (-not (Test-Path $setupExe)) { throw "Expected installer not found at $setupExe" }

$prodDir = Join-Path $root 'production'
New-Item -ItemType Directory -Force -Path $prodDir | Out-Null
Copy-Item -Force $setupExe (Join-Path $prodDir 'VideoUpscale-0.1.0-Setup.exe')
$zipPkg = Join-Path $buildDir 'packages\VideoUpscale-0.1.0-win64.zip'
if (Test-Path $zipPkg) { Copy-Item -Force $zipPkg (Join-Path $prodDir 'VideoUpscale-0.1.0-win64.zip') }

$sizeMb = [math]::Round((Get-Item $setupExe).Length / 1MB, 1)
Write-Host "[installer] DONE -> $setupExe ($sizeMb MB) & copied to $prodDir"
