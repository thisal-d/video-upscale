#!/usr/bin/env python3
"""
Generate controlled development ('dev') and evaluation ('eval') benchmark datasets
for VideoUpscale under ./benchmarks/datasets/.

Separates dev and eval datasets completely:
  - dev:  procedural sub-pixel pan chirp + White-throated Kingfisher crop
  - eval: procedural multi-texture motion scene + bcb0d8f0 high-bitrate real clip crop

Each split contains controlled degradations generated via FFmpeg:
  - clean_2x      (Gaussian/area downscale 2x)
  - compressed_2x (2x downscale + H.264 CRF 29)
  - noisy_2x      (2x downscale + additive temporal noise)
  - motionblur_2x (horizontal blur + 2x downscale)
  - clean_4x      (4x downscale for 4x direct vs progressive tests)
  - clean_8x      (8x downscale for 8x direct vs progressive tests)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = ROOT / "third_party" / "ffmpeg" / "bin" / "ffmpeg.exe"


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{p.stderr}")


def write_synthetic_reference(out_mp4: Path, width: int, height: int, nframes: int, variant: str):
    """Write a crisp, high-bitrate yuv420p reference video using rawvideo piped to FFmpeg."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", "24", "-i", "-",
        "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "6", "-preset", "medium", str(out_mp4)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    for t in range(nframes):
        # Sub-pixel pan + object motion (0.75 HR px / frame = 0.375 LR px / frame at 2x)
        sx = 0.75 * t
        sy = 0.45 * t
        px = xx + sx
        py = yy + sy
        if variant == "dev":
            cx, cy = width * 0.5, height * 0.5
            r = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            env = np.exp(-0.00004 * r * r)
            img = 0.42 + 0.14 * np.sin(0.03 * px) * np.cos(0.025 * py)
            img += 0.22 * env * np.sin(0.003 * r * r)
            img += 0.18 * np.tanh(0.8 * ((px - cx) * 0.7 + (py - cy) * 0.5))
        else:
            # Distinct eval scene: rotated grid + multi-scale concentric rings + moving diamond
            cx, cy = width * 0.45, height * 0.55
            r = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            img = 0.38 + 0.15 * np.cos(0.028 * px + 0.018 * py)
            img += 0.20 * np.exp(-0.00003 * r * r) * np.cos(0.0028 * r * r)
            diamond = np.abs(px - (width * 0.35 + 1.2 * t)) + np.abs(py - (height * 0.4 + 0.6 * t))
            img = np.where(diamond < 48.0, 0.85 - 0.006 * diamond, img)
            img += 0.14 * np.tanh(1.1 * (0.6 * (px - cx) - 0.8 * (py - cy)))
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        proc.stdin.write(u8.tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg synthetic write failed: {err}")


def extract_real_reference(src_mp4: Path, out_mp4: Path, start_sec: float, frames: int, crop_expr: str):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run([
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec), "-i", str(src_mp4),
        "-vf", crop_expr, "-frames:v", str(frames), "-an",
        "-c:v", "libx264", "-crf", "8", "-preset", "medium", "-pix_fmt", "yuv420p",
        str(out_mp4)
    ])


def degrade(ref_mp4: Path, out_mp4: Path, scale: int, mode: str):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    w_div = f"iw/{scale}:ih/{scale}"
    if mode == "clean":
        vf = f"scale={w_div}:flags=area"
        crf = "10"
    elif mode == "compressed":
        vf = f"scale={w_div}:flags=area"
        crf = "29"
    elif mode == "noisy":
        vf = f"scale={w_div}:flags=area,noise=alls=9:allf=t"
        crf = "14"
    elif mode == "motionblur":
        vf = f"avgblur=sizeX=3:sizeY=1,scale={w_div}:flags=area"
        crf = "14"
    else:
        raise ValueError(mode)
    run([
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(ref_mp4), "-vf", vf,
        "-c:v", "libx264", "-crf", crf, "-preset", "medium", "-pix_fmt", "yuv420p",
        str(out_mp4)
    ])


def build_split(split_dir: Path, split: str, frames: int):
    ref_dir = split_dir / "reference"
    deg_dir = split_dir / "degraded"
    refs = []

    syn_ref = ref_dir / f"synth_{split}_384x288.mp4"
    write_synthetic_reference(syn_ref, 384, 288, frames, split)
    refs.append(("synth_" + split, syn_ref))

    if split == "dev":
        real_src = ROOT / "example videos" / "White-throated Kingfisher Video Animation.mp4"
        if real_src.exists():
            r = ref_dir / "kingfisher_crop_384x384.mp4"
            extract_real_reference(real_src, r, 1.0, frames, "crop=384:384:168:448")
            refs.append(("kingfisher", r))
    else:
        real_src = ROOT / "example videos" / "bcb0d8f0-c8f9-4f5f-9705-57e887ace9aa.mp4"
        if real_src.exists():
            r = ref_dir / "real_eval_crop_384x384.mp4"
            extract_real_reference(real_src, r, 2.0, frames, "crop=384:384:168:448")
            refs.append(("real_eval", r))

    items_2x, items_4x, items_8x = [], [], []
    for name, ref_path in refs:
        rel_ref = os.path.relpath(ref_path, split_dir).replace("\\", "/")
        for mode in ("clean", "compressed", "noisy", "motionblur"):
            out_2x = deg_dir / f"{name}_2x_{mode}.mp4"
            degrade(ref_path, out_2x, 2, mode)
            items_2x.append({
                "name": f"{name}_{mode}",
                "degradation": mode,
                "reference": rel_ref,
                "degraded": os.path.relpath(out_2x, split_dir).replace("\\", "/"),
            })
        out_4x = deg_dir / f"{name}_4x_clean.mp4"
        degrade(ref_path, out_4x, 4, "clean")
        items_4x.append({
            "name": f"{name}_4x",
            "degradation": "clean_4x",
            "reference": rel_ref,
            "degraded": os.path.relpath(out_4x, split_dir).replace("\\", "/"),
        })
        out_8x = deg_dir / f"{name}_8x_clean.mp4"
        degrade(ref_path, out_8x, 8, "clean")
        items_8x.append({
            "name": f"{name}_8x",
            "degradation": "clean_8x",
            "reference": rel_ref,
            "degraded": os.path.relpath(out_8x, split_dir).replace("\\", "/"),
        })

    (split_dir / "manifest.json").write_text(json.dumps({"split": split, "scale": 2, "items": items_2x}, indent=2))
    (split_dir / "scale4").mkdir(exist_ok=True)
    # Write relative paths for 4x/8x manifests inside split_dir/scale4 and split_dir/scale8
    for sub, items, sc in (("scale4", items_4x, 4), ("scale8", items_8x, 8)):
        sub_dir = split_dir / sub
        sub_dir.mkdir(exist_ok=True)
        fixed = []
        for it in items:
            fixed.append({
                "name": it["name"],
                "degradation": it["degradation"],
                "reference": "../" + it["reference"],
                "degraded": "../" + it["degraded"],
            })
        (sub_dir / "manifest.json").write_text(json.dumps({"split": split, "scale": sc, "items": fixed}, indent=2))

    print(f"[dataset] generated '{split}' in {split_dir} ({len(items_2x)} 2x pairs, {len(items_4x)} 4x, {len(items_8x)} 8x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "datasets"))
    args = ap.parse_args()
    if not FFMPEG.exists():
        sys.exit(f"FFmpeg not found at {FFMPEG}")
    base = Path(args.out)
    build_split(base / "dev", "dev", args.frames)
    build_split(base / "eval", "eval", args.frames)


if __name__ == "__main__":
    main()
