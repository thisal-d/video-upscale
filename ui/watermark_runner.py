"""Unified wrapper around dola-watermark-remover.py and gemini-watermark-remover.py
with bundled FFmpeg resolution, live progress callbacks, and cancellation support.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import cv2


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _load_script_module(filename: str, mod_name: str):
    candidates = [
        _app_root() / filename,
        _app_root() / "_internal" / filename,
        Path(__file__).resolve().parent.parent / filename,
    ]
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location(mod_name, str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise FileNotFoundError(f"Could not locate {filename} in {[str(c) for c in candidates]}")


def find_ffmpeg_exe() -> str | None:
    root = _app_root()
    candidates = [
        root / "ffmpeg.exe",
        root / "bin" / "ffmpeg.exe",
        root / "third_party" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception:
        pass
    return None


def parse_roi_text(text: str) -> tuple[float, float, float, float] | None:
    s = text.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    if len(parts) != 4:
        raise ValueError("ROI must be 4 numbers: x1 y1 x2 y2 (either 0..1 ratios or pixels)")
    vals = tuple(float(p) for p in parts)
    return vals  # type: ignore[return-value]


def run_watermark_removal(
    input_path: str,
    output_path: str,
    kind: str,  # "dola" | "gemini"
    method: str = "inpaint",  # "inpaint" | "delogo"
    roi_override: tuple[float, float, float, float] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Run Dola or Gemini watermark removal with progress & cancellation."""
    if kind == "dola":
        mod = _load_script_module("dola-watermark-remover.py", "vu_dola_wm")
    elif kind == "gemini":
        mod = _load_script_module("gemini-watermark-remover.py", "vu_gemini_wm")
    else:
        raise ValueError(f"Unknown watermark remover: {kind}")

    ffmpeg_bin = find_ffmpeg_exe() or mod.find_ffmpeg()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for watermark removal: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = mod.compute_watermark_roi(width, height, roi_override)

    static_mask = None
    if kind == "gemini":
        if progress_cb:
            progress_cb(0, max(1, total_frames), f"Analyzing {kind} watermark mask...")
        static_mask = mod.build_static_watermark_mask(cap, roi)

    if method == "delogo":
        cap.release()
        if not ffmpeg_bin:
            raise RuntimeError("FFmpeg is required for the 'delogo' watermark removal method.")
        if kind == "gemini":
            dx, dy, dw, dh = mod._compute_delogo_box(width, height, roi, static_mask, roi_override)
        else:
            x1, y1, x2, y2 = roi
            dx, dy, dw, dh = x1, y1, x2 - x1, y2 - y1
        if progress_cb:
            progress_cb(1, 2, f"Running FFmpeg delogo ({kind})...")
        cmd = [
            ffmpeg_bin, "-y", "-i", input_path,
            "-vf", f"delogo=x={dx}:y={dy}:w={dw}:h={dh}",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "copy", output_path,
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=creationflags)
        if progress_cb:
            progress_cb(2, 2, f"{kind} delogo complete")
        return

    tmp_fd, tmp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open temporary VideoWriter: {tmp_video}")

        frame_idx = 0
        while True:
            if cancel_check and cancel_check():
                cap.release()
                writer.release()
                raise InterruptedError("Cancelled during watermark removal")
            ret, frame = cap.read()
            if not ret:
                break
            if kind == "gemini":
                cleaned = mod.inpaint_watermark_frame(frame, roi, static_mask=static_mask)
            else:
                cleaned = mod.inpaint_watermark_frame(frame, roi)
            writer.write(cleaned)
            frame_idx += 1
            if progress_cb and (frame_idx % 5 == 0 or frame_idx == total_frames):
                progress_cb(frame_idx, max(1, total_frames),
                            f"Removing {kind} watermark: {frame_idx}/{total_frames}")

        cap.release()
        writer.release()

        if ffmpeg_bin:
            if progress_cb:
                progress_cb(total_frames, max(1, total_frames), f"Muxing {kind} cleaned video + audio...")
            # Run muxing cleanly without popping a console window
            cmd = [
                ffmpeg_bin, "-y", "-i", tmp_video, "-i", input_path,
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-shortest", output_path,
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=creationflags)
        else:
            shutil.move(tmp_video, output_path)
    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
