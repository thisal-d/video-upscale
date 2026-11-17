"""Built-in Dola AI and Gemini Sparkle watermark removal (self-contained, no external .py files required)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


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
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _compute_roi(
    width: int,
    height: int,
    defaults: tuple[float, float, float, float],
    roi_override: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int]:
    if roi_override is None:
        rx1, ry1, rx2, ry2 = defaults
        x1, y1, x2, y2 = int(width * rx1), int(height * ry1), int(width * rx2), int(height * ry2)
    elif all(0.0 <= v <= 1.0 for v in roi_override):
        rx1, ry1, rx2, ry2 = roi_override
        x1, y1, x2, y2 = int(width * rx1), int(height * ry1), int(width * rx2), int(height * ry2)
    else:
        x1, y1, x2, y2 = (int(round(v)) for v in roi_override)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


# ---------------- Dola AI Watermark Removal ----------------
def _inpaint_dola_frame(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    patch = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    local_bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_bg)
    diff = cv2.subtract(gray, local_bg)

    _, mask_diff = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    _, mask_bright = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask_bright) < mask_bright.size * 0.5:
        roi_mask = cv2.bitwise_or(mask_diff, mask_bright)
    else:
        roi_mask = mask_diff

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi_mask = cv2.dilate(roi_mask, kernel_dilate, iterations=2)
    if not np.any(roi_mask):
        return frame

    pad = 16
    py1, py2 = max(0, y1 - pad), min(frame.shape[0], y2 + pad)
    px1, px2 = max(0, x1 - pad), min(frame.shape[1], x2 + pad)
    crop_mask = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
    crop_mask[(y1 - py1):(y2 - py1), (x1 - px1):(x2 - px1)] = roi_mask

    inpainted = cv2.inpaint(frame[py1:py2, px1:px2], crop_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    out = frame.copy()
    out[py1:py2, px1:px2] = inpainted
    return out


# ---------------- Gemini Sparkle Watermark Removal ----------------
def _gemini_bg_diff(gray: np.ndarray) -> np.ndarray:
    roi_h, roi_w = gray.shape[:2]
    ksize = max(25, int(min(roi_w, roi_h) * 0.65) | 1)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    local_bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_bg)
    return cv2.subtract(gray, local_bg)


def _build_gemini_static_mask(
    cap: cv2.VideoCapture, roi: tuple[int, int, int, int], sample_count: int = 15
) -> np.ndarray | None:
    x1, y1, x2, y2 = roi
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diffs: list[np.ndarray] = []

    if total_frames > 0:
        count = min(sample_count, total_frames)
        for idx in np.linspace(0, total_frames - 1, count, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                diffs.append(_gemini_bg_diff(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)))
    else:
        for _ in range(sample_count):
            ret, frame = cap.read()
            if not ret:
                break
            diffs.append(_gemini_bg_diff(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not diffs:
        return None

    median_diff = np.median(np.stack(diffs, axis=0), axis=0).astype(np.uint8)
    _, static_mask = cv2.threshold(median_diff, 5, 255, cv2.THRESH_BINARY)
    static_mask = cv2.dilate(static_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return static_mask if np.any(static_mask) else None


def _inpaint_gemini_frame(
    frame: np.ndarray, roi: tuple[int, int, int, int], static_mask: np.ndarray | None
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    patch = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    diff = _gemini_bg_diff(gray)

    _, mask_diff = cv2.threshold(diff, 6, 255, cv2.THRESH_BINARY)
    _, mask_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask_bright) < mask_bright.size * 0.5:
        roi_mask = cv2.bitwise_or(mask_diff, mask_bright)
    else:
        roi_mask = mask_diff

    if static_mask is not None:
        roi_mask = cv2.bitwise_or(roi_mask, static_mask)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    roi_mask = cv2.dilate(roi_mask, kernel_dilate, iterations=2)
    if not np.any(roi_mask):
        return frame

    pad = 16
    py1, py2 = max(0, y1 - pad), min(frame.shape[0], y2 + pad)
    px1, px2 = max(0, x1 - pad), min(frame.shape[1], x2 + pad)
    crop_mask = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
    crop_mask[(y1 - py1):(y2 - py1), (x1 - px1):(x2 - px1)] = roi_mask

    inpainted = cv2.inpaint(frame[py1:py2, px1:px2], crop_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    out = frame.copy()
    out[py1:py2, px1:px2] = inpainted
    return out


def _gemini_delogo_box(
    width: int,
    height: int,
    roi: tuple[int, int, int, int],
    static_mask: np.ndarray | None,
    roi_override: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    if roi_override is None and static_mask is not None and np.any(static_mask):
        bx, by, bw, bh = cv2.boundingRect(static_mask)
        pad = 6
        dx, dy, dw, dh = x1 + bx - pad, y1 + by - pad, bw + 2 * pad, bh + 2 * pad
    elif roi_override is None:
        dx, dy = int(width * 0.79), int(height * 0.895)
        dw, dh = int(width * 0.88) - dx, int(height * 0.96) - dy
    else:
        dx, dy, dw, dh = x1, y1, x2 - x1, y2 - y1

    dx = max(1, min(dx, width - 5))
    dy = max(1, min(dy, height - 5))
    dw = max(4, min(dw, width - dx - 1))
    dh = max(4, min(dh, height - dy - 1))
    return dx, dy, dw, dh


def run_watermark_removal(
    input_path: str,
    output_path: str,
    kind: str,  # "dola" | "gemini"
    method: str = "inpaint",  # "inpaint" | "delogo"
    roi_override: tuple[float, float, float, float] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Run built-in Dola or Gemini watermark removal with progress & cancellation."""
    if kind not in ("dola", "gemini"):
        raise ValueError(f"Unknown watermark remover: {kind}")

    ffmpeg_bin = find_ffmpeg_exe()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for watermark removal: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    defaults = (0.80, 0.935, 0.975, 0.985) if kind == "dola" else (0.77, 0.885, 0.96, 0.975)
    roi = _compute_roi(width, height, defaults, roi_override)

    static_mask = None
    if kind == "gemini":
        if progress_cb:
            progress_cb(0, max(1, total_frames), "Analyzing Gemini watermark mask...")
        static_mask = _build_gemini_static_mask(cap, roi)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    if method == "delogo":
        cap.release()
        if not ffmpeg_bin:
            raise RuntimeError("FFmpeg is required for the 'delogo' watermark removal method.")
        if kind == "gemini":
            dx, dy, dw, dh = _gemini_delogo_box(width, height, roi, static_mask, roi_override)
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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=creationflags)
        if progress_cb:
            progress_cb(2, 2, f"{kind} delogo complete")
        return

    tmp_fd, tmp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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
            cleaned = (
                _inpaint_gemini_frame(frame, roi, static_mask)
                if kind == "gemini"
                else _inpaint_dola_frame(frame, roi)
            )
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
            cmd = [
                ffmpeg_bin, "-y", "-i", tmp_video, "-i", input_path,
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-shortest", output_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=creationflags)
        else:
            shutil.move(tmp_video, output_path)
    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
