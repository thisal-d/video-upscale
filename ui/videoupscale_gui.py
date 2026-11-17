"""VideoUpscale Studio — Desktop GUI for VideoUpscale & Watermark Removal."""

from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from watermark_runner import parse_roi_text, run_watermark_removal


def find_cli_exe() -> Path | None:
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
    else:
        app_dir = Path(__file__).resolve().parent.parent
    candidates = [
        app_dir / "videoupscale.exe",
        app_dir / "bin" / "videoupscale.exe",
        app_dir / "build" / "src" / "videoupscale.exe",
        Path(__file__).resolve().parent.parent / "build" / "src" / "videoupscale.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


ALGORITHMS = [
    ("mfsr", "MFSR (5-Frame Super-Res — best detail & motion)", True),
    ("ibp", "IBP (Single-Frame Iterative Back-Projection — fast & crisp)", False),
    ("edge", "Edge-Aware (Structure-tensor interpolation — very fast)", False),
    ("lanczos", "Lanczos (Sharp classical resampling)", False),
    ("bicubic", "Bicubic (Standard smooth resampling)", False),
    ("bilinear", "Bilinear (Fast linear resampling)", False),
    ("nearest", "Nearest Neighbor (Pixelated / crisp pixels)", False),
]

SCALES = [
    (2, "2× (Recommended)", True),
    (4, "4× (High Res)", False),
    (8, "8× (Experimental)", False),
]


class VideoUpscaleApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("VideoUpscale Studio v0.1.0")
        self.root.geometry("980x780")
        self.root.minsize(860, 680)
        self._set_window_icon()

        self.cli_exe = find_cli_exe()
        self.ui_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.active_proc: subprocess.Popen | None = None

        self.input_files: list[str] = []

        self._setup_style()
        self._build_vars()
        self._build_ui()
        self._update_job_summary()
        self.root.after(100, self._poll_queue)

        if not self.cli_exe:
            self._log("[WARN] videoupscale.exe not found next to app or in build/src/.")
        else:
            self._log(f"[Ready] Engine: {self.cli_exe}")

    def _set_window_icon(self) -> None:
        base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
        for p in (base / "assets" / "logo.ico", base / "_internal" / "logo.ico", base / "logo.ico"):
            if p.is_file():
                try:
                    self.root.iconbitmap(str(p))
                except Exception:
                    pass
                break

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            if "vista" in style.theme_names():
                style.theme_use("vista")
            elif "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#555555")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_vars(self) -> None:
        self.output_dir = tk.StringVar(value=str(Path.home() / "Videos" / "VideoUpscale_Output"))

        # Watermark removers
        self.wm_dola_on = tk.BooleanVar(value=False)
        self.wm_dola_method = tk.StringVar(value="inpaint")
        self.wm_dola_roi = tk.StringVar(value="")

        self.wm_gemini_on = tk.BooleanVar(value=False)
        self.wm_gemini_method = tk.StringVar(value="inpaint")
        self.wm_gemini_roi = tk.StringVar(value="")

        self.wm_save_clean = tk.BooleanVar(value=False)

        # Scales & Algorithms (multi-select)
        self.scale_vars: dict[int, tk.BooleanVar] = {
            s: tk.BooleanVar(value=default) for s, _, default in SCALES
        }
        self.algo_vars: dict[str, tk.BooleanVar] = {
            key: tk.BooleanVar(value=default) for key, _, default in ALGORITHMS
        }

        # Quality & encoding settings
        self.preset_var = tk.StringVar(value="quality")
        self.denoise_var = tk.StringVar(value="auto")
        self.deblock_var = tk.StringVar(value="auto")
        self.sharpen_var = tk.DoubleVar(value=0.20)
        self.halo_var = tk.BooleanVar(value=True)
        self.progressive_var = tk.BooleanVar(value=False)

        self.codec_var = tk.StringVar(value="libx264")
        self.crf_var = tk.IntVar(value=16)
        self.enc_preset_var = tk.StringVar(value="slow")
        self.copy_audio_var = tk.BooleanVar(value=True)
        self.backend_var = tk.StringVar(value="auto")
        self.preview_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=True)

        self.summary_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")
        self.overall_pct = tk.DoubleVar(value=0.0)
        self.job_pct = tk.DoubleVar(value=0.0)

    def _build_ui(self) -> None:
        # Top header
        hdr = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="VideoUpscale Studio", style="Title.TLabel").pack(side="left")
        ttk.Label(
            hdr,
            text="Classical Multi-Frame Super-Resolution + Batch Comparison + Watermark Removal",
            style="Sub.TLabel",
        ).pack(side="left", padx=(12, 0), pady=(3, 0))

        # Scrollable main container
        canvas_wrap = ttk.Frame(self.root)
        canvas_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        canvas = tk.Canvas(canvas_wrap, highlightthickness=0)
        vscroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, padding=(2, 2, 8, 6))
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ---- 1. Input & Output ----
        io_lf = ttk.LabelFrame(body, text="1. Input Videos & Output Folder", style="Section.TLabelframe", padding=8)
        io_lf.pack(fill="x", pady=(0, 8))

        in_row = ttk.Frame(io_lf)
        in_row.pack(fill="x")
        self.input_listbox = tk.Listbox(in_row, height=3, selectmode="extended")
        self.input_listbox.pack(side="left", fill="x", expand=True)

        in_btns = ttk.Frame(in_row)
        in_btns.pack(side="left", padx=(8, 0), fill="y")
        ttk.Button(in_btns, text="Add Video(s)...", command=self._add_videos).pack(fill="x", pady=(0, 3))
        ttk.Button(in_btns, text="Inspect Selected", command=self._inspect_selected).pack(fill="x", pady=(0, 3))
        ttk.Button(in_btns, text="Remove / Clear", command=self._clear_inputs).pack(fill="x")

        out_row = ttk.Frame(io_lf)
        out_row.pack(fill="x", pady=(8, 0))
        ttk.Label(out_row, text="Output Folder:").pack(side="left")
        ttk.Entry(out_row, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(out_row, text="Browse...", command=self._browse_outdir).pack(side="left", padx=(0, 4))
        ttk.Button(out_row, text="Open Folder", command=self._open_outdir).pack(side="left")

        # ---- 2. Watermark Removers ----
        wm_lf = ttk.LabelFrame(
            body,
            text="2. Watermark Removers (Optional Pre-Processing — Enable / Disable)",
            style="Section.TLabelframe",
            padding=8,
        )
        wm_lf.pack(fill="x", pady=(0, 8))

        # Dola row
        d_row = ttk.Frame(wm_lf)
        d_row.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(
            d_row, text="Enable Dola AI Watermark Remover", variable=self.wm_dola_on,
            command=self._update_job_summary
        ).pack(side="left")
        ttk.Label(d_row, text="Method:").pack(side="left", padx=(16, 4))
        ttk.Combobox(d_row, textvariable=self.wm_dola_method, values=["inpaint", "delogo"],
                     state="readonly", width=9).pack(side="left")
        ttk.Label(d_row, text="Custom ROI (optional x1 y1 x2 y2):").pack(side="left", padx=(14, 4))
        ttk.Entry(d_row, textvariable=self.wm_dola_roi, width=20).pack(side="left")

        # Gemini row
        g_row = ttk.Frame(wm_lf)
        g_row.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(
            g_row, text="Enable Gemini Sparkle Watermark Remover", variable=self.wm_gemini_on,
            command=self._update_job_summary
        ).pack(side="left")
        ttk.Label(g_row, text="Method:").pack(side="left", padx=(16, 4))
        ttk.Combobox(g_row, textvariable=self.wm_gemini_method, values=["inpaint", "delogo"],
                     state="readonly", width=9).pack(side="left")
        ttk.Label(g_row, text="Custom ROI (optional x1 y1 x2 y2):").pack(side="left", padx=(14, 4))
        ttk.Entry(g_row, textvariable=self.wm_gemini_roi, width=20).pack(side="left")

        ttk.Checkbutton(
            wm_lf,
            text="Also save 1× watermark-cleaned video to output folder (useful when comparing)",
            variable=self.wm_save_clean,
            command=self._update_job_summary,
        ).pack(anchor="w")

        # ---- 3. Scales & Algorithms (Multi-output) ----
        sa_frame = ttk.Frame(body)
        sa_frame.pack(fill="x", pady=(0, 8))

        sc_lf = ttk.LabelFrame(sa_frame, text="3. Output Scale(s)", style="Section.TLabelframe", padding=8)
        sc_lf.pack(side="left", fill="both", expand=True, padx=(0, 4))
        for s, label, _ in SCALES:
            ttk.Checkbutton(
                sc_lf, text=label, variable=self.scale_vars[s], command=self._update_job_summary
            ).pack(anchor="w", pady=1)
        ttk.Separator(sc_lf, orient="horizontal").pack(fill="x", pady=6)
        ttk.Checkbutton(
            sc_lf, text="Progressive 2× stages (for 4× / 8×)", variable=self.progressive_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            sc_lf, text="Preview Mode (first 24 frames only)", variable=self.preview_var
        ).pack(anchor="w", pady=(2, 0))

        al_lf = ttk.LabelFrame(
            sa_frame,
            text="4. Algorithm(s) — Select Multiple for Side-by-Side Outputs",
            style="Section.TLabelframe",
            padding=8,
        )
        al_lf.pack(side="left", fill="both", expand=True, padx=(4, 0))
        for key, label, _ in ALGORITHMS:
            ttk.Checkbutton(
                al_lf, text=label, variable=self.algo_vars[key], command=self._update_job_summary
            ).pack(anchor="w", pady=1)

        # ---- 4. Quality & Encoder Settings ----
        set_lf = ttk.LabelFrame(body, text="5. Enhancement & Encoding Settings", style="Section.TLabelframe", padding=8)
        set_lf.pack(fill="x", pady=(0, 8))

        r1 = ttk.Frame(set_lf)
        r1.pack(fill="x", pady=(0, 6))
        ttk.Label(r1, text="Profile Preset:").pack(side="left")
        preset_cb = ttk.Combobox(
            r1, textvariable=self.preset_var, values=["quality", "balanced", "fast"],
            state="readonly", width=11
        )
        preset_cb.pack(side="left", padx=(6, 16))
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_changed)

        ttk.Label(r1, text="Denoise:").pack(side="left")
        ttk.Combobox(r1, textvariable=self.denoise_var, values=["auto", "on", "off"],
                     state="readonly", width=7).pack(side="left", padx=(4, 12))

        ttk.Label(r1, text="Deblock:").pack(side="left")
        ttk.Combobox(r1, textvariable=self.deblock_var, values=["auto", "on", "off"],
                     state="readonly", width=7).pack(side="left", padx=(4, 12))

        ttk.Label(r1, text="Sharpen:").pack(side="left")
        ttk.Scale(r1, from_=0.0, to=1.0, variable=self.sharpen_var, length=100,
                  command=lambda _: self.sharpen_lbl.config(text=f"{self.sharpen_var.get():.2f}")).pack(side="left", padx=4)
        self.sharpen_lbl = ttk.Label(r1, text=f"{self.sharpen_var.get():.2f}", width=4)
        self.sharpen_lbl.pack(side="left")
        ttk.Checkbutton(r1, text="Halo Suppression", variable=self.halo_var).pack(side="left", padx=(8, 0))

        r2 = ttk.Frame(set_lf)
        r2.pack(fill="x")
        ttk.Label(r2, text="Codec:").pack(side="left")
        ttk.Combobox(r2, textvariable=self.codec_var, values=["libx264", "libx265"],
                     state="readonly", width=9).pack(side="left", padx=(4, 12))

        ttk.Label(r2, text="CRF (lower=better):").pack(side="left")
        ttk.Spinbox(r2, from_=0, to=35, textvariable=self.crf_var, width=4).pack(side="left", padx=(4, 12))

        ttk.Label(r2, text="Encoder Speed:").pack(side="left")
        ttk.Combobox(r2, textvariable=self.enc_preset_var,
                     values=["ultrafast", "veryfast", "fast", "medium", "slow"],
                     state="readonly", width=10).pack(side="left", padx=(4, 12))

        ttk.Label(r2, text="Backend:").pack(side="left")
        ttk.Combobox(r2, textvariable=self.backend_var, values=["auto", "openmp", "cpu"],
                     state="readonly", width=8).pack(side="left", padx=(4, 12))

        ttk.Checkbutton(r2, text="Preserve Audio", variable=self.copy_audio_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(r2, text="Overwrite", variable=self.overwrite_var).pack(side="left")

        # ---- 5. Run & Progress ----
        run_lf = ttk.LabelFrame(body, text="6. Run & Progress", style="Section.TLabelframe", padding=8)
        run_lf.pack(fill="both", expand=True)

        top_run = ttk.Frame(run_lf)
        top_run.pack(fill="x", pady=(0, 6))
        ttk.Label(top_run, textvariable=self.summary_var, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.cancel_btn = ttk.Button(top_run, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="right", padx=(6, 0))
        self.start_btn = ttk.Button(
            top_run, text="Start Processing", style="Accent.TButton", command=self._start
        )
        self.start_btn.pack(side="right")

        ttk.Label(run_lf, textvariable=self.status_var).pack(anchor="w")
        ttk.Progressbar(run_lf, variable=self.job_pct, maximum=100.0).pack(fill="x", pady=(3, 4))
        ttk.Progressbar(run_lf, variable=self.overall_pct, maximum=100.0).pack(fill="x", pady=(0, 6))

        self.log_box = scrolledtext.ScrolledText(run_lf, height=9, state="disabled", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)

    # ---- UI Callbacks ----
    def _on_preset_changed(self, _event=None) -> None:
        p = self.preset_var.get()
        if p == "quality":
            self.crf_var.set(16)
            self.enc_preset_var.set("slow")
            self.sharpen_var.set(0.20)
        elif p == "balanced":
            self.crf_var.set(18)
            self.enc_preset_var.set("medium")
            self.sharpen_var.set(0.15)
        elif p == "fast":
            self.crf_var.set(18)
            self.enc_preset_var.set("veryfast")
            self.sharpen_var.set(0.00)
        self.sharpen_lbl.config(text=f"{self.sharpen_var.get():.2f}")

    def _add_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select Input Video(s)",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi *.webm"), ("All files", "*.*")],
        )
        if not paths:
            return
        added = False
        for p in paths:
            if p not in self.input_files:
                self.input_files.append(p)
                self.input_listbox.insert("end", p)
                added = True
        if added and len(self.input_files) == 1:
            # Default output folder next to the first added video
            self.output_dir.set(str(Path(self.input_files[0]).parent / "upscaled_outputs"))
        self._update_job_summary()

    def _clear_inputs(self) -> None:
        sel = list(self.input_listbox.curselection())
        if sel:
            for idx in reversed(sel):
                self.input_listbox.delete(idx)
                del self.input_files[idx]
        else:
            self.input_listbox.delete(0, "end")
            self.input_files.clear()
        self._update_job_summary()

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_dir.set(d)

    def _open_outdir(self) -> None:
        d = Path(self.output_dir.get())
        d.mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))

    def _inspect_selected(self) -> None:
        if not self.cli_exe:
            messagebox.showerror("Engine Missing", "videoupscale.exe was not found.")
            return
        sel = self.input_listbox.curselection()
        if sel:
            path = self.input_files[sel[0]]
        elif self.input_files:
            path = self.input_files[0]
        else:
            messagebox.showinfo("No Video", "Add a video first to inspect it.")
            return

        self._log(f"[Analyze] Inspecting {path} ...")

        def _worker():
            try:
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                res = subprocess.run(
                    [str(self.cli_exe), "analyze", path, "--max-frames", "30", "--json", "-q"],
                    capture_output=True, text=True, check=True, creationflags=creationflags
                )
                data = json.loads(res.stdout)
                v = data["video"]
                a = data["analysis"]
                audio_desc = ", ".join(
                    f"{s['codec']} {s['sample_rate']}Hz {s['channels']}ch" for s in v.get("audio_streams", [])
                ) or "none"
                adv = "\n  • ".join(a.get("advisories", [])) or "None (source looks healthy)"
                msg = (
                    f"File: {Path(path).name}\n"
                    f"Resolution: {v['width']} × {v['height']}  |  Codec: {v['codec']} ({v['pixel_format']})\n"
                    f"FPS: {v['fps']:.2f}  |  Duration: {v['duration_sec']:.2f}s  |  Est. Frames: {v['frame_count_estimate']}\n"
                    f"Audio: {audio_desc}\n\n"
                    f"Quality Analysis ({a['frames_analyzed']} sampled frames):\n"
                    f"  Noise Sigma: {a['mean_noise_sigma']:.4f}\n"
                    f"  Blur Score:  {a['mean_blur_score']:.3f}  (0=sharp, 1=blurry)\n"
                    f"  Blockiness:  {a['mean_blockiness']:.3f}  (>0.12 triggers deblocking)\n"
                    f"  Scene Cuts:  {len(a['scene_cut_frames'])}\n\n"
                    f"Advisories:\n  • {adv}"
                )
                self.ui_queue.put(("msgbox", ("Video Analysis", msg)))
            except Exception as exc:
                self.ui_queue.put(("log", f"[Analyze Error] {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _selected_scales(self) -> list[int]:
        return [s for s, v in self.scale_vars.items() if v.get()]

    def _selected_algos(self) -> list[str]:
        return [k for k, v in self.algo_vars.items() if v.get()]

    def _update_job_summary(self) -> None:
        nv = len(self.input_files)
        ns = len(self._selected_scales())
        na = len(self._selected_algos())
        total = nv * ns * na
        wm = []
        if self.wm_dola_on.get():
            wm.append("Dola")
        if self.wm_gemini_on.get():
            wm.append("Gemini")
        wm_txt = f" | Watermark removal: {' + '.join(wm)}" if wm else ""
        self.summary_var.set(
            f"Queue: {nv} video(s) × {ns} scale(s) × {na} algorithm(s) = {total} upscaled output(s){wm_txt}"
        )

    def _log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "job_pct":
                    self.job_pct.set(payload)
                elif kind == "overall_pct":
                    self.overall_pct.set(payload)
                elif kind == "msgbox":
                    title, text = payload
                    messagebox.showinfo(title, text)
                elif kind == "done":
                    self._on_finished(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _start(self) -> None:
        if not self.cli_exe:
            messagebox.showerror("Engine Missing", "videoupscale.exe was not found.")
            return
        if not self.input_files:
            messagebox.showwarning("No Input", "Please add at least one input video.")
            return
        scales = self._selected_scales()
        algos = self._selected_algos()
        if not scales or not algos:
            messagebox.showwarning("Selection Required", "Select at least one Scale and one Algorithm.")
            return
        try:
            dola_roi = parse_roi_text(self.wm_dola_roi.get()) if self.wm_dola_on.get() else None
            gemini_roi = parse_roi_text(self.wm_gemini_roi.get()) if self.wm_gemini_on.get() else None
        except ValueError as exc:
            messagebox.showerror("Invalid ROI", str(exc))
            return

        out_dir = Path(self.output_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "inputs": list(self.input_files),
            "out_dir": out_dir,
            "scales": scales,
            "algos": algos,
            "wm_dola": self.wm_dola_on.get(),
            "wm_dola_method": self.wm_dola_method.get(),
            "wm_dola_roi": dola_roi,
            "wm_gemini": self.wm_gemini_on.get(),
            "wm_gemini_method": self.wm_gemini_method.get(),
            "wm_gemini_roi": gemini_roi,
            "wm_save_clean": self.wm_save_clean.get(),
            "preset": self.preset_var.get(),
            "denoise": self.denoise_var.get(),
            "deblock": self.deblock_var.get(),
            "sharpen": float(self.sharpen_var.get()),
            "halo": self.halo_var.get(),
            "progressive": self.progressive_var.get(),
            "codec": self.codec_var.get(),
            "crf": int(self.crf_var.get()),
            "enc_preset": self.enc_preset_var.get(),
            "copy_audio": self.copy_audio_var.get(),
            "backend": self.backend_var.get(),
            "preview": self.preview_var.get(),
            "overwrite": self.overwrite_var.get(),
        }

        self.cancel_event.clear()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.overall_pct.set(0.0)
        self.job_pct.set(0.0)
        self._log("=" * 64)
        self._log(f"[Batch Start] {len(cfg['inputs'])} input(s), scales={scales}, algos={algos}")

        self.worker_thread = threading.Thread(target=self._run_batch, args=(cfg,), daemon=True)
        self.worker_thread.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling...")
        self._log("[Cancel] Cancellation requested...")
        proc = self.active_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _run_batch(self, cfg: dict) -> None:
        t0 = time.time()
        outputs_created: list[str] = []
        errors: list[str] = []
        tmp_files: list[Path] = []

        # Calculate total steps: watermark steps per video + upscale jobs
        wm_steps_per_video = (1 if cfg["wm_dola"] else 0) + (1 if cfg["wm_gemini"] else 0)
        total_steps = len(cfg["inputs"]) * (wm_steps_per_video + len(cfg["scales"]) * len(cfg["algos"]))
        done_steps = 0

        def advance_step():
            nonlocal done_steps
            done_steps += 1
            self.ui_queue.put(("overall_pct", 100.0 * done_steps / max(1, total_steps)))

        try:
            for in_path_str in cfg["inputs"]:
                if self.cancel_event.is_set():
                    break
                in_path = Path(in_path_str)
                stem = in_path.stem
                effective_input = in_path
                wm_tag = ""

                # Optional Watermark Pre-Pass (runs once per input video)
                if cfg["wm_dola"] or cfg["wm_gemini"]:
                    wm_steps: list[tuple[str, str, tuple | None]] = []
                    if cfg["wm_dola"]:
                        wm_steps.append(("dola", cfg["wm_dola_method"], cfg["wm_dola_roi"]))
                    if cfg["wm_gemini"]:
                        wm_steps.append(("gemini", cfg["wm_gemini_method"], cfg["wm_gemini_roi"]))

                    wm_tag = "_clean"
                    cur_src = str(in_path)
                    for idx, (kind, method, roi) in enumerate(wm_steps):
                        if self.cancel_event.is_set():
                            break
                        is_last_wm = idx == len(wm_steps) - 1
                        if is_last_wm and cfg["wm_save_clean"]:
                            dest = cfg["out_dir"] / f"{stem}_clean.mp4"
                        else:
                            fd, tmp_name = tempfile.mkstemp(suffix=f"_{kind}.mp4")
                            os.close(fd)
                            dest = Path(tmp_name)
                            tmp_files.append(dest)

                        self.ui_queue.put(("status", f"[{in_path.name}] Removing {kind} watermark ({method})..."))
                        self.ui_queue.put(("job_pct", 0.0))
                        self.ui_queue.put(("log", f"[Watermark] {in_path.name} -> {kind} ({method})"))

                        def _wm_prog(cur: int, tot: int, msg: str) -> None:
                            self.ui_queue.put(("job_pct", 100.0 * cur / max(1, tot)))
                            self.ui_queue.put(("status", msg))

                        run_watermark_removal(
                            cur_src,
                            str(dest),
                            kind=kind,
                            method=method,
                            roi_override=roi,
                            progress_cb=_wm_prog,
                            cancel_check=self.cancel_event.is_set,
                        )
                        cur_src = str(dest)
                        effective_input = dest
                        if is_last_wm and cfg["wm_save_clean"]:
                            outputs_created.append(str(dest))
                            self.ui_queue.put(("log", f"[OK] Saved 1x cleaned video: {dest}"))
                        advance_step()

                if self.cancel_event.is_set():
                    break

                # Upscale matrix: scales x algos
                for scale in cfg["scales"]:
                    for algo in cfg["algos"]:
                        if self.cancel_event.is_set():
                            break
                        prog_tag = "_prog" if (cfg["progressive"] and scale > 2) else ""
                        prev_tag = "_preview" if cfg["preview"] else ""
                        out_name = f"{stem}{wm_tag}_{algo}_{scale}x{prog_tag}{prev_tag}.mp4"
                        out_path = cfg["out_dir"] / out_name

                        # Write per-job JSON config so halo_suppression and preset are exact
                        job_cfg = {
                            "preset": cfg["preset"],
                            "scale": scale,
                            "upscaler": algo,
                            "progressive": cfg["progressive"],
                            "backend": cfg["backend"],
                            "enhance": {
                                "denoise": cfg["denoise"],
                                "deblock": cfg["deblock"],
                                "sharpen": cfg["sharpen"],
                                "halo_suppression": cfg["halo"],
                            },
                            "encoder": {
                                "codec": cfg["codec"],
                                "crf": cfg["crf"],
                                "preset": cfg["enc_preset"],
                                "copy_audio": cfg["copy_audio"],
                            },
                        }
                        fd, cfg_tmp = tempfile.mkstemp(suffix=".json")
                        os.close(fd)
                        cfg_path = Path(cfg_tmp)
                        cfg_path.write_text(json.dumps(job_cfg, indent=2), encoding="utf-8")
                        tmp_files.append(cfg_path)

                        cmd = [
                            str(self.cli_exe), "upscale", str(effective_input),
                            "--output", str(out_path),
                            "--config", str(cfg_path),
                        ]
                        if cfg["preview"]:
                            cmd += ["--max-frames", "24"]
                        if cfg["overwrite"]:
                            cmd.append("-y")

                        self.ui_queue.put(("job_pct", 0.0))
                        self.ui_queue.put(("status", f"[{in_path.name}] Upscaling {scale}x ({algo})..."))
                        self.ui_queue.put(("log", f"[Upscale] {out_name} (scale={scale}x, algo={algo})"))

                        rc = self._run_cli_with_progress(cmd, f"{in_path.name} [{algo} {scale}x]")
                        if rc == 0:
                            outputs_created.append(str(out_path))
                            self.ui_queue.put(("log", f"[OK] Created: {out_path}"))
                        elif self.cancel_event.is_set():
                            self.ui_queue.put(("log", f"[Cancelled] Stopped during {out_name}"))
                        else:
                            err = f"{out_name} failed with exit code {rc}"
                            errors.append(err)
                            self.ui_queue.put(("log", f"[ERROR] {err}"))
                        advance_step()

        except InterruptedError:
            self.ui_queue.put(("log", "[Cancelled] Processing stopped by user."))
        except Exception as exc:
            errors.append(str(exc))
            self.ui_queue.put(("log", f"[ERROR] {exc}"))
        finally:
            for tf in tmp_files:
                try:
                    if tf.exists():
                        tf.unlink()
                except Exception:
                    pass
            elapsed = time.time() - t0
            self.ui_queue.put(("done", {
                "outputs": outputs_created,
                "errors": errors,
                "cancelled": self.cancel_event.is_set(),
                "elapsed": elapsed,
            }))

    def _run_cli_with_progress(self, cmd: list[str], label: str) -> int:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self.active_proc = proc
        prog_re = re.compile(r"frame\s+(\d+)(?:/(\d+))?\s*(?:\(\s*([\d.]+)%\))?\s*\|\s*([\d.]+)\s*fps")

        buf = bytearray()
        assert proc.stderr is not None
        while True:
            ch = proc.stderr.read(1)
            if not ch:
                break
            if ch in (b"\r", b"\n"):
                line = buf.decode("utf-8", errors="replace").strip()
                buf.clear()
                if line:
                    m = prog_re.search(line)
                    if m:
                        pct = float(m.group(3)) if m.group(3) else 0.0
                        if m.group(3):
                            self.ui_queue.put(("job_pct", pct))
                        self.ui_queue.put(("status", f"{label}: {line}"))
                    else:
                        self.ui_queue.put(("log", f"  {line}"))
            else:
                buf.extend(ch)
        if buf:
            line = buf.decode("utf-8", errors="replace").strip()
            if line:
                self.ui_queue.put(("log", f"  {line}"))

        rc = proc.wait()
        self.active_proc = None
        if rc == 0:
            self.ui_queue.put(("job_pct", 100.0))
        return rc

    def _on_finished(self, info: dict) -> None:
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        outs = info["outputs"]
        errs = info["errors"]
        elapsed = info["elapsed"]

        if info["cancelled"]:
            self.status_var.set(f"Cancelled after {elapsed:.1f}s ({len(outs)} output(s) saved).")
            return
        if errs:
            self.status_var.set(f"Finished with {len(errs)} error(s) in {elapsed:.1f}s.")
            messagebox.showwarning(
                "Completed With Errors",
                f"Created {len(outs)} file(s), encountered {len(errs)} error(s):\n\n" + "\n".join(errs[:5]),
            )
        else:
            self.overall_pct.set(100.0)
            self.status_var.set(f"All done! Created {len(outs)} file(s) in {elapsed:.1f}s.")
            self._log(f"[Complete] {len(outs)} file(s) saved to {self.output_dir.get()} in {elapsed:.1f}s.")
            messagebox.showinfo(
                "Done!",
                f"Successfully created {len(outs)} output file(s) in {elapsed:.1f}s:\n\n"
                + "\n".join(Path(p).name for p in outs[:12])
                + ("\n..." if len(outs) > 12 else ""),
            )


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    VideoUpscaleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
