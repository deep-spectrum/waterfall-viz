"""Render a waterfall (spectrogram) diagram from an iq-sdk IQ recording.

A quick visual sanity-check tool for raw captures stored in the iq-sdk format
(see ``iq-sdk/docs/format.md``). Reads through ``iq_sdk.Receiver`` so it tracks
the recording format. By default only the frames it plots are read (time
striding), so a multi-GB recording renders about as fast as a small one.

Dependencies: iq-sdk, numpy, pyyaml, matplotlib.

Examples:
    # Whole recording, default 1024-pt FFT, striding to <=2000 rows
    python waterfall.py path/to/rec/rx0 -o waterfall.png

    # Zoom into a 20 ms slice starting 10 ms in, with a finer FFT
    python waterfall.py path/to/rec/rx0 --start-sec 0.01 --duration-sec 0.02 --nfft 2048

    # Override the axis calibration (e.g. if meta.yaml is incomplete)
    python waterfall.py path/to/rec/rx0 --sample-rate 50e6 --center-freq 2.4e9
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import yaml
from iq_sdk import Receiver

WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "rect": np.ones,
}


# ---------------------------------------------------------------------------
# Input resolution + axis metadata
# ---------------------------------------------------------------------------


def resolve_rx_dir(path: str) -> str:
    """Resolve `path` to the ``rx*`` directory to hand to ``Receiver``.

    `path` may be: an ``rx*`` directory (has ``meta.yaml``), a recording
    directory containing one or more ``rx*`` subdirs (the first is chosen), or
    a chunk file -- in which case its containing receiver directory is used
    (``Receiver`` reads the whole recording, since it needs ``meta.yaml`` and
    ``ts.f8``).
    """
    if os.path.isfile(path):
        return os.path.dirname(path) or "."

    if not os.path.isdir(path):
        raise SystemExit(f"Path not found: {path}")

    if os.path.exists(os.path.join(path, "meta.yaml")):
        return path

    rx_subdirs = sorted(
        d for d in glob.glob(os.path.join(path, "rx*")) if os.path.isdir(d)
    )
    if rx_subdirs:
        chosen = rx_subdirs[0]
        if len(rx_subdirs) > 1:
            print(
                f"Multiple receivers found; using {os.path.basename(chosen)} "
                f"(of {[os.path.basename(d) for d in rx_subdirs]})"
            )
        return chosen

    raise SystemExit(f"No meta.yaml or rx* subdir found under: {path}")


def load_axis_meta(
    rx_dir: str, sample_rate: float | None, center_freq: float | None
) -> tuple[float, float]:
    """Read sample rate and center frequency for axis labelling.

    ``Receiver`` handles the sample *data* but does not surface the RF
    parameters, so we read them from ``meta.yaml`` (and ``ts.f8``) here.

    Sample rate priority:
      1. ``--sample-rate`` override.
      2. ``ts.f8`` per-capture cadence: ``samples_per_capture /
         median(diff(timestamps))`` -- most reliable; correct on every real
         recording tested.
      3. ``captures * samples_per_capture / parameters.capture_duration``.
         NOTE: in real recordings ``capture_duration`` is the *total* session
         length (not per-capture as the format doc states), so divide the
         total sample count by it.
      4. ``parameters.bandwidth`` (often a filter width below fs; last resort).

    Prints the chosen value and its source so it can be sanity-checked.
    """
    meta = {}
    meta_path = os.path.join(rx_dir, "meta.yaml")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
    params = meta.get("parameters", {}) if isinstance(meta, dict) else {}

    fs = sample_rate
    fs_src = "--sample-rate"
    spc = meta.get("samples_per_capture")
    captures = meta.get("captures")

    if fs is None and spc:
        # 2. Per-capture timestamp cadence (captures are gapless, so
        # consecutive timestamps differ by samples_per_capture / fs).
        ts_path = os.path.join(rx_dir, "ts.f8")
        if os.path.exists(ts_path):
            ts = np.fromfile(ts_path, dtype="<f8")
            if len(ts) >= 2:
                dt = float(np.median(np.diff(ts)))
                if dt > 0:
                    fs = float(spc) / dt
                    fs_src = "ts.f8 cadence"

    # 3. Total session duration (capture_duration is whole-session, not
    # per-capture, in real recordings; `duration` is the same idea).
    cap_dur = params.get("capture_duration") or params.get("duration")
    if fs is None and spc and captures and cap_dur:
        fs = float(captures) * float(spc) / float(cap_dur)
        fs_src = "captures*samples_per_capture/capture_duration"

    # 4. Bandwidth (may be a filter width below the true sample rate).
    if fs is None and params.get("bandwidth"):
        fs = float(params["bandwidth"])
        fs_src = "parameters.bandwidth (may be < sample rate)"

    if fs is None:
        raise SystemExit(
            "Could not determine the sample rate. Provide --sample-rate "
            "(no usable ts.f8 cadence, capture_duration, or bandwidth)."
        )

    fc = center_freq
    if fc is None:
        fc = params.get("center_frequency", 0.0)
    print(f"sample rate: {float(fs) / 1e6:.4f} MHz (from {fs_src})")
    return float(fs), float(fc)


def resolve_bandwidth(
    rx_dir: str, override: float | None, full_band: bool, fs: float
) -> float | None:
    """Determine the occupied-signal bandwidth (Hz) to confine the axis to.

    The capture spans the full sample rate (the whole Nyquist width), but the
    signal of interest usually occupies a narrower band; outside it is just the
    receiver's out-of-band noise floor, far quieter than the in-band content.

    Priority:
      1. ``--full-band`` -> ``None`` (show the entire span, no cropping).
      2. ``--bandwidth`` override.
      3. ``parameters.bandwidth`` from ``meta.yaml`` (the configured filter
         width -- the occupied band, even though it is unreliable as a sample
         *rate*, see ``load_axis_meta``).

    Returns ``None`` (no crop) when nothing usable is found, the value is
    non-positive, or it is >= ``fs`` (cropping to >= the full span is a no-op).
    """
    if full_band:
        return None

    bw = override
    bw_src = "--bandwidth"
    if bw is None:
        meta_path = os.path.join(rx_dir, "meta.yaml")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = yaml.safe_load(f) or {}
            params = meta.get("parameters", {}) if isinstance(meta, dict) else {}
            mbw = params.get("bandwidth")
            if mbw:
                bw = float(mbw)
                bw_src = "parameters.bandwidth"

    if bw is None or bw <= 0:
        return None
    if bw >= fs:
        print(
            f"bandwidth {bw / 1e6:.4f} MHz >= sample rate "
            f"{fs / 1e6:.4f} MHz; showing full span."
        )
        return None
    print(
        f"bandwidth: {bw / 1e6:.4f} MHz (from {bw_src}); "
        f"confining frequency axis to center +/- {bw / 2e6:.4f} MHz."
    )
    return bw


def find_schemes(rx_dir: str, explicit: str | None) -> str | None:
    """Locate a transmitter ``schemes.yaml`` to overlay on the time axis.

    Precedence:
      1. an explicit ``--schemes`` path (used verbatim, even if missing);
      2. ``<rx>/schemes.yaml`` -- a manual copy into the rx dir;
      3. a sibling ``tx*/schemes.yaml`` in the recording dir -- where the
         transmitter actually writes it, so no copy is needed.
    Returns ``None`` if nothing is found.
    """
    if explicit:
        return explicit
    local = os.path.join(rx_dir, "schemes.yaml")
    if os.path.exists(local):
        return local
    recording_dir = os.path.dirname(rx_dir.rstrip("/\\"))
    if recording_dir:
        for tx in sorted(glob.glob(os.path.join(recording_dir, "tx*"))):
            cand = os.path.join(tx, "schemes.yaml")
            if os.path.exists(cand):
                return cand
    return None


def load_schemes(path: str | None) -> list[dict]:
    """Load a transmitter ``schemes.yaml`` for the time-axis overlay.

    Each entry has a ``modulation`` name and epoch ``start`` / ``stop`` times.
    Returns ``[]`` if the file is missing or unreadable (the overlay is
    optional).
    """
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or []
    except (OSError, yaml.YAMLError):
        return []
    schemes = []
    for s in data if isinstance(data, list) else []:
        try:
            schemes.append(
                {
                    "modulation": str(s.get("modulation", "?")),
                    "run": s.get("run"),
                    "start": float(s["start"]),
                    "stop": float(s["stop"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return schemes


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------


def sort_chunks_in_place(rx: Receiver) -> None:
    """Order ``rx.metadata.chunks`` by the numeric index in each *filename*.

    ``iq_sdk`` keys its chunk sort on the first digit-run of the *full path*, so
    any digit in the recording path (e.g. ``.../2.45GHz/rx0``) makes every
    chunk's key identical and the order collapses to filesystem (``glob``)
    order. ``read_frames`` indexes ``meta.chunks`` positionally and assumes
    position == chunk number, so a wrong order silently shuffles the waterfall
    in chunk-sized time blocks -- or crashes when the short final chunk lands
    mid-list. The list is shared with ``read_frames``, so sorting it in place
    fixes both call sites.
    """

    def chunk_index(path: str) -> int:
        # Match the digits immediately after the ``iq`` prefix, e.g. ``iq03.c8``
        # -> 3. Anchoring to ``iq`` avoids picking up the ``8`` in the ``.c8``
        # extension or digits elsewhere in the name.
        m = re.search(r"iq(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0

    rx.metadata.chunks.sort(key=chunk_index)


def read_frames(rx: Receiver, f_start: int, n_frames: int, nfft: int) -> np.ndarray:
    """Read ``n_frames`` contiguous frames as one ``[n_frames, nfft]`` array.

    Same chunk-walk as ``iq_sdk.Receiver`` (skips trailing page-padding,
    crosses chunk boundaries gaplessly) but in one pass, so batched FFTs
    aren't bottlenecked by per-frame overhead.
    """
    meta = rx.metadata
    count = n_frames * nfft
    out = np.empty(count, dtype=np.complex64)
    written = 0
    pos = f_start * nfft
    while written < count:
        chunk_idx = pos // meta.samples_per_chunk
        offset = pos % meta.samples_per_chunk
        chunk_samples = min(
            meta.samples_per_chunk,
            meta.total_samples - chunk_idx * meta.samples_per_chunk,
        )
        mmap = np.memmap(
            meta.chunks[chunk_idx],
            dtype=np.complex64,
            mode="r",
            shape=(chunk_samples,),
        )
        to_read = min(count - written, chunk_samples - offset)
        out[written : written + to_read] = mmap[offset : offset + to_read]
        written += to_read
        pos += to_read
    return out.reshape(n_frames, nfft)


def build_waterfall(
    rx: Receiver,
    fs: float,
    fc: float,
    nfft: int,
    window_name: str,
    max_rows: int,
    start_frame: int,
    end_frame: int,
    average: bool,
    block_frames: int = 4096,
):
    """Compute the waterfall image and its axes.

    FFTs are batched (one ``fft(..., axis=1)`` per block, not per frame).
    ``--average`` streams each row's span in ``block_frames``-sized sub-blocks
    to bound peak memory.

    Returns (spec[rows, nfft] dB float32, freqs_hz[nfft], times_s[rows],
    samples_read).
    """
    win = WINDOWS[window_name](nfft).astype(np.float64)
    win_norm = float(np.sum(win)) ** 2  # tone peak -> ~0 dB regardless of nfft
    eps = 1e-20

    frames_avail = end_frame - start_frame
    if frames_avail <= 0:
        raise SystemExit(
            f"Selected range holds no full {nfft}-sample frame. Lower --nfft "
            f"or widen --duration-sec / --num-frames."
        )

    rows = min(frames_avail, max_rows)
    stride = frames_avail / rows  # float, so rows span the extent evenly

    # Per-interval (== per-frame) epoch timestamps, relative to recording start.
    ts = rx.metadata.timestamps
    t0 = float(ts[0]) if len(ts) else 0.0

    spec = np.empty((rows, nfft), dtype=np.float32)
    times_s = np.empty(rows, dtype=np.float64)

    if average:
        # Sum |FFT|^2 over every frame in each row's span, streamed in
        # sub-blocks to bound peak memory.
        for r in range(rows):
            f0 = start_frame + int(r * stride)
            f1 = max(start_frame + int((r + 1) * stride), f0 + 1)
            times_s[r] = float(ts[f0]) - t0 if len(ts) else f0 * nfft / fs

            acc = np.zeros(nfft, dtype=np.float64)
            nfr = f1 - f0
            read = 0
            while read < nfr:
                b = min(block_frames, nfr - read)
                block = read_frames(rx, f0 + read, b, nfft)
                acc += (np.abs(np.fft.fft(block * win, axis=1)) ** 2).sum(axis=0)
                read += b
            power = acc / nfr
            spec[r] = 10.0 * np.log10(np.fft.fftshift(power) / win_norm + eps)
    else:
        # One strided frame per row; gather all (<= max_rows), one batched FFT.
        frames = np.empty((rows, nfft), dtype=np.complex64)
        for r in range(rows):
            f0 = start_frame + int(r * stride)
            times_s[r] = float(ts[f0]) - t0 if len(ts) else f0 * nfft / fs
            frames[r] = read_frames(rx, f0, 1, nfft)[0]
        power = np.abs(np.fft.fft(frames * win, axis=1)) ** 2
        spec = (
            10.0 * np.log10(np.fft.fftshift(power, axes=1) / win_norm + eps)
        ).astype(np.float32)

    bins = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    freqs_hz = fc + bins
    samples_read = (frames_avail if average else rows) * nfft
    return spec, freqs_hz, times_s, samples_read


def crop_to_band(
    spec: np.ndarray, freqs_hz: np.ndarray, fc: float, bandwidth: float | None
):
    """Crop the spectrogram columns and freq axis to ``fc +/- bandwidth/2``.

    Restricting to the occupied band before the color scale is auto-fit is the
    point of the crop: the default ``vmin``/``vmax`` are percentiles of the
    whole ``spec``, so the quiet out-of-band columns otherwise drag ``vmin``
    down toward the out-of-band floor and waste most of the colormap on empty
    spectrum, washing out subtle in-band structure. After cropping, the range
    spans only the in-band noise floor to the signal peaks, so faint features
    become visible.

    No-op (returns the inputs unchanged) when ``bandwidth`` is falsy or the
    band would keep fewer than 2 bins or every bin.
    """
    if not bandwidth or bandwidth <= 0:
        return spec, freqs_hz
    lo, hi = fc - bandwidth / 2.0, fc + bandwidth / 2.0
    keep = (freqs_hz >= lo) & (freqs_hz <= hi)
    n_keep = int(keep.sum())
    if n_keep < 2 or n_keep == len(freqs_hz):
        return spec, freqs_hz
    return spec[:, keep], freqs_hz[keep]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _freq_unit(max_abs_hz: float) -> tuple[float, str]:
    for scale, label in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if max_abs_hz >= scale:
            return scale, label
    return 1.0, "Hz"


def _time_unit(max_s: float) -> tuple[float, str]:
    if max_s >= 1.0:
        return 1.0, "s"
    if max_s >= 1e-3:
        return 1e-3, "ms"
    return 1e-6, "µs"


def plot_waterfall(
    spec,
    freqs_hz,
    times_s,
    fs,
    fc,
    abs_start_epoch,
    args,
    samples_read,
    total_samples,
    schemes=(),
    rec_t0_epoch=None,
) -> None:
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fscale, funit = _freq_unit(np.max(np.abs(freqs_hz)))
    f = freqs_hz / fscale
    df = (freqs_hz[1] - freqs_hz[0]) if len(freqs_hz) > 1 else 0.0
    f_lo, f_hi = f[0], (freqs_hz[-1] + df) / fscale

    tspan = float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0
    tscale, tunit = _time_unit(max(times_s[-1], abs(tspan), 1e-9))
    t = times_s / tscale

    vmin = args.vmin if args.vmin is not None else float(np.percentile(spec, 5))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(spec, 99.5))

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        spec,
        aspect="auto",
        origin="upper",
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[f_lo, f_hi, t[-1], t[0]],
        interpolation="nearest",
    )
    ax.set_xlabel(f"Frequency ({funit})")
    ax.set_ylabel(f"Time ({tunit})")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Power (dB, rel. full-scale)")

    rows = spec.shape[0]
    title = (
        f"Waterfall  |  center {fc / 1e6:.4g} MHz  "
        f"fs {fs / 1e6:.4g} MHz  nfft {args.nfft}  {rows} rows"
    )
    if abs_start_epoch is not None and np.isfinite(abs_start_epoch):
        title += "\nstart " + time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(abs_start_epoch)
        )

    # Scheme overlay: mark transmitted modulation bursts on the time axis.
    if schemes and rec_t0_epoch is not None and np.isfinite(rec_t0_epoch):
        colors = {
            "BPSK": "#ff5252",
            "QPSK": "#448aff",
            "MPSK_8": "#ffb300",
            "MQAM_16": "#b388ff",
        }
        t_first, t_last = float(times_s[0]), float(times_s[-1])
        drawn = 0
        for s in schemes:
            a = s["start"] - rec_t0_epoch
            b = s["stop"] - rec_t0_epoch
            if b < t_first or a > t_last:  # outside the rendered window
                continue
            col = colors.get(s["modulation"], "#e0e0e0")
            # Clip the shaded band to the rendered data so it never extends into
            # a not-yet-copied chunk; draw a boundary line only when the burst
            # actually starts / ends inside the window.
            a_clip, b_clip = max(a, t_first), min(b, t_last)
            ax.axhspan(
                a_clip / tscale, b_clip / tscale, color=col, alpha=0.10, zorder=2
            )
            if a >= t_first:
                ax.axhline(a / tscale, color=col, lw=0.8, zorder=3)
            if b <= t_last:
                ax.axhline(b / tscale, color=col, lw=0.8, ls="--", zorder=3)
            label = s["modulation"]
            if s.get("run") is not None:
                label += f" r{s['run']}"
            ax.text(
                f_lo + 0.012 * (f_hi - f_lo),
                a_clip / tscale,
                label,
                color="white",
                va="top",
                ha="left",
                fontsize=8,
                fontweight="bold",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.85),
            )
            drawn += 1
        if drawn:
            title += f"  |  {drawn} scheme(s) overlaid"

    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    fig.savefig(args.output, dpi=args.dpi)
    print(
        f"Wrote {args.output}  ({rows} rows x {args.nfft} bins, "
        f"read {samples_read:,} of {total_samples:,} samples = "
        f"{100.0 * samples_read / max(total_samples, 1):.2f}%)"
    )
    if args.show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a waterfall/spectrogram from an iq-sdk recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("path", help="rx* directory, recording dir, or a .c8 file.")
    p.add_argument("-o", "--output", help="Output PNG (default: <name>_waterfall.png).")
    p.add_argument("--nfft", type=int, default=1024, help="FFT size (frequency bins).")
    p.add_argument(
        "--window", choices=sorted(WINDOWS), default="hann", help="FFT window."
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Max time rows; striding keeps huge files fast.",
    )
    p.add_argument("--start-sec", type=float, help="Start offset in seconds.")
    p.add_argument("--start-sample", type=int, help="Start offset in samples.")
    p.add_argument("--duration-sec", type=float, help="Extent to cover, in seconds.")
    p.add_argument("--num-frames", type=int, help="Extent to cover, in FFT frames.")
    p.add_argument("--sample-rate", type=float, help="Override sample rate (Hz).")
    p.add_argument("--center-freq", type=float, help="Override center frequency (Hz).")
    p.add_argument(
        "--bandwidth",
        type=float,
        help="Confine the frequency axis (and color normalization) to "
        "center +/- bandwidth/2 (Hz). Default: parameters.bandwidth from "
        "meta.yaml, if present and below the sample rate.",
    )
    p.add_argument(
        "--full-band",
        action="store_true",
        help="Show the full sample-rate span; disable bandwidth cropping.",
    )
    p.add_argument(
        "--average",
        action="store_true",
        help="Average power across each stride window (reads all "
        "data in the extent; slower, smoother).",
    )
    p.add_argument(
        "--block-frames",
        type=int,
        default=4096,
        help="Frames per batched FFT read under --average; caps peak memory "
        "(~block_frames*nfft*16 B). Larger = fewer, bigger reads.",
    )
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    p.add_argument("--vmin", type=float, help="Color floor in dB (default 5th pct).")
    p.add_argument("--vmax", type=float, help="Color ceiling in dB (default 99.5 pct).")
    p.add_argument("--dpi", type=int, default=120, help="Output PNG DPI.")
    p.add_argument("--show", action="store_true", help="Display the figure too.")
    p.add_argument(
        "--schemes",
        help="Path to a tx schemes.yaml to overlay on the time axis "
        "(default: auto-detect schemes.yaml in the rx dir, then in a "
        "sibling tx* dir).",
    )
    p.add_argument(
        "--no-schemes",
        action="store_true",
        help="Disable the schemes.yaml time-axis overlay.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.nfft <= 0:
        raise SystemExit("--nfft must be positive.")
    if args.block_frames <= 0:
        raise SystemExit("--block-frames must be positive.")
    if args.start_sample is not None and args.start_sec is not None:
        raise SystemExit("Use only one of --start-sample / --start-sec.")
    if args.duration_sec is not None and args.num_frames is not None:
        raise SystemExit("Use only one of --duration-sec / --num-frames.")

    rx_dir = resolve_rx_dir(args.path)
    fs, fc = load_axis_meta(rx_dir, args.sample_rate, args.center_freq)
    rx = Receiver(rx_dir, interval=args.nfft)
    sort_chunks_in_place(rx)

    # Clamp to the chunks actually present on disk: meta.yaml declares the full
    # recording, but a partial / in-progress copy may have only the first few
    # iq*.c8 files. Reading past them would index a missing chunk.
    meta_total = rx.metadata.total_samples
    present_samples = min(
        meta_total, len(rx.metadata.chunks) * rx.metadata.samples_per_chunk
    )
    total_samples = present_samples
    total_frames = present_samples // args.nfft
    if total_frames <= 0:
        raise SystemExit(
            f"Recording has fewer than nfft={args.nfft} samples available."
        )
    if present_samples < meta_total:
        n_total = -(-meta_total // rx.metadata.samples_per_chunk)  # ceil div
        print(
            f"note: {len(rx.metadata.chunks)} of ~{n_total} chunks present on "
            f"disk; rendering the first "
            f"{100.0 * present_samples / meta_total:.1f}% "
            f"({present_samples:,} of {meta_total:,} samples)."
        )

    # Start offset -> starting frame.
    if args.start_sample is not None:
        start_sample = args.start_sample
    elif args.start_sec is not None:
        start_sample = int(round(args.start_sec * fs))
    else:
        start_sample = 0
    start_sample = max(0, min(start_sample, total_samples))
    start_frame = min(start_sample // args.nfft, total_frames)

    # Extent -> ending frame.
    if args.num_frames is not None:
        end_frame = start_frame + args.num_frames
    elif args.duration_sec is not None:
        end_frame = start_frame + int(round(args.duration_sec * fs)) // args.nfft
    else:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)

    if args.output is None:
        rx_dir_clean = rx_dir.rstrip("/\\")
        base = (
            f"{os.path.basename(os.path.dirname(rx_dir_clean))}"
            f"_{os.path.basename(rx_dir_clean)}".strip("_")
        )
        args.output = f"{base or 'iq'}_waterfall.png"

    spec, freqs_hz, times_s, samples_read = build_waterfall(
        rx,
        fs,
        fc,
        args.nfft,
        args.window,
        args.max_rows,
        start_frame,
        end_frame,
        args.average,
        args.block_frames,
    )

    # Confine to the occupied band so the auto-scaled color range isn't
    # dominated by the quiet out-of-band spectrum.
    bandwidth = resolve_bandwidth(rx_dir, args.bandwidth, args.full_band, fs)
    spec, freqs_hz = crop_to_band(spec, freqs_hz, fc, bandwidth)

    ts = rx.metadata.timestamps
    abs_start = float(ts[start_frame]) if start_frame < len(ts) else None
    rec_t0 = float(ts[0]) if len(ts) else None

    schemes = []
    if not args.no_schemes:
        schemes_path = find_schemes(rx_dir, args.schemes)
        schemes = load_schemes(schemes_path)
        if schemes:
            print(
                f"schemes: loaded {len(schemes)} burst(s) from "
                f"{schemes_path} for overlay."
            )

    plot_waterfall(
        spec,
        freqs_hz,
        times_s,
        fs,
        fc,
        abs_start,
        args,
        samples_read,
        total_samples,
        schemes,
        rec_t0,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
