"""Generate a small synthetic IQ recording in the iq-sdk format.

Writes an ``rx0/`` directory containing ``iq*.c8`` chunks (``complex64``,
little-endian), ``ts.f8`` (``float64`` epoch seconds, one per capture) and a
``meta.yaml`` matching the receiver format documented in ``iq-sdk/docs/format.md``.

The signal is a sum of complex tones plus complex Gaussian noise. One tone is
gated on/off partway through the recording so that the rendered waterfall lets
you visually confirm *both* the frequency axis (tones at the expected offsets)
and the time axis (the gated tone appears, then disappears).

This lets the whole waterfall pipeline be tested end-to-end without the real
multi-GB capture.

Example:
    python make_test_data.py --out testdata/synthetic
    python waterfall.py testdata/synthetic/rx0 -o out.png
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import yaml

# 8 bytes / complex64 sample; pad chunks up to a 4096-byte page boundary.
BYTES_PER_SAMPLE = 8
PAGE_BYTES = 4096
SAMPLES_PER_PAGE = PAGE_BYTES // BYTES_PER_SAMPLE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic iq-sdk recording for testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out",
        default="testdata/synthetic",
        help="Output recording directory (an rx0/ subdir is created inside).",
    )
    p.add_argument("--captures", type=int, default=16, help="Total number of captures.")
    p.add_argument(
        "--samples-per-capture",
        type=int,
        default=65536,
        help="I/Q samples per capture.",
    )
    p.add_argument(
        "--captures-per-chunk",
        type=int,
        default=8,
        help="Captures per .c8 chunk (split exercises boundaries).",
    )
    p.add_argument(
        "--sample-rate", type=float, default=1_000_000.0, help="Sample rate in Hz."
    )
    p.add_argument(
        "--center-freq",
        type=float,
        default=100_000_000.0,
        help="Center frequency in Hz (for meta.yaml only).",
    )
    p.add_argument(
        "--tones",
        type=float,
        nargs="*",
        default=[-200_000.0, 100_000.0, 300_000.0],
        help="Baseband tone offsets in Hz. The LAST tone is gated on/off.",
    )
    p.add_argument(
        "--snr-db",
        type=float,
        default=20.0,
        help="Per-tone SNR over the noise floor, in dB.",
    )
    p.add_argument(
        "--no-pad",
        action="store_true",
        help="Do not page-align chunks with trailing zero padding.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducible noise."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    spc = args.samples_per_capture
    captures = args.captures
    cpc = args.captures_per_chunk
    fs = args.sample_rate
    total = captures * spc

    if cpc <= 0 or spc <= 0 or captures <= 0:
        raise SystemExit(
            "captures, samples-per-capture, captures-per-chunk must all be positive."
        )

    # Continuous time base across the whole recording (captures are gapless).
    t = np.arange(total, dtype=np.float64) / fs
    tone_amp = 1.0
    iq = np.zeros(total, dtype=np.complex128)

    for i, offset in enumerate(args.tones):
        phasor = tone_amp * np.exp(2j * np.pi * offset * t)
        if i == len(args.tones) - 1 and len(args.tones) > 1:
            # Gate the last tone ON for the middle 40%-80% of the recording.
            mask = np.zeros(total, dtype=np.float64)
            mask[int(0.4 * total) : int(0.8 * total)] = 1.0
            phasor *= mask
        iq += phasor

    # Complex Gaussian noise: total tone power / 10^(snr/10), split re/im.
    tone_power = tone_amp**2 * max(len(args.tones), 1)
    noise_power = tone_power / (10.0 ** (args.snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    iq += sigma * (rng.standard_normal(total) + 1j * rng.standard_normal(total))

    iq = iq.astype("<c8")  # complex64, little-endian

    rx_dir = os.path.join(args.out, "rx0")
    os.makedirs(rx_dir, exist_ok=True)

    # Write chunks of `cpc` captures each; pad the final bytes of each chunk to
    # a page boundary so the reader's padding-skip logic gets exercised.
    samples_per_chunk = cpc * spc
    n_chunks = (captures + cpc - 1) // cpc
    for c in range(n_chunks):
        start = c * samples_per_chunk
        block = iq[start : start + samples_per_chunk]
        if not args.no_pad:
            pad = (-len(block)) % SAMPLES_PER_PAGE
            if pad:
                block = np.concatenate([block, np.zeros(pad, dtype="<c8")])
        path = os.path.join(rx_dir, f"iq{c:02d}.c8")
        block.tofile(path)

    # Timestamps: one per capture, spaced by the capture duration.
    capture_duration = spc / fs
    t0 = time.time()
    ts = t0 + np.arange(captures, dtype=np.float64) * capture_duration
    ts.astype("<f8").tofile(os.path.join(rx_dir, "ts.f8"))

    meta = {
        "captures": int(captures),
        "captures_per_chunk": int(cpc),
        "samples_per_capture": int(spc),
        "sample_loss": False,
        "device_configurations": {
            "decimation": int(round(50_000_000.0 / fs)) if fs else 1,
            "device": "SYNTHETIC",
            "serial": -1,
        },
        "parameters": {
            "bandwidth": float(0.8 * fs),
            "capture_duration": float(capture_duration),
            "center_frequency": float(args.center_freq),
            "stop_if_sample_loss": False,
            "ref_level": -5.0
        },
    }
    with open(os.path.join(rx_dir, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    gated = args.tones[-1] if len(args.tones) > 1 else None
    print(
        f"Wrote {n_chunks} chunk(s), {captures} captures, {total} samples to {rx_dir}"
    )
    print(f"  sample_rate = {fs:.0f} Hz, center = {args.center_freq:.0f} Hz")
    print(f"  tones (baseband Hz): {args.tones}")
    if gated is not None:
        print(f"  gated tone (on for middle 40-80% of time): {gated:.0f} Hz")


if __name__ == "__main__":
    main()
