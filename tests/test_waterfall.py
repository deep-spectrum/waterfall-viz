"""Ground-truth sanity tests for waterfall.py.

Generates a tiny synthetic iq-sdk recording with *known* tones (one gated in
time), then probes the actual computed spectrogram rather than eyeballing the
PNG. Verifies:

  1. read_frames() byte-for-byte matches iq_sdk.Receiver.__getitem__ across
     chunk boundaries (the frame-walk is duplicated in waterfall.py).
  2. fs derivation from ts.f8 cadence is correct.
  3. Frequency axis: tones land at the right bins with the right SIGN
     (no I/Q swap / fftshift error).
  4. Power normalization: a full-scale tone reads ~0 dB.
  5. Time axis: the gated tone is present only in its on-window.
  6. average vs non-average agree on tone frequencies.

Run inside the container:
  docker exec -w /app/tests pvenv-waterfall-viz /app/.venv/bin/python test_waterfall.py
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iq_sdk import Receiver  # noqa: E402

import waterfall as wf  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    mark = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Synthetic recording generator (gapless, page-padded, chunk-split).
# ---------------------------------------------------------------------------

PAGE_SAMPLES = 4096 // 8  # complex64


def make_recording(
    out, *, captures, spc, cpc, fs, fc, tones, gated_idx, snr_db, seed=0, pad=True
):
    rng = np.random.default_rng(seed)
    total = captures * spc
    t = np.arange(total, dtype=np.float64) / fs
    iq = np.zeros(total, dtype=np.complex128)
    gate_lo, gate_hi = int(0.4 * total), int(0.8 * total)
    for i, off in enumerate(tones):
        ph = np.exp(2j * np.pi * off * t)
        if i == gated_idx:
            m = np.zeros(total)
            m[gate_lo:gate_hi] = 1.0
            ph = ph * m
        iq += ph
    tone_power = float(len(tones))
    noise_power = tone_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    iq += sigma * (rng.standard_normal(total) + 1j * rng.standard_normal(total))
    iq = iq.astype("<c8")

    rx = os.path.join(out, "rx0")
    os.makedirs(rx, exist_ok=True)
    spchunk = cpc * spc
    nchunks = (captures + cpc - 1) // cpc
    for c in range(nchunks):
        block = iq[c * spchunk : (c + 1) * spchunk]
        if pad:
            p = (-len(block)) % PAGE_SAMPLES
            if p:
                block = np.concatenate([block, np.zeros(p, dtype="<c8")])
        block.tofile(os.path.join(rx, f"iq{c:02d}.c8"))

    cap_dur = spc / fs
    t0 = 1_700_000_000.0
    ts = t0 + np.arange(captures, dtype=np.float64) * cap_dur
    ts.astype("<f8").tofile(os.path.join(rx, "ts.f8"))

    meta = {
        "captures": captures,
        "captures_per_chunk": cpc,
        "samples_per_capture": spc,
        "sample_loss": False,
        "parameters": {
            "bandwidth": float(0.8 * fs),
            "capture_duration": float(captures * cap_dur),  # TOTAL (real-world quirk)
            "center_frequency": float(fc),
        },
    }
    with open(os.path.join(rx, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return rx, iq, total, gate_lo, gate_hi


def main():
    tmp = tempfile.mkdtemp(prefix="wf_test_")
    try:
        fs, fc = 1_000_000.0, 100_000_000.0
        nfft = 1024
        tones = [-200_000.0, 100_000.0, 300_000.0]  # last is gated
        # cpc*spc not a multiple of nfft alignment is fine; choose boundary-crossing
        rx_dir, iq, total, glo, ghi = make_recording(
            tmp,
            captures=16,
            spc=4096,
            cpc=3,
            fs=fs,
            fc=fc,
            tones=tones,
            gated_idx=2,
            snr_db=30.0,
        )
        print(f"synthetic: {total} samples, {total // nfft} frames of nfft={nfft}")

        # --- 0. chunk ordering survives a digit-containing recording path ---
        print("\n[0] chunk ordering (regression: iq_sdk path-digit sort bug)")
        # Build the SAME recording under a path whose name contains digits, the
        # way real recordings do (e.g. ".../2.45GHz/...").
        digitdir = os.path.join(tmp, "run3_2.45GHz")
        rx_d, _, _, _, _ = make_recording(
            digitdir,
            captures=16,
            spc=4096,
            cpc=3,
            fs=fs,
            fc=fc,
            tones=tones,
            gated_idx=2,
            snr_db=30.0,
        )
        rx_bad = Receiver(rx_d, interval=nfft)
        order_before = [os.path.basename(c) for c in rx_bad.metadata.chunks]
        wf.sort_chunks_in_place(rx_bad)
        order_after = [os.path.basename(c) for c in rx_bad.metadata.chunks]
        expected = [f"iq0{i}.c8" for i in range(6)]
        check(
            "sort_chunks_in_place yields true numeric order",
            order_after == expected,
            f"{order_after}",
        )
        # Reading through the (now-sorted) receiver must equal the linear truth.
        nfr_d = rx_bad.metadata.total_samples // nfft
        lin = wf.read_frames(rx_bad, 0, nfr_d, nfft).reshape(-1)
        check(
            "data read in true recording order after fix",
            np.array_equal(lin, iq[: nfr_d * nfft]),
            f"glob order was {order_before}",
        )

        # Real-world layout: UNPADDED names (iq0..iq36) under a digit-laden
        # path, where even a naive string sort misorders (iq10 < iq2). Use a
        # per-chunk ramp so any misorder is caught byte-for-byte.
        real = os.path.join(tmp, "data0", "lab_switching-6-18-26", "noisy", "rx1")
        os.makedirs(real)
        spc_r, ncw = 512, 37
        ramp = np.arange(spc_r * ncw, dtype=np.float32).astype("<c8")
        for c in range(ncw):
            ramp[c * spc_r : (c + 1) * spc_r].tofile(os.path.join(real, f"iq{c}.c8"))
        (1.7e9 + np.arange(ncw) * (spc_r / fs)).astype("<f8").tofile(
            os.path.join(real, "ts.f8")
        )
        yaml.safe_dump(
            {
                "captures": ncw,
                "captures_per_chunk": 1,
                "samples_per_capture": spc_r,
                "parameters": {"center_frequency": fc},
            },
            open(os.path.join(real, "meta.yaml"), "w"),
        )
        rr = Receiver(real, interval=spc_r)
        wf.sort_chunks_in_place(rr)
        check(
            "unpadded iq0..iq36 sorted numerically",
            [os.path.basename(c) for c in rr.metadata.chunks]
            == [f"iq{i}.c8" for i in range(ncw)],
        )
        rd = wf.read_frames(rr, 0, (spc_r * ncw) // spc_r, spc_r).reshape(-1)
        check(
            "unpadded layout reads in true linear order",
            np.array_equal(rd, ramp.astype(np.complex64)),
        )

        # --- 1. read_frames matches the SDK reader across chunk boundaries ---
        print("\n[1] read_frames vs Receiver.__getitem__")
        rx = Receiver(rx_dir, interval=nfft)
        wf.sort_chunks_in_place(rx)
        nframes = rx.metadata.total_samples // nfft
        # batch read all frames
        batch = wf.read_frames(rx, 0, nframes, nfft)
        mismatch = 0
        for fi in (0, 1, nframes // 2, nframes - 2, nframes - 1):
            ref = rx[fi].iq[0]
            if not np.array_equal(batch[fi], ref):
                mismatch += 1
        check(
            "batched read_frames == per-frame SDK read",
            mismatch == 0,
            f"{mismatch} mismatched frames",
        )
        # cross-boundary contiguity: frame straddling samples_per_chunk
        spchunk = rx.metadata.samples_per_chunk
        bf = spchunk // nfft  # first frame fully past boundary
        straddle = wf.read_frames(rx, bf - 1, 3, nfft).reshape(-1)
        ref_lin = iq[(bf - 1) * nfft : (bf + 2) * nfft]
        check(
            "frames straddling chunk boundary are gapless",
            np.array_equal(straddle, ref_lin.astype(np.complex64)),
        )

        # --- 2. fs from ts.f8 cadence ---
        print("\n[2] sample-rate derivation")
        fs_got, fc_got = wf.load_axis_meta(rx_dir, None, None)
        # float64 epoch timestamps (~1.7e9) only carry ~6 fractional digits, so
        # median(diff(ts)) quantizes -> a few-ppm error. Fine for axis labels.
        check(
            "fs from ts.f8 cadence (within 0.01%)",
            abs(fs_got - fs) / fs < 1e-4,
            f"{fs_got} vs {fs}",
        )
        check("center freq from meta", fc_got == fc, f"{fc_got}")

        # --- 3 & 4. frequency placement + power normalization (averaged) ---
        print("\n[3/4] frequency placement, sign, and 0 dB normalization (average)")
        spec, freqs, times, _ = wf.build_waterfall(
            rx, fs, fc, nfft, "hann", 2000, 0, nframes, average=True
        )

        # column index for a given baseband offset
        def col_of(off):
            return int(np.argmin(np.abs(freqs - (fc + off))))

        mid_row = spec.shape[0] // 2  # inside gated window (40-80%)
        for off in tones:
            c = col_of(off)
            check(
                f"tone {off:+.0f} Hz present at correct bin/sign",
                spec[mid_row, c] > spec[mid_row].mean() + 10,
                f"col={c} freq={freqs[c] / 1e6:.4f}MHz val={spec[mid_row, c]:.1f}dB "
                f"mean={spec[mid_row].mean():.1f}dB",
            )
        # full-scale-ish tone level near 0 dB (3 unit tones summed; each ~0 dB).
        # check the peak dB is within a few dB of 0 (coherent-gain normalization).
        peak_db = float(spec[mid_row, col_of(-200_000.0)])
        check(
            "tone peak within ~3 dB of 0 dBFS",
            abs(peak_db) < 3.5,
            f"peak={peak_db:.2f} dB",
        )

        # negative vs positive sign distinguishable
        c_neg, c_pos = col_of(-200_000.0), col_of(+300_000.0)
        check(
            "negative offset left of center, positive right",
            freqs[c_neg] < fc < freqs[c_pos],
            f"{freqs[c_neg] / 1e6:.3f} < {fc / 1e6:.3f} < {freqs[c_pos] / 1e6:.3f}",
        )

        # --- 5. time gating: gated tone only in middle 40-80% ---
        print("\n[5] time gating of the third tone (+300 kHz)")
        cg = col_of(300_000.0)
        gated_strength = spec[:, cg]
        t_total = times[-1] - times[0] if len(times) > 1 else 1.0
        frac = (times - times[0]) / t_total
        on = (frac >= 0.42) & (frac <= 0.78)
        off = (frac < 0.35) | (frac > 0.85)
        on_med = float(np.median(gated_strength[on]))
        off_med = float(np.median(gated_strength[off]))
        check(
            "gated tone strong inside on-window",
            on_med > off_med + 15,
            f"on={on_med:.1f}dB off={off_med:.1f}dB",
        )
        # the non-gated tones must NOT be gated: present well above the noise
        # floor in every row, and equally strong inside vs outside the window.
        cs = col_of(100_000.0)
        s = spec[:, cs]
        floor_db = float(np.median(spec[mid_row]))  # row median ~ noise floor
        check(
            "steady tone present across all rows (and not time-gated)",
            float(np.median(s)) > floor_db + 20
            and abs(float(np.median(s[on])) - float(np.median(s[off]))) < 6,
            f"level={np.median(s):.1f}dB floor={floor_db:.1f}dB "
            f"on={np.median(s[on]):.1f} off={np.median(s[off]):.1f}",
        )

        # --- 6. average vs non-average agree on peak frequencies ---
        print("\n[6] average vs non-average peak agreement")
        spec_n, freqs_n, times_n, _ = wf.build_waterfall(
            rx, fs, fc, nfft, "hann", 2000, 0, nframes, average=False
        )
        mid_n = spec_n.shape[0] // 2
        for off in (-200_000.0, 100_000.0):
            ca = col_of(off)
            ok = spec_n[mid_n, ca] > spec_n[mid_n].mean() + 8
            check(
                f"non-average sees tone {off:+.0f} Hz",
                ok,
                f"val={spec_n[mid_n, ca]:.1f} mean={spec_n[mid_n].mean():.1f}",
            )

        # --- 7. DC / spectral symmetry sanity: noise floor flat-ish ---
        print("\n[7] noise floor sanity (no spurious huge DC spike from windowing)")
        # remove tone bins, check remaining floor std is modest
        floor = spec[mid_row].copy()
        for off in tones:
            c = col_of(off)
            floor[max(0, c - 2) : c + 3] = np.nan
        floor_vals = floor[~np.isnan(floor)]
        check(
            "noise floor spread reasonable (<25 dB IQR)",
            float(np.subtract(*np.percentile(floor_vals, [90, 10]))) < 25,
            f"IQR(10-90)={np.subtract(*np.percentile(floor_vals, [90, 10])):.1f} dB",
        )

        # --- 8. end-to-end via main(): numbered recording dir + rx subdir ---
        print("\n[8] end-to-end main() on a numbered recording dir w/ rx subdir")
        # Recording dir name is laden with digits (year, run, freq) and holds
        # an rx0 subdir of MULTIPLE chunks -- the exact shape of real data.
        rec_dir = os.path.join(tmp, "capture_2.4GHz_run5-6-18-26")
        rx_e, _, _, _, _ = make_recording(
            rec_dir,
            captures=18,
            spc=4096,
            cpc=3,
            fs=fs,
            fc=fc,
            tones=tones,
            gated_idx=2,
            snr_db=30.0,
        )
        n_chunks_e = len([f for f in os.listdir(rx_e) if f.startswith("iq")])
        check(
            "recording has multiple chunk files", n_chunks_e > 1, f"{n_chunks_e} chunks"
        )
        # Add a second receiver so resolve_rx_dir must choose among rx*.
        shutil.copytree(rx_e, os.path.join(rec_dir, "rx1"))

        # resolve_rx_dir: pointed at the *recording* dir, it must descend to an
        # rx* subdir (rx0, the first) despite the digit-laden parent path.
        resolved = wf.resolve_rx_dir(rec_dir)
        check(
            "resolve_rx_dir descends recording dir -> rx0 subdir",
            os.path.basename(resolved.rstrip("/\\")) == "rx0",
            resolved,
        )

        # Drive the real CLI entry point on the recording dir (not the rx dir).
        out_png = os.path.join(tmp, "e2e.png")
        buf = io.StringIO()
        ok_run = True
        try:
            with contextlib.redirect_stdout(buf):
                wf.main([rec_dir, "-o", out_png, "--average"])
        except SystemExit as e:  # argparse/guards only; success path returns
            ok_run = e.code in (None, 0)
        except Exception as e:  # noqa: BLE001
            ok_run = False
            buf.write(f"\nEXCEPTION: {e!r}")
        log = buf.getvalue()
        check("main() runs end-to-end without error", ok_run, log.strip()[-200:])
        check(
            "main() reports 100% of present samples read",
            "100.00%" in log,
            log.strip()[-200:],
        )
        check(
            "main() wrote a non-trivial PNG",
            os.path.exists(out_png) and os.path.getsize(out_png) > 5000,
            f"{os.path.getsize(out_png) if os.path.exists(out_png) else 0} bytes",
        )
        # auto-named output when -o omitted, still on the numbered path.
        out2 = os.path.join(tmp, "auto.png")
        with contextlib.redirect_stdout(io.StringIO()):
            wf.main([rx_e, "-o", out2, "--nfft", "512"])
        check(
            "main() accepts an rx dir directly (nfft override)",
            os.path.exists(out2) and os.path.getsize(out2) > 5000,
        )

        print(f"\n==== {PASS} passed, {FAIL} failed ====")
        sys.exit(1 if FAIL else 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
