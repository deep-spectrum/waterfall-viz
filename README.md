# waterfall-viz

A lightweight waterfall (spectrogram) viewer for IQ recordings in the
[iq-sdk format](../iq-sdk/docs/format.md). It's a quick visual sanity-check for
raw captures: tones, interference, drift, sample loss, dead bands, etc.

- **Reads through [iq-sdk](../iq-sdk):** uses `iq_sdk.Receiver`, so it tracks the
  recording format if it changes.
- **Few dependencies:** `iq-sdk` (+ its small deps), `numpy`, `pyyaml`, `matplotlib`.
- **Big-file friendly:** by default only the plotted frames are read (time
  *striding*), so a multi-GB recording renders about as fast as a small one.
  Partial / in-progress copies render the chunks present, and you can slice by
  time, sample, or chunk range.
- **Auto color normalization:** the frequency axis is confined to the occupied
  band (`parameters.bandwidth`) by default, so the color scale fits the in-band
  content instead of being dragged down by the quiet out-of-band noise floor.
- **Transmitter overlay:** if a `schemes.yaml` is present, labeled time bands for
  each transmitted modulation burst are drawn on the waterfall, so you can see
  whether the receiver caught each scheme. `schemes.yaml` is a transmitter
  artifact (it ships in the `tx*` dir) — it is auto-detected from a sibling
  `tx*` dir, or point at one with `--schemes`; see
  [Scheme overlay](#scheme-overlay-transmitter-schedule).

## Install

With [uv](https://docs.astral.sh/uv/) (recommended — installs the exact locked
versions from `uv.lock`):

```bash
cd waterfall-viz
uv sync
uv run python waterfall.py ...   # or activate .venv once
```

Or with pip, from the pinned `requirements.txt`:

```bash
cd waterfall-viz
python -m venv .venv
# Windows:  .venv\Scripts\pip install -r requirements.txt
# POSIX:    .venv/bin/pip install -r requirements.txt
```

`requirements.txt` is generated from `uv.lock` — change deps in `pyproject.toml`
and re-run `uv lock`, don't hand-edit it. iq-sdk is pinned to a commit; to hack
on the reader locally, `uv pip install -e ../iq-sdk` afterwards.

## Quick start (no real data needed)

Generate a tiny synthetic recording, then render it:

```bash
python make_test_data.py --out testdata/synthetic
python waterfall.py testdata/synthetic/rx0 -o out.png
```

Open `out.png`. With the defaults you should see three vertical tone lines at
`center - 200 kHz`, `center + 100 kHz`, and `center + 300 kHz`, where the
`+300 kHz` tone switches **on** for the middle of the recording and **off**
again — confirming both the frequency axis and the time axis are correct.

## Usage

```bash
python waterfall.py PATH [options]
```

`PATH` can be:
- an `rx*` directory (e.g. `rec/rx0`) — **preferred**;
- a recording directory containing `rx*/` subdirs (the first is used);
- a `.c8` chunk file — its receiver directory is used (the whole recording is
  read, since `iq_sdk.Receiver` needs `meta.yaml` and `ts.f8`).

Sample rate, center frequency and timestamps come from `meta.yaml` / `ts.f8`;
use `--sample-rate` / `--center-freq` to override the axis calibration.

Common options:

| Option | Default | Purpose |
| --- | --- | --- |
| `-o, --output` | `<name>_waterfall.png` | Output PNG path. |
| `--nfft` | `1024` | FFT size → number of frequency bins. |
| `--window` | `hann` | `hann`, `hamming`, `blackman`, or `rect`. |
| `--max-rows` | `2000` | Max time rows. Striding to this keeps huge files fast. |
| `--start-sec` / `--start-sample` / `--start-chunk` | `0` | Where to start reading (seconds, samples, or chunk index). |
| `--duration-sec` / `--num-frames` / `--end-chunk` | whole file | How much to cover (seconds, frames, or stop-before chunk index). |
| `--average` | off | Average power across each stride window (reads **all** data in the extent; slower, smoother) instead of skipping. |
| `--block-frames` | `4096` | Frames per batched FFT read under `--average`; caps peak memory. |
| `--sample-rate` / `--center-freq` | from `meta.yaml` | Override axis calibration (Hz). |
| `--bandwidth` / `--full-band` | `parameters.bandwidth` | Confine the frequency axis (and color normalization) to `center ± bandwidth/2`; `--full-band` shows the whole span. See [Bandwidth confinement](#bandwidth-confinement). |
| `--ref-level` | `parameters.ref_level` | Radio reference level (dB) shown as a debug label above the color scale. |
| `--cmap` | `viridis` | Matplotlib colormap. |
| `--vmin` / `--vmax` | 5th / 99.5th pct | Color scale floor/ceiling (dB). |
| `--schemes` / `--no-schemes` | auto-detect | Overlay TX modulation bursts from a `schemes.yaml` (auto-detected in the rx dir, then a sibling `tx*` dir); `--no-schemes` disables it. |
| `--show` | off | Also open an interactive window. |

The three start selectors are mutually exclusive, as are the three extent
selectors. `--end-chunk` is an *absolute* stop-before index (exclusive), so
`--start-chunk 10 --end-chunk 20` renders chunks `[10, 20)`.

### Large recordings

```bash
# Overview of the whole capture, fast (reads only ~max_rows*nfft samples):
python waterfall.py /data/rec/rx0 -o overview.png

# Zoom into a 50 ms slice 2 s in, finer resolution:
python waterfall.py /data/rec/rx0 --start-sec 2.0 --duration-sec 0.05 --nfft 4096

# Use every sample in a slice (no skipping) for a smoother image:
python waterfall.py /data/rec/rx0 --duration-sec 0.1 --average

# Render a specific chunk range (here chunks 10..19):
python waterfall.py /data/rec/rx0 --start-chunk 10 --end-chunk 20

# Just "from chunk 10 on" — renders a bounded default of 10 chunks:
python waterfall.py /data/rec/rx0 --start-chunk 10
```

The footer printed after rendering reports how many samples were actually read
vs. the recording total, so you can see the striding in action.

Chunk-range notes: `--end-chunk` past the last available chunk is clamped to the
last one (with a note) — handy for partial copies. `--start-chunk` past the last
chunk is an error. `--start-chunk` with no end selector renders a bounded
default of 10 chunks, so "start here" never reads an entire multi-GB recording.

### Bandwidth confinement

A capture spans the full sample rate (the whole Nyquist width), but the signal
of interest usually occupies a narrower band; outside it is just the receiver's
out-of-band noise floor, far quieter than the in-band content. Since the default
color scale is auto-fit to percentiles of the whole image, those quiet
out-of-band bins drag `vmin` down and waste most of the colormap on empty
spectrum — washing out subtle in-band structure.

By default the frequency axis is confined to `center ± bandwidth/2` using
`parameters.bandwidth` from `meta.yaml` (when present and below the sample rate),
which tightens the color range onto the in-band content. Override the width with
`--bandwidth HZ`, or disable cropping entirely with `--full-band`.

### Scheme overlay (transmitter schedule)

`schemes.yaml` lists the transmitted modulation bursts (BPSK, QPSK, …) with epoch
start/stop times. It is generated with the **transmitter**, so in a recording it
sits in the `tx*` directory — *not* the `rx*` directory this tool reads. No copy
is needed: it is auto-detected, and can be pointed at explicitly.

```bash
# 1. Auto-detect: <rx>/schemes.yaml first, then a sibling tx*/schemes.yaml.
python waterfall.py recording/rx0

# 2. Or point at it directly:
python waterfall.py recording/rx0 --schemes recording/tx0/schemes.yaml
```

The transmitter and receiver must be from the **same session** — bands are placed
using the shared epoch clock (`ts.f8`), so a mismatched pair will land in the
wrong place or off-screen. Use `--no-schemes` to turn the overlay off.

## Notes

- **Sample rate** is read from `ts.f8` / `meta.yaml`; the chosen value and its
  source are printed. Override with `--sample-rate` if it looks wrong.
- **Partial copies:** only the `iq*.c8` chunks present on disk are read; the run
  prints how many of N chunks are available, so in-progress copies still render.
- Power is in dB relative to full-scale; a full-amplitude tone sits near 0 dB.
- **Bandwidth confinement** crops the frequency axis to the occupied band by
  default for better color normalization; see
  [Bandwidth confinement](#bandwidth-confinement). Use `--full-band` to disable.
- **Reference level:** if `parameters.ref_level` is set (or `--ref-level` is
  given), it is shown as a debug label above the color scale. It is purely
  informational — it does not affect the DSP or normalization.
