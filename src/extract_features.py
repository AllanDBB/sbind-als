#!/usr/bin/env python3
"""
Acoustic feature extraction for the VOC-ALS database (sustained vowels + DDK).

Three physiologically motivated feature "tracks":
  * Phonation (laryngeal)   : F0 statistics, jitter, shimmer, HNR, CPPS  (Praat via Parselmouth)
  * Articulation (oral)     : F1/F2, vowel-space metrics, DDK rate & regularity
  * Respiratory support     : phonation duration proxies, intensity decay, syllables per breath

Expected layout (as released on Synapse, syn53009474):
  <data>/phonationA/PZ001_phonationA.wav ... phonationU, rhythmPA, rhythmTA, rhythmKA
  Files are 8 kHz, 16-bit PCM wav. Folder names are not required: files are
  discovered recursively and the task is parsed from the file name.

Usage:
  python extract_features.py --data DATA_DIR [--xlsx CLINICAL.xlsx] --out results/features.csv
                             [--jobs 4] [--limit N] [--debug-plots results/ddk_plots --debug-n 12]

Outputs:
  <out>                       one row per subject, wide table (task__feature columns + aggregates)
  <out stem>_per_file.csv     one row per audio file (long-ish table, useful for QC)
  <out stem>_dictionary.csv   feature names, track, unit, description
  <out stem>_log.csv          per-file status / error messages
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

VOWEL_TASKS = ["phonationA", "phonationE", "phonationI", "phonationO", "phonationU"]
DDK_TASKS = ["rhythmPA", "rhythmTA", "rhythmKA"]
TASK_SHORT = {
    "phonationA": "A", "phonationE": "E", "phonationI": "I", "phonationO": "O", "phonationU": "U",
    "rhythmPA": "PA", "rhythmTA": "TA", "rhythmKA": "KA",
}
FILE_RE = re.compile(r"^(?P<id>.+?)_(?P<task>phonation[AEIOU]|rhythm(?:PA|TA|KA))\.wav$", re.IGNORECASE)

# Praat analysis settings (documented in the paper's Methods section)
PITCH_FLOOR = 60.0      # Hz  (ALS voices can be low / rough)
PITCH_CEILING = 500.0   # Hz
PITCH_STEP = 0.01       # s
EDGE_TRIM = 0.10        # s removed at both ends of the longest voiced run (onset/offset effects)
MIN_STEADY = 0.50       # s minimum steady segment to compute perturbation measures
VOICED_GAP = 0.15       # s gaps shorter than this are bridged when finding the longest voiced run
MAX_FORMANT = 4000.0    # Hz (recordings are 8 kHz -> Nyquist 4 kHz)
N_FORMANTS = 4

# DDK syllable detection (energy-envelope peak picking); tune with --debug-plots
DDK_FRAME = 0.020       # s RMS window
DDK_HOP = 0.005         # s
DDK_SMOOTH = 0.030      # s moving-average smoothing of the dB envelope
DDK_MIN_IOI = 0.090     # s minimum inter-onset interval (max ~11 syll/s)
DDK_REL_THRESH_DB = 20  # peaks must exceed (max - this)
DDK_PROMINENCE_DB = 4.0 # minimum peak prominence in dB
DDK_PAUSE_DB = 25       # frames below (max - this) count as pause/gap


@dataclass
class AudioFile:
    path: Path
    subject: str
    task: str


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

def discover(data_dir: Path) -> list[AudioFile]:
    files = []
    for p in sorted(data_dir.rglob("*.wav")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        task = m.group("task")
        # normalise capitalisation to the canonical task names
        canon = {t.lower(): t for t in VOWEL_TASKS + DDK_TASKS}[task.lower()]
        files.append(AudioFile(p, m.group("id").strip().upper(), canon))
    return files


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _nan(x):
    """Praat returns NaN / undefined as nan or raises; normalise to float nan."""
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def longest_voiced_run(times: np.ndarray, voiced: np.ndarray, step: float, max_gap: float):
    """Return (t_start, t_end) of the longest run of voiced frames, bridging gaps <= max_gap."""
    if voiced.sum() == 0:
        return None
    idx = np.where(voiced)[0]
    runs = []
    start = idx[0]
    prev = idx[0]
    max_gap_frames = int(round(max_gap / step))
    for i in idx[1:]:
        if i - prev > max_gap_frames + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    s, e = max(runs, key=lambda r: r[1] - r[0])
    return float(times[s]), float(times[e] + step)


def linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 3:
        return np.nan
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def tremor_index(times: np.ndarray, contour: np.ndarray, step: float, zero_is_missing: bool = True,
                 band=(3.0, 8.0), total=(0.5, 15.0)):
    """Low-frequency modulation of a contour (F0 in Hz or intensity in dB).
    Missing (zero) frames are linearly interpolated, the contour is converted to semitones when it
    looks like F0 (all > 20), detrended, and a Welch PSD is computed. Returns (band power / total
    power, peak frequency in the band). Needs >= 1 s of data."""
    from scipy.signal import welch, detrend
    c = contour.astype(float).copy()
    valid = np.isfinite(c) & ((c > 0) if zero_is_missing else np.ones_like(c, dtype=bool))
    if valid.sum() < int(1.0 / step):
        return np.nan, np.nan
    c = np.interp(times, times[valid], c[valid])
    if np.nanmin(c[valid]) > 20:  # F0 in Hz -> semitones
        c = 12 * np.log2(c / np.median(c))
    c = detrend(c)
    fs = 1.0 / step
    nper = min(len(c), int(4.0 * fs))
    f, p = welch(c, fs=fs, nperseg=nper)
    tot = p[(f >= total[0]) & (f <= total[1])].sum()
    inb = (f >= band[0]) & (f <= band[1])
    if tot <= 0 or not inb.any():
        return np.nan, np.nan
    return float(p[inb].sum() / tot), float(f[inb][np.argmax(p[inb])])


# --------------------------------------------------------------------------------------
# Sustained vowel features (Praat)
# --------------------------------------------------------------------------------------

def vowel_features(path: Path) -> dict:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    feats: dict = {"sr": snd.sampling_frequency, "dur_total": snd.duration}

    pitch = snd.to_pitch_cc(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    voiced = f0 > 0
    feats["voiced_dur"] = float(voiced.sum() * PITCH_STEP)
    feats["voiced_frac"] = float(voiced.mean()) if len(voiced) else np.nan

    # whole-file F0 variability (comparable to the dataset's reference stdevF0, which used the full file)
    f0_all = f0[voiced]
    if len(f0_all) >= 5:
        feats["f0_sd_all"] = float(f0_all.std(ddof=1))
        feats["f0_sd_st_all"] = float((12 * np.log2(f0_all / np.median(f0_all))).std(ddof=1))
    else:
        feats["f0_sd_all"] = feats["f0_sd_st_all"] = np.nan

    run = longest_voiced_run(times, voiced, PITCH_STEP, VOICED_GAP)
    if run is None:
        feats["mpt_proxy"] = 0.0
        return feats
    t0, t1 = run
    feats["mpt_proxy"] = t1 - t0  # longest continuous phonation (s)

    # voice breaks inside the longest (gap-bridged) phonation
    inrun = (times >= t0) & (times < t1)
    v_in = voiced[inrun]
    feats["unvoiced_frac_run"] = float(1 - v_in.mean()) if inrun.any() else np.nan
    feats["n_voice_breaks"] = int(np.sum(np.diff(v_in.astype(int)) == -1)) if inrun.sum() > 1 else 0

    # Steady segment: longest voiced run minus onset/offset edges
    s0, s1 = t0 + EDGE_TRIM, t1 - EDGE_TRIM
    if s1 - s0 < MIN_STEADY:       # too short -> fall back to the full run
        s0, s1 = t0, t1
    feats["steady_dur"] = s1 - s0
    seg = snd.extract_part(from_time=s0, to_time=s1, preserve_times=False)

    # --- F0 statistics on the steady segment
    p_seg = seg.to_pitch_cc(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0_raw = p_seg.selected_array["frequency"]
    f0s = f0_raw[f0_raw > 0]
    if len(f0s) >= 5:
        feats["f0_mean"] = float(f0s.mean())
        feats["f0_median"] = float(np.median(f0s))
        feats["f0_sd"] = float(f0s.std(ddof=1))
        feats["f0_cv"] = feats["f0_sd"] / feats["f0_mean"]
        # semitone SD is sex-independent
        st = 12 * np.log2(f0s / feats["f0_median"])
        feats["f0_sd_st"] = float(st.std(ddof=1))
        feats["f0_slope_st"] = linear_slope(p_seg.xs()[f0_raw > 0], st)  # semitones/s drift
        # F0 tremor: modulation of the (detrended, interpolated) semitone contour, 3-8 Hz vs 0.5-15 Hz
        feats["f0_tremor_idx"], feats["f0_tremor_hz"] = tremor_index(p_seg.xs(), f0_raw, PITCH_STEP)
    else:
        for k in ["f0_mean", "f0_median", "f0_sd", "f0_cv", "f0_sd_st", "f0_slope_st", "f0_tremor_idx", "f0_tremor_hz"]:
            feats[k] = np.nan

    # --- Perturbation measures (jitter / shimmer / HNR)
    try:
        pp = call(seg, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        feats["n_pulses"] = int(call(pp, "Get number of points"))
        feats["jitter_local"] = 100 * _nan(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        feats["jitter_rap"] = 100 * _nan(call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3))
        feats["jitter_ppq5"] = 100 * _nan(call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3))
        feats["shimmer_local"] = 100 * _nan(call([seg, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["shimmer_apq3"] = 100 * _nan(call([seg, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["shimmer_apq5"] = 100 * _nan(call([seg, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["shimmer_apq11"] = 100 * _nan(call([seg, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception as e:  # pragma: no cover
        feats["perturbation_error"] = str(e)

    try:
        harm = call(seg, "To Harmonicity (cc)", PITCH_STEP, PITCH_FLOOR, 0.1, 1.0)
        feats["hnr"] = _nan(call(harm, "Get mean", 0, 0))
    except Exception as e:  # pragma: no cover
        feats["hnr"] = np.nan

    # --- CPPS (smoothed cepstral peak prominence), Praat 6.x
    try:
        cep = call(seg, "To PowerCepstrogram", PITCH_FLOOR, 0.002, MAX_FORMANT, 50)
        feats["cpps"] = _nan(call(cep, "Get CPPS", "yes", 0.02, 0.0005, PITCH_FLOOR, 330, 0.05,
                                  "Parabolic", 0.001, 0.05, "Exponential decay", "Robust"))
    except Exception as e:  # pragma: no cover
        feats["cpps"] = np.nan

    # --- Intensity (loudness stability and decay = respiratory support proxy)
    try:
        inten = seg.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=PITCH_STEP)
        iv = inten.values[0]
        it = inten.xs()
        ok = np.isfinite(iv)
        feats["int_mean"] = float(iv[ok].mean())
        feats["int_sd"] = float(iv[ok].std(ddof=1))
        feats["int_slope"] = linear_slope(it[ok], iv[ok])           # dB/s
        feats["int_drop"] = float(iv[ok][-len(iv[ok]) // 5:].mean() - iv[ok][: len(iv[ok]) // 5].mean())  # last-first fifth, dB
        ivv = np.where(ok, iv, np.nan)
        feats["int_tremor_idx"], feats["int_tremor_hz"] = tremor_index(it, np.where(ok, iv, 0.0), PITCH_STEP, zero_is_missing=True)
    except Exception:
        for k in ["int_mean", "int_sd", "int_slope", "int_drop", "int_tremor_idx", "int_tremor_hz"]:
            feats[k] = np.nan

    # --- Spectral balance (strain / breathiness proxy): energy above 1 kHz relative to below (alpha ratio)
    try:
        spec = seg.to_spectrum()
        freqs = spec.xs()
        power = np.abs(spec.values[0]) ** 2 + np.abs(spec.values[1]) ** 2
        lo = power[(freqs >= 50) & (freqs < 1000)].sum()
        hi = power[(freqs >= 1000) & (freqs < 4000)].sum()
        feats["alpha_ratio_db"] = float(10 * np.log10((hi + 1e-12) / (lo + 1e-12)))
    except Exception:
        feats["alpha_ratio_db"] = np.nan

    # --- Formants (articulation): median F1/F2 over voiced frames of the steady segment
    try:
        fmt = call(seg, "To Formant (burg)", 0.0, N_FORMANTS, MAX_FORMANT, 0.025, 50)
        tt = p_seg.xs()
        vmask = p_seg.selected_array["frequency"] > 0
        f1 = np.array([_nan(call(fmt, "Get value at time", 1, t, "Hertz", "Linear")) for t in tt[vmask]])
        f2 = np.array([_nan(call(fmt, "Get value at time", 2, t, "Hertz", "Linear")) for t in tt[vmask]])
        feats["f1_med"] = float(np.nanmedian(f1)) if np.isfinite(f1).any() else np.nan
        feats["f2_med"] = float(np.nanmedian(f2)) if np.isfinite(f2).any() else np.nan
        feats["f1_iqr"] = float(np.nanpercentile(f1, 75) - np.nanpercentile(f1, 25)) if np.isfinite(f1).any() else np.nan
        feats["f2_iqr"] = float(np.nanpercentile(f2, 75) - np.nanpercentile(f2, 25)) if np.isfinite(f2).any() else np.nan
    except Exception:
        for k in ["f1_med", "f2_med", "f1_iqr", "f2_iqr"]:
            feats[k] = np.nan

    return feats


# --------------------------------------------------------------------------------------
# DDK (/pa/ /ta/ /ka/) features
# --------------------------------------------------------------------------------------

def energy_envelope_db(y: np.ndarray, sr: int):
    """Short-time RMS energy in dB with light smoothing. Returns (times, env_db)."""
    from scipy.ndimage import uniform_filter1d
    n = max(int(DDK_FRAME * sr), 8)
    h = max(int(DDK_HOP * sr), 1)
    y = y - np.mean(y)
    nfr = 1 + max(0, (len(y) - n) // h)
    idx = np.arange(n)[None, :] + h * np.arange(nfr)[:, None]
    frames = y[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    env = 20 * np.log10(rms + 1e-9)
    k = max(int(DDK_SMOOTH / DDK_HOP), 1)
    env = uniform_filter1d(env, size=k, mode="nearest")
    times = (np.arange(nfr) * h + n / 2) / sr
    return times, env


def detect_syllables(y: np.ndarray, sr: int):
    """Two-pass energy-peak picking. Pass 1 finds candidate syllable peaks; pass 2 re-detects with a
    minimum distance of 0.6 x the median inter-peak interval, which removes secondary peaks inside
    one syllable (e.g. burst + vowel). Returns peak times, peak levels (dB), and the envelope."""
    from scipy.signal import find_peaks
    t, env = energy_envelope_db(y, sr)
    floor = np.percentile(env, 5)
    top = np.percentile(env, 99)
    height = max(top - DDK_REL_THRESH_DB, floor + 6)
    dist = max(int(DDK_MIN_IOI / DDK_HOP), 1)
    peaks, _ = find_peaks(env, height=height, prominence=DDK_PROMINENCE_DB, distance=dist)
    if len(peaks) >= 3:
        ioi_med = float(np.median(np.diff(t[peaks])))
        dist2 = max(dist, int(0.6 * ioi_med / DDK_HOP))
        peaks, _ = find_peaks(env, height=height, prominence=DDK_PROMINENCE_DB, distance=dist2)
    return t[peaks], env[peaks], (t, env)


def ddk_features(path: Path, debug_dir: Path | None = None) -> dict:
    import parselmouth
    from scipy.io import wavfile

    sr, y = wavfile.read(str(path))
    is_int = np.issubdtype(y.dtype, np.integer)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    if is_int:
        y = y / 32768.0
    feats: dict = {"sr": sr, "dur_total": len(y) / sr}

    pk_t, pk_db, (t, env) = detect_syllables(y, sr)
    n = len(pk_t)
    feats["n_syll"] = int(n)
    if n >= 3:
        ioi_all = np.diff(pk_t)
        span = pk_t[-1] - pk_t[0]
        ioi_med = float(np.median(ioi_all))
        # intervals > 2.5 x median are pauses (breath / hesitation); rhythm measures exclude them
        is_pause = ioi_all > 2.5 * ioi_med
        ioi = ioi_all[~is_pause]
        feats["ddk_span"] = float(span)                       # s, first to last syllable
        feats["ddk_rate"] = float(1.0 / ioi_med)              # syll/s from the median interval
        feats["ddk_rate_gross"] = float((n - 1) / span)       # syll/s including pauses
        feats["ioi_mean"] = float(ioi.mean() * 1000)          # ms (pauses excluded)
        feats["ioi_sd"] = float(ioi.std(ddof=1) * 1000) if len(ioi) > 2 else np.nan
        feats["ioi_cv"] = float(ioi.std(ddof=1) / ioi.mean()) if len(ioi) > 2 else np.nan
        mad = float(np.median(np.abs(ioi - np.median(ioi))))
        feats["ioi_rcv"] = float(1.4826 * mad / np.median(ioi))  # robust CV (MAD-based)
        feats["ioi_slope"] = linear_slope(np.arange(len(ioi), dtype=float), ioi * 1000)  # ms per syllable (slowing)
        feats["n_pauses"] = int(is_pause.sum())
        feats["pause_time_frac"] = float(ioi_all[is_pause].sum() / span) if span > 0 else np.nan
        feats["syll_per_run"] = float(n / (is_pause.sum() + 1))  # syllables per uninterrupted run
        amp = 10 ** (pk_db / 20)
        feats["amp_cv"] = float(amp.std(ddof=1) / amp.mean())
        feats["amp_slope_db"] = linear_slope(pk_t, pk_db)     # dB/s across syllable peaks (decay)
        feats["amp_range_db"] = float(pk_db.max() - pk_db.min())
        # low-energy proportion between first and last syllable (gaps between syllables)
        inside = (t >= pk_t[0]) & (t <= pk_t[-1])
        feats["pause_ratio"] = float(np.mean(env[inside] < np.percentile(env, 99) - DDK_PAUSE_DB)) if inside.any() else np.nan
    else:
        for k in ["ddk_span", "ddk_rate", "ddk_rate_gross", "ioi_mean", "ioi_sd", "ioi_cv", "ioi_rcv", "ioi_slope",
                  "n_pauses", "pause_time_frac", "syll_per_run", "amp_cv", "amp_slope_db", "amp_range_db", "pause_ratio"]:
            feats[k] = np.nan

    # Voicing within the DDK sequence (poor stop closure / continuous voicing)
    try:
        snd = parselmouth.Sound(str(path))
        if snd.n_channels > 1:
            snd = snd.convert_to_mono()
        pitch = snd.to_pitch_cc(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        f0 = pitch.selected_array["frequency"]
        tt = pitch.xs()
        if n >= 3:
            m = (tt >= pk_t[0]) & (tt <= pk_t[-1])
            feats["voiced_frac"] = float((f0[m] > 0).mean()) if m.any() else np.nan
            # F0 at each syllable nucleus (median of voiced frames within +-40 ms of the energy peak)
            syl_f0 = []
            for pt in pk_t:
                w = (tt >= pt - 0.04) & (tt <= pt + 0.04) & (f0 > 0)
                if w.sum() >= 2:
                    syl_f0.append(np.median(f0[w]))
            syl_f0 = np.array(syl_f0)
            if len(syl_f0) >= 5:
                st = 12 * np.log2(syl_f0 / np.median(syl_f0))
                feats["syl_f0_sd_st"] = float(st.std(ddof=1))          # across-syllable pitch instability
                feats["syl_f0_slope_st"] = linear_slope(np.arange(len(st), dtype=float), st)  # drift per syllable
            else:
                feats["syl_f0_sd_st"] = feats["syl_f0_slope_st"] = np.nan
        else:
            feats["voiced_frac"] = float((f0 > 0).mean())
            feats["syl_f0_sd_st"] = feats["syl_f0_slope_st"] = np.nan
        v = f0[f0 > 0]
        feats["f0_mean"] = float(v.mean()) if len(v) >= 5 else np.nan
        feats["f0_sd"] = float(v.std(ddof=1)) if len(v) >= 5 else np.nan
        feats["f0_sd_st_all"] = float((12 * np.log2(v / np.median(v))).std(ddof=1)) if len(v) >= 5 else np.nan
    except Exception:
        for k in ["voiced_frac", "f0_mean", "f0_sd", "f0_sd_st_all", "syl_f0_sd_st", "syl_f0_slope_st"]:
            feats[k] = np.nan

    if debug_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 2.6))
            ax.plot(t, env, lw=0.8)
            ax.plot(pk_t, pk_db, "o", ms=4)
            ax.set_title(f"{path.name}  n={n}  rate={feats.get('ddk_rate', np.nan):.2f}/s  ioi_cv={feats.get('ioi_cv', np.nan):.2f}")
            ax.set_xlabel("time (s)"); ax.set_ylabel("dB")
            fig.tight_layout()
            debug_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(debug_dir / (path.stem + ".png"), dpi=110)
            plt.close(fig)
        except Exception:
            pass
    return feats


# --------------------------------------------------------------------------------------
# Per-file driver
# --------------------------------------------------------------------------------------

def process_file(af: AudioFile, debug_dir: Path | None = None) -> dict:
    row = {"subject": af.subject, "task": af.task, "file": str(af.path), "status": "ok", "error": ""}
    t0 = time.time()
    try:
        if af.task in VOWEL_TASKS:
            row.update(vowel_features(af.path))
        else:
            row.update(ddk_features(af.path, debug_dir))
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 2)
    return row


# --------------------------------------------------------------------------------------
# Aggregation to one row per subject
# --------------------------------------------------------------------------------------

VOWEL_FEATS = ["dur_total", "voiced_dur", "mpt_proxy", "steady_dur", "f0_mean", "f0_median", "f0_sd", "f0_cv",
               "f0_sd_st", "f0_sd_all", "f0_sd_st_all", "f0_slope_st", "f0_tremor_idx", "f0_tremor_hz",
               "unvoiced_frac_run", "n_voice_breaks",
               "jitter_local", "jitter_rap", "jitter_ppq5", "shimmer_local", "shimmer_apq3",
               "shimmer_apq5", "shimmer_apq11", "hnr", "cpps", "alpha_ratio_db",
               "int_mean", "int_sd", "int_slope", "int_drop", "int_tremor_idx", "int_tremor_hz",
               "f1_med", "f2_med", "f1_iqr", "f2_iqr"]
DDK_FEATS = ["dur_total", "n_syll", "ddk_span", "ddk_rate", "ddk_rate_gross", "ioi_mean", "ioi_sd", "ioi_cv", "ioi_rcv",
             "ioi_slope", "n_pauses", "pause_time_frac", "syll_per_run", "amp_cv", "amp_slope_db", "amp_range_db",
             "pause_ratio", "voiced_frac", "f0_mean", "f0_sd", "f0_sd_st_all", "syl_f0_sd_st", "syl_f0_slope_st"]


def aggregate(per_file: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for subj, g in per_file.groupby("subject"):
        r: dict = {"subject": subj}
        byt = {t: gg.iloc[0] for t, gg in g.groupby("task")}
        # per-task columns
        for task, s in byt.items():
            short = TASK_SHORT[task]
            feats = VOWEL_FEATS if task in VOWEL_TASKS else DDK_FEATS
            for f in feats:
                r[f"{short}__{f}"] = s.get(f, np.nan)
        # vowel means
        vrows = [byt[t] for t in VOWEL_TASKS if t in byt]
        r["n_vowel_files"] = len(vrows)
        for f in VOWEL_FEATS:
            vals = np.array([vr.get(f, np.nan) for vr in vrows], dtype=float)
            r[f"V__{f}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        r["V__mpt_proxy_max"] = float(np.nanmax([vr.get("mpt_proxy", np.nan) for vr in vrows])) if vrows else np.nan
        # DDK means
        drows = [byt[t] for t in DDK_TASKS if t in byt]
        r["n_ddk_files"] = len(drows)
        for f in DDK_FEATS:
            vals = np.array([dr.get(f, np.nan) for dr in drows], dtype=float)
            r[f"DDK__{f}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        # articulation: vowel space from /a/, /i/, /u/
        def F(task, k):
            return float(byt[task].get(k, np.nan)) if task in byt else np.nan
        f1a, f2a = F("phonationA", "f1_med"), F("phonationA", "f2_med")
        f1i, f2i = F("phonationI", "f1_med"), F("phonationI", "f2_med")
        f1u, f2u = F("phonationU", "f1_med"), F("phonationU", "f2_med")
        if np.all(np.isfinite([f1a, f2a, f1i, f2i, f1u, f2u])):
            r["ART__vsa"] = 0.5 * abs(f1a * (f2i - f2u) + f1i * (f2u - f2a) + f1u * (f2a - f2i))  # Hz^2
            r["ART__fcr"] = (f2u + f2a + f1i + f1u) / (f2i + f1a)      # formant centralisation ratio (Sapir 2010)
            r["ART__vai"] = 1.0 / r["ART__fcr"]                        # vowel articulation index
            r["ART__f2_ratio"] = f2i / f2u
        else:
            r["ART__vsa"] = r["ART__fcr"] = r["ART__vai"] = r["ART__f2_ratio"] = np.nan
        rows[subj] = r
    return pd.DataFrame(list(rows.values())).set_index("subject").reset_index()


FEATURE_DICTIONARY = {
    # phonation
    "f0_mean": ("phonation", "Hz", "mean fundamental frequency, steady segment"),
    "f0_median": ("phonation", "Hz", "median F0, steady segment"),
    "f0_sd": ("phonation", "Hz", "F0 standard deviation (pitch instability)"),
    "f0_cv": ("phonation", "ratio", "F0 coefficient of variation (sd/mean, sex-normalised)"),
    "f0_sd_st": ("phonation", "semitones", "F0 SD in semitones (sex-normalised)"),
    "jitter_local": ("phonation", "%", "cycle-to-cycle period perturbation (Praat local)"),
    "jitter_rap": ("phonation", "%", "relative average perturbation (3-period)"),
    "jitter_ppq5": ("phonation", "%", "5-point period perturbation quotient"),
    "shimmer_local": ("phonation", "%", "cycle-to-cycle amplitude perturbation (Praat local)"),
    "shimmer_apq3": ("phonation", "%", "3-point amplitude perturbation quotient"),
    "shimmer_apq5": ("phonation", "%", "5-point amplitude perturbation quotient"),
    "shimmer_apq11": ("phonation", "%", "11-point amplitude perturbation quotient"),
    "hnr": ("phonation", "dB", "harmonics-to-noise ratio (breathiness/noise)"),
    "cpps": ("phonation", "dB", "smoothed cepstral peak prominence (dysphonia severity)"),
    "int_sd": ("phonation", "dB", "intensity SD over steady segment (loudness instability)"),
    "f0_sd_all": ("phonation", "Hz", "F0 SD over all voiced frames of the recording (reference-compatible)"),
    "f0_sd_st_all": ("phonation", "semitones", "F0 SD over all voiced frames, semitones"),
    "f0_slope_st": ("phonation", "st/s", "F0 drift over the steady segment"),
    "f0_tremor_idx": ("phonation", "ratio", "share of F0 modulation power in 3-8 Hz (vocal tremor)"),
    "f0_tremor_hz": ("phonation", "Hz", "dominant F0 modulation frequency in 3-8 Hz"),
    "int_tremor_idx": ("phonation", "ratio", "share of intensity modulation power in 3-8 Hz"),
    "int_tremor_hz": ("phonation", "Hz", "dominant intensity modulation frequency in 3-8 Hz"),
    "unvoiced_frac_run": ("phonation", "ratio", "unvoiced fraction inside the longest phonation (voice breaks)"),
    "n_voice_breaks": ("phonation", "count", "number of voice breaks inside the longest phonation"),
    "alpha_ratio_db": ("phonation", "dB", "energy 1-4 kHz minus 50-1000 Hz (spectral balance; strain/breathiness)"),
    "syl_f0_sd_st": ("phonation", "semitones", "SD of F0 across syllable nuclei in DDK (laryngeal control)"),
    "syl_f0_slope_st": ("phonation", "st/syll", "F0 drift across syllables in DDK"),
    # articulation
    "f1_med": ("articulation", "Hz", "median first formant"),
    "f2_med": ("articulation", "Hz", "median second formant"),
    "f1_iqr": ("articulation", "Hz", "F1 interquartile range (formant instability)"),
    "f2_iqr": ("articulation", "Hz", "F2 interquartile range (formant instability)"),
    "ART__vsa": ("articulation", "Hz^2", "vowel space area from /a/,/i/,/u/ (tongue range)"),
    "ART__fcr": ("articulation", "ratio", "formant centralisation ratio (higher = more centralised)"),
    "ART__vai": ("articulation", "ratio", "vowel articulation index (1/FCR)"),
    "ART__f2_ratio": ("articulation", "ratio", "F2(i)/F2(u), tongue front-back range"),
    "ddk_rate": ("articulation", "syll/s", "diadochokinetic rate"),
    "ioi_mean": ("articulation", "ms", "mean inter-syllable interval"),
    "ioi_cv": ("articulation", "ratio", "inter-syllable interval CV (rhythm irregularity)"),
    "ioi_slope": ("articulation", "ms/syll", "change of interval along the sequence (slowing)"),
    "amp_cv": ("articulation", "ratio", "syllable peak-amplitude CV (irregularity)"),
    "voiced_frac": ("articulation", "ratio", "voiced fraction inside DDK (stop closure / hypernasality proxy)"),
    # respiratory support
    "mpt_proxy": ("respiration", "s", "longest continuous phonation in the recording (MPT proxy; task asked >=5 s)"),
    "voiced_dur": ("respiration", "s", "total voiced duration"),
    "int_slope": ("respiration", "dB/s", "intensity decline over the steady segment"),
    "int_drop": ("respiration", "dB", "intensity change last vs first fifth of segment"),
    "n_syll": ("respiration", "count", "syllables produced in one breath (DDK)"),
    "ddk_span": ("respiration", "s", "duration of the syllable train (one breath)"),
    "amp_slope_db": ("respiration", "dB/s", "amplitude decay across syllables"),
    "pause_ratio": ("respiration", "ratio", "proportion of low-energy frames inside the syllable train"),
}


def write_dictionary(path: Path, wide_cols: list[str]):
    recs = []
    for c in wide_cols:
        base = c.split("__", 1)[1] if "__" in c else c
        key = c if c in FEATURE_DICTIONARY else base
        track, unit, desc = FEATURE_DICTIONARY.get(key, ("other", "", ""))
        prefix = c.split("__", 1)[0] if "__" in c else ""
        scope = {"V": "mean over vowels", "DDK": "mean over /pa/,/ta/,/ka/", "ART": "from vowel formants"}.get(prefix, f"task {prefix}" if prefix else "")
        recs.append({"column": c, "track": track, "unit": unit, "scope": scope, "description": desc})
    pd.DataFrame(recs).to_csv(path, index=False)


# --------------------------------------------------------------------------------------
# Clinical merge
# --------------------------------------------------------------------------------------

def load_clinical(xlsx: Path) -> pd.DataFrame:
    """Load the VOC-ALS clinical sheet. The released file has a dummy first row ('Column-1', ...)
    and the real header on the second row; '-' marks not-applicable (healthy controls)."""
    raw = pd.read_excel(xlsx, sheet_name=0, header=None)
    header_row = 0
    for i in range(min(5, len(raw))):
        if str(raw.iloc[i, 0]).strip().upper() == "ID":
            header_row = i
            break
    df = pd.read_excel(xlsx, sheet_name=0, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace({"-": np.nan, "": np.nan})
    idcol = next((c for c in df.columns if c.strip().lower() in ("id", "subject", "subject_id", "participant")), None)
    if idcol is None:
        idcol = next((c for c in df.columns if "id" in c.lower()), df.columns[0])
    df = df.rename(columns={idcol: "subject"})
    df["subject"] = df["subject"].astype(str).str.strip().str.upper()
    return df


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="root folder containing the wav files (searched recursively)")
    ap.add_argument("--xlsx", type=Path, default=None, help="VOC-ALS clinical/acoustic .xlsx to merge (by ID)")
    ap.add_argument("--out", required=True, type=Path, help="output csv (one row per subject)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (joblib)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (smoke test)")
    ap.add_argument("--debug-plots", type=Path, default=None, help="folder for DDK syllable-detection plots")
    ap.add_argument("--debug-n", type=int, default=12, help="how many DDK files to plot")
    ap.add_argument("--only", choices=["vowels", "ddk"], default=None, help="process only one task family")
    ap.add_argument("--per-file-in", type=Path, default=None,
                    help="existing *_per_file.csv whose rows for the tasks NOT processed now are kept")
    args = ap.parse_args(argv)

    files = discover(args.data)
    if not files:
        sys.exit(f"No wav files matching <ID>_<task>.wav found under {args.data}")
    if args.only == "vowels":
        files = [f for f in files if f.task in VOWEL_TASKS]
    elif args.only == "ddk":
        files = [f for f in files if f.task in DDK_TASKS]
    if args.limit:
        files = files[: args.limit]
    n_subj = len({f.subject for f in files})
    print(f"Found {len(files)} files from {n_subj} subjects under {args.data}")

    # decide which DDK files get a debug plot
    debug_set = set()
    if args.debug_plots is not None:
        ddk_files = [f for f in files if f.task in DDK_TASKS]
        step = max(1, len(ddk_files) // max(args.debug_n, 1))
        debug_set = {f.path for f in ddk_files[::step][: args.debug_n]}

    t_start = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(process_file)(f, args.debug_plots if f.path in debug_set else None) for f in files)
    else:
        rows = []
        for i, f in enumerate(files, 1):
            rows.append(process_file(f, args.debug_plots if f.path in debug_set else None))
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files  ({time.time() - t_start:.0f}s)", flush=True)

    per_file = pd.DataFrame(rows)
    if args.per_file_in is not None and args.per_file_in.exists():
        old = pd.read_csv(args.per_file_in)
        done = set(per_file["task"])
        keep = old[~old["task"].isin(done)]
        per_file = pd.concat([keep, per_file], ignore_index=True, sort=False)
        print(f"Kept {len(keep)} rows from {args.per_file_in} for tasks not re-processed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    per_file.to_csv(f"{stem}_per_file.csv", index=False)
    per_file[["subject", "task", "file", "status", "error", "seconds"]].to_csv(f"{stem}_log.csv", index=False)
    n_err = int((per_file["status"] != "ok").sum())
    print(f"Extraction done in {time.time() - t_start:.0f}s; {n_err} file(s) with errors")

    wide = aggregate(per_file[per_file["status"] == "ok"])
    feature_cols = [c for c in wide.columns if c != "subject"]
    if args.xlsx is not None:
        clin = load_clinical(args.xlsx)
        before = len(wide)
        wide = wide.merge(clin, on="subject", how="left", suffixes=("", "_ref"))
        matched = int(wide[clin.columns[1]].notna().sum()) if len(clin.columns) > 1 else 0
        print(f"Merged clinical table: {matched}/{before} subjects matched by ID")
    wide.to_csv(args.out, index=False)
    write_dictionary(Path(f"{stem}_dictionary.csv"), feature_cols)
    print(f"Wrote {args.out}  ({wide.shape[0]} subjects x {wide.shape[1]} columns)")


if __name__ == "__main__":
    main()
