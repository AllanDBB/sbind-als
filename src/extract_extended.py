#!/usr/bin/env python3
"""
Extended signal-processing measures for VOC-ALS (companion to extract_features.py).

A. Time-scale decomposition of laryngeal instability (sustained vowels)
   Modulation spectrum of the semitone F0 contour (and of the intensity contour) over the longest
   phonation, Welch PSD at 100 Hz frame rate:
     B1  0.25-1 Hz   slow wander (laryngeal postural control)
     B2  1-4 Hz      slow modulation
     B3  4-8 Hz      tremor
     B4  8-15 Hz     flutter (described in ALS)
   Absolute band power (st^2 or dB^2) and band share of 0.25-15 Hz power; spectral centroid of the
   modulation spectrum.

B. Micro vs macro F0 instability in syllable repetition (DDK)
   For each syllable nucleus: median F0 and within-nucleus F0 SD (semitones). Across syllables:
   SD of nucleus medians (macro), SD after removing the linear trend (macro, detrended), lag-1
   autocorrelation of the nucleus sequence (drift-like vs random), mean within-nucleus SD (micro).

C. Voice re-initiation dynamics (DDK)
   Per syllable: amplitude rise time (10-90 % of the valley-to-peak rise of the dB envelope),
   burst-to-voicing delay (VOT proxy: first voiced 2-ms pitch frame minus the high-frequency burst
   onset), and F0 onset deviation (semitones, first 30 ms of voicing vs nucleus median). Mean and
   variability (SD, CV) across syllables.

Usage:
  python extract_extended.py --data DATA_DIR --out results/extended.csv [--jobs 2] [--limit N]
                             [--debug-plots results/onset_plots --debug-n 12]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import (AudioFile, discover, VOWEL_TASKS, DDK_TASKS, TASK_SHORT, PITCH_FLOOR,  # noqa: E402
                              PITCH_CEILING, PITCH_STEP, VOICED_GAP, longest_voiced_run, detect_syllables,
                              energy_envelope_db, linear_slope, DDK_HOP)

BANDS = {"b1_wander": (0.25, 1.0), "b2_slow": (1.0, 4.0), "b3_tremor": (4.0, 8.0), "b4_flutter": (8.0, 15.0)}
OCTAVE_ST = 6.0  # |deviation| > 6 st from the file median = octave-level event (subharmonic / period doubling / tracking error)


def semitones_folded(f0_hz: np.ndarray, ref_hz: float):
    """Semitone deviation from ref, with octave-level deviations folded back by multiples of 12 st.
    Returns (folded st, octave-event mask)."""
    st = 12 * np.log2(f0_hz / ref_hz)
    octave = np.abs(st) > OCTAVE_ST
    folded = st - 12 * np.round(st / 12)
    return folded, octave
TOTAL = (0.25, 15.0)
FINE_STEP = 0.002  # s, pitch step for onset timing in DDK


# --------------------------------------------------------------------------------------
# A. modulation spectrum of a contour
# --------------------------------------------------------------------------------------

def modulation_bands(times: np.ndarray, contour: np.ndarray, step: float, prefix: str, to_semitones: bool) -> dict:
    """Band powers of the detrended, gap-interpolated contour. Zeros/NaNs are treated as missing."""
    from scipy.signal import welch, detrend
    out = {}
    c = contour.astype(float).copy()
    valid = np.isfinite(c) & (c > 0)
    if valid.sum() < int(2.0 / step):
        for b in BANDS:
            out[f"{prefix}_{b}_abs"] = out[f"{prefix}_{b}_rel"] = np.nan
        out[f"{prefix}_mod_centroid"] = out[f"{prefix}_total_abs"] = np.nan
        return out
    if to_semitones:
        folded, _ = semitones_folded(c[valid], float(np.median(c[valid])))
        c = np.interp(times, times[valid], folded)
    else:
        c = np.interp(times, times[valid], c[valid])
    c = detrend(c)
    fs = 1.0 / step
    nper = min(len(c), int(4.0 * fs))
    f, p = welch(c, fs=fs, nperseg=nper, noverlap=nper // 2)
    df = f[1] - f[0]
    tot_mask = (f >= TOTAL[0]) & (f <= TOTAL[1])
    tot = float(p[tot_mask].sum() * df)
    out[f"{prefix}_total_abs"] = tot
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        ab = float(p[m].sum() * df)
        out[f"{prefix}_{b}_abs"] = ab
        out[f"{prefix}_{b}_rel"] = ab / tot if tot > 0 else np.nan
    out[f"{prefix}_mod_centroid"] = float((f[tot_mask] * p[tot_mask]).sum() / p[tot_mask].sum()) if tot > 0 else np.nan
    return out


def vowel_extended(path: Path) -> dict:
    import parselmouth
    snd = parselmouth.Sound(str(path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    pitch = snd.to_pitch_cc(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    t = pitch.xs()
    run = longest_voiced_run(t, f0 > 0, PITCH_STEP, VOICED_GAP)
    feats: dict = {}
    if run is None:
        return feats
    t0, t1 = run
    m = (t >= t0) & (t < t1)
    feats["run_dur"] = t1 - t0
    v = f0[(f0 > 0)]
    if len(v) >= 5:
        folded, octv = semitones_folded(v, float(np.median(v)))
        feats["f0_oct_frac"] = float(octv.mean())                       # share of voiced frames at octave-level deviation
        feats["f0_oct_events_per_s"] = float(np.sum(np.diff(octv.astype(int)) != 0) / (len(v) * PITCH_STEP))
        feats["f0_sd_st_fold"] = float(folded.std(ddof=1))              # octave-folded F0 spread
        keep = ~octv
        feats["f0_sd_st_rob"] = float((12 * np.log2(v[keep] / np.median(v[keep]))).std(ddof=1)) if keep.sum() > 5 else np.nan
    feats.update(modulation_bands(t[m], f0[m], PITCH_STEP, "f0", True))
    inten = snd.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=PITCH_STEP)
    iv, it = inten.values[0], inten.xs()
    mi = (it >= t0) & (it < t1)
    feats.update(modulation_bands(it[mi], np.where(np.isfinite(iv[mi]), iv[mi], 0.0), PITCH_STEP, "int", False))
    return feats


# --------------------------------------------------------------------------------------
# B + C. DDK: micro/macro F0 and voice re-initiation
# --------------------------------------------------------------------------------------

def hf_envelope_db(y: np.ndarray, sr: int, lo=2000.0, hi=3900.0, frame=0.010, hop=FINE_STEP):
    """Band-limited (lo-hi Hz) RMS envelope in dB at a fine hop, for burst detection."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo, min(hi, 0.49 * sr)], btype="band", fs=sr, output="sos")
    yh = sosfiltfilt(sos, y)
    n = max(int(frame * sr), 8)
    h = max(int(hop * sr), 1)
    nfr = 1 + max(0, (len(yh) - n) // h)
    idx = np.arange(n)[None, :] + h * np.arange(nfr)[:, None]
    rms = np.sqrt(np.mean(yh[idx] ** 2, axis=1) + 1e-12)
    return (np.arange(nfr) * h + n / 2) / sr, 20 * np.log10(rms + 1e-9)


def ddk_extended(path: Path, debug_dir: Path | None = None) -> dict:
    import parselmouth
    from scipy.io import wavfile
    sr, y = wavfile.read(str(path))
    is_int = np.issubdtype(y.dtype, np.integer)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64) / (32768.0 if is_int else 1.0)
    feats: dict = {}

    pk_t, pk_db, (te, env) = detect_syllables(y, sr)
    n = len(pk_t)
    feats["n_syll"] = int(n)
    if n < 5:
        return feats

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch_cc(time_step=FINE_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    tp = pitch.xs()
    th, hf = hf_envelope_db(y, sr)

    nuc_med, nuc_sd, rise, vot, onset_dev, onset_slope = [], [], [], [], [], []
    bounds = np.concatenate([[te[0]], (pk_t[1:] + pk_t[:-1]) / 2, [te[-1]]])  # syllable regions between mid-points
    dbg = []
    for k, p in enumerate(pk_t):
        lo_b, hi_b = bounds[k], bounds[k + 1]
        # --- nucleus F0 (voiced frames within +-40 ms of the energy peak)
        w = (tp >= p - 0.04) & (tp <= p + 0.04) & (f0 > 0)
        if w.sum() >= 5:
            med = float(np.median(f0[w]))
            nuc_med.append(med)
            fold_w, _ = semitones_folded(f0[w], med)
            nuc_sd.append(float(fold_w.std(ddof=1)))
        else:
            med = np.nan
        # --- preceding valley of the dB envelope and amplitude rise time (10-90 %)
        seg = (te >= lo_b) & (te <= p)
        if seg.sum() >= 3:
            e_seg, t_seg = env[seg], te[seg]
            iv = int(np.argmin(e_seg))
            v, pkv = e_seg[iv], e_seg[-1]
            if pkv - v >= 6:
                e_after = e_seg[iv:]
                t_after = t_seg[iv:]
                t10 = t_after[np.argmax(e_after >= v + 0.1 * (pkv - v))]
                t90 = t_after[np.argmax(e_after >= v + 0.9 * (pkv - v))]
                rise.append(float(t90 - t10))
                t_valley = float(t_seg[iv])
            else:
                t_valley = None
        else:
            t_valley = None
        # --- burst onset (HF envelope) and voicing onset (first voiced fine pitch frame) after the valley
        if t_valley is not None:
            mh = (th >= t_valley) & (th <= p)
            if mh.sum() >= 3:
                h_seg, th_seg = hf[mh], th[mh]
                base = np.percentile(h_seg, 10)
                thr = base + 8.0
                ib = np.argmax(h_seg >= thr) if (h_seg >= thr).any() else None
                t_burst = float(th_seg[ib]) if ib is not None else None
            else:
                t_burst = None
            mv = (tp >= t_valley) & (tp <= p + 0.02) & (f0 > 0)
            t_voice = float(tp[mv][0]) if mv.any() else None
            if t_burst is not None and t_voice is not None and -0.02 <= t_voice - t_burst <= 0.25:
                vot.append(float(t_voice - t_burst))
            # --- F0 onset deviation and slope over the first 30 ms of voicing
            if t_voice is not None and np.isfinite(med):
                mo = (tp >= t_voice) & (tp <= t_voice + 0.03) & (f0 > 0)
                if mo.sum() >= 3:
                    st = 12 * np.log2(f0[mo] / med)
                    onset_dev.append(float(st.mean()))
                    onset_slope.append(linear_slope(tp[mo], st))
            if debug_dir is not None and k < 60:
                dbg.append((t_valley, t_burst, t_voice, p))

    nuc_med = np.array(nuc_med)
    if len(nuc_med) >= 5:
        st_raw = 12 * np.log2(nuc_med / np.median(nuc_med))
        st, octn = semitones_folded(nuc_med, float(np.median(nuc_med)))
        feats["macro_f0_sd_raw_st"] = float(st_raw.std(ddof=1))       # = syllable F0 SD of the main pipeline
        feats["macro_oct_frac"] = float(octn.mean())                   # share of nuclei at octave-level deviation
        feats["macro_f0_sd_st"] = float(st.std(ddof=1))                # octave-folded
        res = st - np.polyval(np.polyfit(np.arange(len(st)), st, 1), np.arange(len(st)))
        feats["macro_f0_sd_detr_st"] = float(res.std(ddof=1))
        feats["macro_f0_drift_st"] = float(abs(np.polyfit(np.arange(len(st)), st, 1)[0]) * (len(st) - 1))  # total linear drift
        d = st - st.mean()
        feats["macro_f0_lag1_ac"] = float((d[:-1] * d[1:]).sum() / (d ** 2).sum()) if (d ** 2).sum() > 0 else np.nan
        feats["macro_f0_abs_diff_st"] = float(np.abs(np.diff(st)).mean())  # syllable-to-syllable jump
        feats["micro_f0_sd_st"] = float(np.mean(nuc_sd))
        feats["n_nuclei"] = int(len(nuc_med))
    for name, arr in [("rise_time", rise), ("vot", vot), ("onset_dev", onset_dev), ("onset_slope", onset_slope)]:
        a = np.array(arr, dtype=float)
        if len(a) >= 5:
            feats[f"{name}_mean"] = float(a.mean()) if name != "onset_dev" else float(np.abs(a).mean())
            feats[f"{name}_sd"] = float(a.std(ddof=1))
            feats[f"{name}_cv"] = float(a.std(ddof=1) / abs(a.mean())) if abs(a.mean()) > 1e-9 and name in ("rise_time", "vot") else np.nan
            feats[f"{name}_n"] = int(len(a))
        else:
            feats[f"{name}_mean"] = feats[f"{name}_sd"] = feats[f"{name}_cv"] = np.nan
            feats[f"{name}_n"] = int(len(a))

    if debug_dir is not None and dbg:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
            t_end = min(te[-1], pk_t[min(7, n - 1)] + 0.3)
            m1 = te <= t_end
            ax[0].plot(te[m1], env[m1], lw=0.8, label="RMS env (dB)")
            m2 = th <= t_end
            ax[0].plot(th[m2], hf[m2], lw=0.6, alpha=0.7, label="HF 2-4 kHz env (dB)")
            m3 = (tp <= t_end) & (f0 > 0)
            ax[1].plot(tp[m3], f0[m3], ".", ms=2)
            for tv, tb, tvo, p in dbg:
                if p > t_end:
                    break
                if tb is not None:
                    ax[0].axvline(tb, color="C3", lw=0.7)
                if tvo is not None:
                    ax[0].axvline(tvo, color="C2", lw=0.7)
                    ax[1].axvline(tvo, color="C2", lw=0.7)
            ax[0].legend(fontsize=7, loc="lower right")
            ax[0].set_title(f"{path.name}: red = burst onset, green = voicing onset; VOT n={len(vot)} mean={np.mean(vot) * 1000 if vot else float('nan'):.0f} ms")
            ax[1].set_ylabel("F0 (Hz)")
            fig.tight_layout()
            debug_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(debug_dir / (path.stem + ".png"), dpi=110)
            plt.close(fig)
        except Exception:
            pass
    return feats


# --------------------------------------------------------------------------------------
# driver / aggregation
# --------------------------------------------------------------------------------------

def process(af: AudioFile, debug_dir: Path | None) -> dict:
    row = {"subject": af.subject, "task": af.task, "status": "ok", "error": ""}
    t0 = time.time()
    try:
        row.update(vowel_extended(af.path) if af.task in VOWEL_TASKS else ddk_extended(af.path, debug_dir))
    except Exception as e:
        row["status"], row["error"] = "error", f"{type(e).__name__}: {e}"
    row["seconds"] = round(time.time() - t0, 2)
    return row


def aggregate(pf: pd.DataFrame) -> pd.DataFrame:
    vcols = [c for c in pf.columns if c.startswith(("f0_", "int_")) or c == "run_dur"]
    dcols = [c for c in pf.columns if c.startswith(("macro_", "micro_", "rise_", "vot_", "onset_", "n_nuclei"))]
    rows = []
    for subj, g in pf.groupby("subject"):
        r = {"subject": subj}
        v = g[g.task.isin(VOWEL_TASKS)]
        d = g[g.task.isin(DDK_TASKS)]
        for task, gg in g.groupby("task"):
            s = gg.iloc[0]
            cols = vcols if task in VOWEL_TASKS else dcols
            for c in cols:
                r[f"{TASK_SHORT[task]}__{c}"] = s.get(c, np.nan)
        for c in vcols:
            r[f"V__{c}"] = float(np.nanmean(v[c].to_numpy(float))) if len(v) and np.isfinite(v[c].to_numpy(float)).any() else np.nan
        for c in dcols:
            r[f"DDK__{c}"] = float(np.nanmean(d[c].to_numpy(float))) if len(d) and np.isfinite(d[c].to_numpy(float)).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", choices=["vowels", "ddk"], default=None)
    ap.add_argument("--debug-plots", type=Path, default=None)
    ap.add_argument("--debug-n", type=int, default=12)
    args = ap.parse_args(argv)
    files = discover(args.data)
    if args.only == "vowels":
        files = [f for f in files if f.task in VOWEL_TASKS]
    elif args.only == "ddk":
        files = [f for f in files if f.task in DDK_TASKS]
    if args.limit:
        files = files[: args.limit]
    debug = set()
    if args.debug_plots is not None:
        dd = [f for f in files if f.task in DDK_TASKS]
        step = max(1, len(dd) // max(args.debug_n, 1))
        debug = {f.path for f in dd[::step][: args.debug_n]}
    print(f"{len(files)} files")
    t0 = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(process)(f, args.debug_plots if f.path in debug else None) for f in files)
    else:
        rows = [process(f, args.debug_plots if f.path in debug else None) for f in files]
    pf = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    pf.to_csv(f"{stem}_per_file.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; errors: {int((pf.status != 'ok').sum())}")
    wide = aggregate(pf[pf.status == "ok"])
    wide.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({wide.shape[0]} x {wide.shape[1]})")


if __name__ == "__main__":
    main()
