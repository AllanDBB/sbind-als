#!/usr/bin/env python3
"""
Revision analyses answering reviewer concerns.

R1  Subsystem comparison with paired bootstrap of AUC differences (single measures and CV composites).
R2  Sensitivity: HC vs nested subsets of ALS-ND with an increasingly normal bulbar examination.
R3  Robustness of the syllable-F0 marker to the pitch tracker (Praat cc vs ac; floor 60 vs 75 Hz) and
    validation of octave-level nuclei with a subharmonic-to-harmonic ratio (SHR) computed on the spectrum.
R4  Recording and task-execution covariates: SNR, level, DDK rate, syllable count, mean F0; adjusted AUC.
R5  Repeatability of the marker: ICC(2,1) across /pa/, /ta/, /ka/ and split-half (odd/even syllables).
R6  Generic prosodic+spectral reference (F0 functionals in semitones, jitter, shimmer, HNR, intensity,
    voicing, duration + MFCC-39) under the same CV protocol, per task and pooled.

Usage: python analyze_revision.py --data data/raw --features results/features.csv --extended results/extended.csv
                                  --out results/analysis [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from analyze_subclinical import (auc_mwu, bh_fdr, residualise, resolve_columns, cv_auc, pooled_auc_ci, TRACKS,  # noqa: E402
                                 VOWEL_TASK_FEATS, DDK_TASK_FEATS)
from extract_features import (discover, VOWEL_TASKS, DDK_TASKS, TASK_SHORT, detect_syllables, PITCH_FLOOR,  # noqa: E402
                              PITCH_CEILING, PITCH_STEP)
from extract_extended import semitones_folded  # noqa: E402

KEY = "DDK__syl_f0_sd_st"


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

def auc_pair(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    return stats.mannwhitneyu(b, a).statistic / (len(a) * len(b))


def paired_boot_auc_diff(y, x1, x2, n_boot, rng):
    """Bootstrap subjects; AUC(x1) - AUC(x2) with the same resample (NaNs handled per measure)."""
    n = len(y)
    d = np.empty(n_boot)
    a1s = np.empty(n_boot); a2s = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            d[i] = np.nan; continue
        a1 = auc_pair(x1[idx][yy == 0], x1[idx][yy == 1])
        a2 = auc_pair(x2[idx][yy == 0], x2[idx][yy == 1])
        a1s[i], a2s[i], d[i] = a1, a2, a1 - a2
    d = d[np.isfinite(d)]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))


def icc21(M):
    """ICC(2,1), two-way random, absolute agreement, single measures. M: subjects x raters (no NaN)."""
    n, k = M.shape
    gm = M.mean()
    ms_r = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    ss_e = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


# ----------------------------------------------------------------------------------------------
# per-file audio computations (R3, R4, R5, R6)
# ----------------------------------------------------------------------------------------------

def nucleus_f0(f0, tt, pk_t):
    med = []
    for pt in pk_t:
        w = (tt >= pt - 0.04) & (tt <= pt + 0.04) & (f0 > 0)
        med.append(np.median(f0[w]) if w.sum() >= 2 else np.nan)
    return np.array(med)


def syl_sd(med):
    med = med[np.isfinite(med)]
    if len(med) < 5:
        return np.nan
    return float((12 * np.log2(med / np.median(med))).std(ddof=1))


def shr_db(y, sr, t_center, f_low, win=0.05):
    """Subharmonic-to-harmonic ratio around t_center for a candidate fundamental f_low: energy at odd
    multiples of f_low (subharmonics if the true F0 is 2*f_low) vs even multiples (harmonics of 2*f_low)."""
    n = int(win * sr)
    c = int(t_center * sr)
    seg = y[max(0, c - n // 2): c + n // 2]
    if len(seg) < n // 2:
        return np.nan
    seg = seg * np.hanning(len(seg))
    nfft = 8192
    spec = np.abs(np.fft.rfft(seg, nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    tol = max(0.08 * f_low, 8.0)
    def peak(f):
        m = (freqs >= f - tol) & (freqs <= f + tol)
        return spec[m].max() if m.any() else 0.0
    odd = sum(peak(k * f_low) ** 2 for k in range(1, 12, 2) if k * f_low < sr / 2 - tol)
    even = sum(peak(k * f_low) ** 2 for k in range(2, 12, 2) if k * f_low < sr / 2 - tol)
    return float(10 * np.log10((odd + 1e-12) / (even + 1e-12)))


REAPER_BIN = Path(__file__).resolve().parent.parent / "tools" / "reaper"


def reaper_f0(path, step=PITCH_STEP, fmin=PITCH_FLOOR, fmax=PITCH_CEILING):
    """F0 contour from Google's REAPER (independent epoch-based tracker). Returns (times, f0) with 0 for unvoiced,
    or None when the binary is not available."""
    import subprocess, tempfile, os
    if not REAPER_BIN.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "f0.txt")
        r = subprocess.run([str(REAPER_BIN), "-i", str(path), "-f", out, "-a", "-e", f"{step}", "-m", f"{fmin}", "-x", f"{fmax}"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        rows = []
        with open(out) as fh:
            header = True
            for line in fh:
                if header:
                    if line.strip() == "EST_Header_End":
                        header = False
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    rows.append((float(parts[0]), int(parts[1]), float(parts[2])))
    if not rows:
        return None
    a = np.array(rows)
    f0 = np.where(a[:, 1] > 0, a[:, 2], 0.0)
    return a[:, 0], f0


def process_ddk(af):
    import parselmouth
    from scipy.io import wavfile
    sr, y = wavfile.read(str(af.path))
    is_int = np.issubdtype(y.dtype, np.integer)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64) / (32768.0 if is_int else 1.0)
    r = {"subject": af.subject, "task": af.task}
    pk_t, pk_db, (te, env) = detect_syllables(y, sr)
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    trackers = {"cc60": snd.to_pitch_cc(time_step=PITCH_STEP, pitch_floor=60.0, pitch_ceiling=PITCH_CEILING),
                "ac60": snd.to_pitch_ac(time_step=PITCH_STEP, pitch_floor=60.0, pitch_ceiling=PITCH_CEILING),
                "cc75": snd.to_pitch_cc(time_step=PITCH_STEP, pitch_floor=75.0, pitch_ceiling=PITCH_CEILING),
                "ac75": snd.to_pitch_ac(time_step=PITCH_STEP, pitch_floor=75.0, pitch_ceiling=PITCH_CEILING)}
    meds = {}
    contours = {name: (p.selected_array["frequency"], p.xs()) for name, p in trackers.items()}
    rp = reaper_f0(af.path)
    if rp is not None:
        contours["reaper"] = (rp[1], rp[0])
    for name, (f0, tt) in contours.items():
        med = nucleus_f0(f0, tt, pk_t)
        meds[name] = med
        r[f"syl_sd_{name}"] = syl_sd(med)
        ok = med[np.isfinite(med)]
        if len(ok) >= 5:
            fold, octv = semitones_folded(ok, float(np.median(ok)))
            r[f"syl_sd_fold_{name}"] = float(fold.std(ddof=1))
            r[f"oct_frac_{name}"] = float(octv.mean())
    # split-half (odd / even syllables) on the main tracker
    med = meds["cc60"]
    r["syl_sd_odd"] = syl_sd(med[0::2]); r["syl_sd_even"] = syl_sd(med[1::2])
    # SHR validation of octave-level nuclei (main tracker)
    ok = np.isfinite(med)
    if ok.sum() >= 5:
        ref = float(np.median(med[ok]))
        st = 12 * np.log2(med / ref)
        flagged = np.where(ok & (np.abs(st) > 6))[0]
        shr_vals, down = [], []
        for i in flagged:
            f_low = med[i] if med[i] < ref else ref  # candidate lower fundamental
            shr_vals.append(shr_db(y, sr, pk_t[i], f_low))
            down.append(bool(med[i] < ref))        # octave-down flag: the SHR test is informative
        shr_arr, down_arr = np.array(shr_vals), np.array(down, dtype=bool)
        r["n_nuclei"] = int(ok.sum()); r["n_oct"] = int(len(flagged))
        r["n_oct_genuine"] = int(np.sum(shr_arr > -6)) if shr_vals else 0
        r["n_oct_down"] = int(down_arr.sum()) if shr_vals else 0
        r["n_oct_down_genuine"] = int(np.sum((shr_arr > -6) & down_arr)) if shr_vals else 0
        r["n_oct_up"] = int((~down_arr).sum()) if shr_vals else 0
        r["shr_down_median"] = float(np.nanmedian(shr_arr[down_arr])) if down_arr.any() else np.nan
        # control: SHR at the same definition for unflagged nuclei using half their F0 as f_low
        unf = np.where(ok & (np.abs(st) <= 6))[0]
        ctrl = [shr_db(y, sr, pk_t[i], med[i] / 2) for i in unf[:: max(1, len(unf) // 8)][:8]]
        r["shr_ctrl_median"] = float(np.nanmedian(ctrl)) if ctrl else np.nan
        r["n_ctrl"] = int(np.sum(np.isfinite(ctrl))) if ctrl else 0
        r["n_ctrl_above"] = int(np.sum(np.array(ctrl) > -6)) if ctrl else 0   # control false positives at the -6 dB threshold
        r["shr_flag_median"] = float(np.nanmedian(shr_vals)) if shr_vals else np.nan
    # recording covariates: level, noise floor, SNR, rate, n syllables, mean F0
    r["level_db"] = float(np.percentile(env, 95)); r["floor_db"] = float(np.percentile(env, 5))
    r["snr_db"] = r["level_db"] - r["floor_db"]
    r["n_syll"] = int(len(pk_t))
    r["ddk_rate"] = float(1.0 / np.median(np.diff(pk_t))) if len(pk_t) >= 3 else np.nan
    f0 = trackers["cc60"].selected_array["frequency"]
    r["f0_mean_hz"] = float(f0[f0 > 0].mean()) if (f0 > 0).sum() >= 5 else np.nan
    return r


def generic_functionals(af):
    """Generic prosodic/spectral descriptors for any task (eGeMAPS-like functionals) + MFCC-39."""
    import parselmouth
    snd = parselmouth.Sound(str(af.path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    r = {"subject": af.subject, "task": af.task, "dur": snd.duration}
    p = snd.to_pitch_cc(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = p.selected_array["frequency"]; v = f0[f0 > 0]
    r["voiced_frac"] = float((f0 > 0).mean())
    if len(v) >= 5:
        st = 12 * np.log2(v / 55.0)  # semitones re 55 Hz (eGeMAPS convention: 27.5 Hz; any constant works)
        r["f0_st_mean"] = float(st.mean()); r["f0_st_sd"] = float(st.std(ddof=1))
        r["f0_st_range"] = float(np.percentile(st, 90) - np.percentile(st, 10))
        tt = p.xs()[f0 > 0]
        r["f0_st_slope"] = float(np.polyfit(tt, st, 1)[0]) if len(v) > 10 else np.nan
    inten = snd.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=PITCH_STEP).values[0]
    inten = inten[np.isfinite(inten)]
    r["int_mean"] = float(inten.mean()); r["int_sd"] = float(inten.std(ddof=1))
    r["int_range"] = float(np.percentile(inten, 90) - np.percentile(inten, 10))
    from parselmouth.praat import call
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        r["jitter"] = 100 * float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        r["shimmer"] = 100 * float(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        h = call(snd, "To Harmonicity (cc)", PITCH_STEP, PITCH_FLOOR, 0.1, 1.0)
        r["hnr"] = float(call(h, "Get mean", 0, 0))
    except Exception:
        pass
    mf = snd.to_mfcc(number_of_coefficients=13, window_length=0.025, time_step=0.01, firstFilterFrequency=100.0,
                     distance_between_filters=100.0, maximum_frequency=4000.0).to_array()[1:]
    d = np.diff(mf, axis=1)
    for j, x in enumerate(np.concatenate([mf.mean(1), mf.std(1), np.abs(d).mean(1)])):
        r[f"m{j}"] = float(x)
    return r


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--extended", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = args.out; (out / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features).merge(pd.read_csv(args.extended), on="subject", how="left").copy()
    cols = resolve_columns(df)
    sp = pd.to_numeric(df[cols["speech"]], errors="coerce")
    df["is_als"] = df[cols["category"]].astype(str).str.upper().str.startswith("ALS").astype(int)
    df["group"] = np.where(df.is_als == 0, "HC", np.where(sp >= 4, "ALS-ND", "ALS-D"))
    df["sex_m"] = (df[cols["sex"]].astype(str).str.upper().str[0] == "M").astype(float)
    df["age_num"] = pd.to_numeric(df[cols["age"]], errors="coerce")
    pu_all = pd.to_numeric(df[cols["pumns_bulbar"]], errors="coerce")
    df[cols["pumns_bulbar"]] = pu_all.where(pu_all <= 4)  # one released value (7) exceeds the 0-4 range: treated as missing
    sub = df[df.group.isin(["HC", "ALS-ND"])].reset_index(drop=True)
    y = (sub.group == "ALS-ND").to_numpy().astype(int)
    summary = {}

    # ---------------- R1 subsystem comparison
    main39 = [f for fs in TRACKS.values() for f in fs if f in df.columns]
    track_of = {f: t for t, fs in TRACKS.items() for f in fs}
    eff = pd.read_csv(out / "tables" / "table2_feature_effects.csv")
    nd = eff[eff.contrast == "HC vs ALS-ND"].set_index("feature")
    best = {}
    for t in TRACKS:
        fs = [f for f in TRACKS[t] if f in nd.index]
        best[t] = max(fs, key=lambda f: abs(nd.loc[f, "auc"] - 0.5))
    rows = []
    x_ph = sub[best["Phonation"]].to_numpy(float)
    for t in ["Articulation", "Respiratory support"]:
        x2 = sub[best[t]].to_numpy(float)
        # orient both to AUC >= 0.5 direction so differences are in magnitude
        s1 = np.sign(nd.loc[best["Phonation"], "auc"] - 0.5) or 1
        s2 = np.sign(nd.loc[best[t], "auc"] - 0.5) or 1
        m, lo, hi, p_le0 = paired_boot_auc_diff(y, s1 * x_ph, s2 * x2, args.boot, rng)
        rows.append({"comparison": f"single: {best['Phonation']} vs {best[t]}", "auc1": float(nd.loc[best['Phonation'], 'auc']),
                     "auc2": float(abs(nd.loc[best[t], 'auc'] - 0.5) + 0.5), "diff": m, "ci_lo": lo, "ci_hi": hi, "p_boot_le0": p_le0})
    # CV composites per subsystem
    oof = {}
    for t, fs in TRACKS.items():
        fl = [f for f in fs if f in sub.columns]
        X = sub[fl + ["age_num", "sex_m"]].to_numpy(float)
        r = cv_auc(X, y, 5, args.repeats, args.seed, 0.3)
        oof[t] = r["oof"]
        lo, hi = pooled_auc_ci(y, r["oof"], args.boot, rng)
        rows.append({"comparison": f"composite: {t} ({len(fl)} measures)", "auc1": r["auc_pooled"], "auc2": np.nan, "diff": np.nan,
                     "ci_lo": lo, "ci_hi": hi, "p_boot_le0": np.nan})
    for t in ["Articulation", "Respiratory support"]:
        m, lo, hi, p_le0 = paired_boot_auc_diff(y, oof["Phonation"], oof[t], args.boot, rng)
        rows.append({"comparison": f"composite: Phonation vs {t}", "auc1": np.nan, "auc2": np.nan, "diff": m, "ci_lo": lo, "ci_hi": hi, "p_boot_le0": p_le0})
    r1 = pd.DataFrame(rows); r1.to_csv(out / "tables" / "table10_subsystem_comparison.csv", index=False)
    print("R1 subsystem comparison:\n", r1.round(3).to_string(index=False))

    # ---------------- R2 sensitivity: nested bulbar-normal subsets
    sal = pd.to_numeric(df[cols["salivation"]], errors="coerce"); sw = pd.to_numeric(df[cols["swallowing"]], errors="coerce")
    pu = pd.to_numeric(df[cols["pumns_bulbar"]], errors="coerce"); mrc = pd.to_numeric(df[cols["mrc_head"]], errors="coerce")
    defs = {"ALS-ND (speech item = 4)": df.group == "ALS-ND",
            "+ salivation = swallowing = 4": (df.group == "ALS-ND") & (sal == 4) & (sw == 4),
            "+ MRC head = 15": (df.group == "ALS-ND") & (sal == 4) & (sw == 4) & (mrc == 15),
            "+ PUMNS bulbar = 0": (df.group == "ALS-ND") & (sal == 4) & (sw == 4) & (mrc == 15) & (pu == 0)}
    hc = df[df.group == "HC"]
    keys = [KEY, "DDK__macro_f0_sd_st", "DDK__macro_oct_frac", "V__f0_sd_st_fold", "V__jitter_ppq5", "ART__fcr", "DDK__ddk_rate", "DDK__syll_per_run"]
    rows = []
    from analyze_subclinical import auc_boot_ci
    for dname, mask in defs.items():
        g = df[mask]
        for f in keys:
            a, p = auc_mwu(hc[f].to_numpy(float), g[f].to_numpy(float))
            lo, hi = auc_boot_ci(hc[f].to_numpy(float), g[f].to_numpy(float), args.boot, rng)
            rows.append({"subset": dname, "n": int(mask.sum()), "feature": f, "auc": a, "ci_lo": lo, "ci_hi": hi, "p": p})
    r2 = pd.DataFrame(rows); r2.to_csv(out / "tables" / "table11_sensitivity_bulbar_normal.csv", index=False)
    print("\nR2 sensitivity:\n", r2.pivot(index="feature", columns="subset", values="auc").round(3).to_string())
    print(r2.groupby("subset")["n"].first().to_dict())

    # ---------------- audio-based parts (R3-R6)
    files = discover(args.data)
    ddk_files = [f for f in files if f.task in DDK_TASKS]
    from joblib import Parallel, delayed
    cache = out / "tables" / "revision_ddk_per_file.csv"
    if cache.exists():
        pf = pd.read_csv(cache)
    else:
        pf = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(process_ddk)(f) for f in ddk_files))
        pf.to_csv(cache, index=False)
    pf["short"] = pf.task.map(TASK_SHORT)
    agg = pf.groupby("subject").mean(numeric_only=True)
    sub2 = sub.merge(agg, left_on="subject", right_index=True, how="left", suffixes=("", "_rev"))

    # R3 trackers
    rows = []
    base = sub2["syl_sd_cc60"].to_numpy(float)
    for name in ["cc60", "ac60", "cc75", "ac75", "reaper"]:
        for kind in ["syl_sd", "syl_sd_fold", "oct_frac"]:
            if f"{kind}_{name}" not in sub2.columns:
                continue
            x = sub2[f"{kind}_{name}"].to_numpy(float)
            a, p = auc_mwu(x[y == 0], x[y == 1])
            rho = stats.spearmanr(x, base, nan_policy="omit").statistic
            # agreement with the same measure on the main tracker (per speaker, HC + ALS-ND)
            rho_same = stats.spearmanr(x, sub2[f"{kind}_cc60"].to_numpy(float), nan_policy="omit").statistic
            rows.append({"tracker": name, "measure": kind, "auc": a, "p": p, "rho_vs_main": float(rho), "rho_same_measure": float(rho_same),
                         "median_hc": float(np.nanmedian(x[y == 0])), "median_nd": float(np.nanmedian(x[y == 1]))})
    r3 = pd.DataFrame(rows); r3.to_csv(out / "tables" / "table12_tracker_robustness.csv", index=False)
    print("\nR3 tracker robustness:\n", r3.round(3).to_string(index=False))
    # SHR validation
    tot = pf.groupby("subject")[["n_nuclei", "n_oct", "n_oct_genuine", "n_oct_down", "n_oct_down_genuine", "n_oct_up"]].sum()
    tot["oct_frac"] = tot.n_oct / tot.n_nuclei; tot["oct_genuine_frac"] = tot.n_oct_genuine / tot.n_nuclei
    tot["oct_spurious_frac"] = (tot.n_oct - tot.n_oct_genuine) / tot.n_nuclei
    tot["oct_down_frac"] = tot.n_oct_down / tot.n_nuclei; tot["oct_down_genuine_frac"] = tot.n_oct_down_genuine / tot.n_nuclei
    tot["oct_down_spurious_frac"] = (tot.n_oct_down - tot.n_oct_down_genuine) / tot.n_nuclei; tot["oct_up_frac"] = tot.n_oct_up / tot.n_nuclei
    s3 = sub.merge(tot, left_on="subject", right_index=True, how="left")
    ctrl_tot = pf[["n_ctrl", "n_ctrl_above"]].sum()
    shr = {"ctrl_nuclei_tested": int(ctrl_tot.n_ctrl), "ctrl_above_threshold": int(ctrl_tot.n_ctrl_above),
           "ctrl_fp_rate": float(ctrl_tot.n_ctrl_above / max(ctrl_tot.n_ctrl, 1)),
           "flagged_nuclei_total": int(tot.n_oct.sum()), "genuine_by_SHR": int(tot.n_oct_genuine.sum()),
           "flagged_down": int(tot.n_oct_down.sum()), "flagged_down_genuine": int(tot.n_oct_down_genuine.sum()), "flagged_up": int(tot.n_oct_up.sum()),
           "shr_flag_median_db": float(np.nanmedian(pf.shr_flag_median)), "shr_down_median_db": float(np.nanmedian(pf.shr_down_median)),
           "shr_ctrl_median_db": float(np.nanmedian(pf.shr_ctrl_median))}
    for c in ["oct_frac", "oct_genuine_frac", "oct_spurious_frac", "oct_down_frac", "oct_down_genuine_frac", "oct_down_spurious_frac", "oct_up_frac"]:
        x = s3[c].to_numpy(float); a, p = auc_mwu(x[y == 0], x[y == 1])
        shr[f"auc_{c}"] = a; shr[f"p_{c}"] = p; shr[f"median_hc_{c}"] = float(np.nanmedian(x[y == 0])); shr[f"median_nd_{c}"] = float(np.nanmedian(x[y == 1]))
    summary["R3_shr"] = shr
    print("\nR3 SHR validation:", json.dumps(shr, indent=1))

    # R4 covariates
    rows = []
    key = sub2[KEY].to_numpy(float)
    for c in ["snr_db", "level_db", "ddk_rate", "n_syll", "f0_mean_hz", "age_num"]:
        x = sub2[c].to_numpy(float)
        rho_all = stats.spearmanr(key, x, nan_policy="omit").statistic
        rho_hc = stats.spearmanr(key[y == 0], x[y == 0], nan_policy="omit").statistic
        a, p = auc_mwu(x[y == 0], x[y == 1])
        rows.append({"covariate": c, "rho_with_marker_all": float(rho_all), "rho_with_marker_HC": float(rho_hc),
                     "auc_group_difference": a, "p_group": p, "median_hc": float(np.nanmedian(x[y == 0])), "median_nd": float(np.nanmedian(x[y == 1]))})
    r4 = pd.DataFrame(rows)
    Z = sub2[["age_num", "sex_m", "snr_db", "level_db", "ddk_rate", "n_syll"]].to_numpy(float)
    res = residualise(key, Z)
    a_adj, p_adj = auc_mwu(res[y == 0], res[y == 1])
    Z2 = sub2[["age_num", "sex_m"]].to_numpy(float)
    res2 = residualise(key, Z2); a_adj2, _ = auc_mwu(res2[y == 0], res2[y == 1])
    summary["R4_marker_adjusted"] = {"auc_raw": float(auc_mwu(key[y == 0], key[y == 1])[0]), "auc_adj_age_sex": a_adj2,
                                     "auc_adj_age_sex_snr_level_rate_nsyll": a_adj, "p_adj_full": p_adj}
    r4.to_csv(out / "tables" / "table13_covariates.csv", index=False)
    print("\nR4 covariates:\n", r4.round(3).to_string(index=False)); print(summary["R4_marker_adjusted"])

    # R5 repeatability
    wide = pf.pivot(index="subject", columns="short", values="syl_sd_cc60")[["PA", "TA", "KA"]].dropna()
    icc_all = icc21(wide.to_numpy())
    wide_hc = wide.loc[wide.index.isin(df[df.group == "HC"].subject)]
    icc_hc = icc21(wide_hc.to_numpy())
    rho_pairs = {f"{a}-{b}": float(stats.spearmanr(wide[a], wide[b]).statistic) for a, b in [("PA", "TA"), ("PA", "KA"), ("TA", "KA")]}
    sh = pf.dropna(subset=["syl_sd_odd", "syl_sd_even"])
    r_sh = {}
    for t, g in sh.groupby("short"):
        r = stats.spearmanr(g.syl_sd_odd, g.syl_sd_even).statistic
        r_sh[t] = {"spearman_odd_even": float(r), "spearman_brown": float(2 * r / (1 + r)), "n": int(len(g))}
    summary["R5_repeatability"] = {"icc21_across_syllables_all": icc_all, "icc21_across_syllables_HC": icc_hc, "n_subjects": int(len(wide)),
                                   "spearman_between_tasks": rho_pairs, "split_half": r_sh}
    print("\nR5 repeatability:", json.dumps(summary["R5_repeatability"], indent=1))

    # R6 generic reference
    cache2 = out / "tables" / "revision_generic_per_file.csv"
    if cache2.exists():
        gf = pd.read_csv(cache2)
    else:
        gf = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(generic_functionals)(f) for f in files))
        gf.to_csv(cache2, index=False)
    gcols = [c for c in gf.columns if c not in ("subject", "task")]
    pros = [c for c in gcols if not c.startswith("m")]
    mf = [c for c in gcols if c.startswith("m")]
    wide = {}
    for t, g in gf.groupby("task"):
        gg = g.set_index("subject")[gcols]; gg.columns = [f"{TASK_SHORT[t]}__g_{c}" for c in gcols]; wide[t] = gg
    vmean = pd.concat([wide[t].set_axis(gcols, axis=1) for t in VOWEL_TASKS]).groupby(level=0).mean(); vmean.columns = [f"V__g_{c}" for c in gcols]
    dmean = pd.concat([wide[t].set_axis(gcols, axis=1) for t in DDK_TASKS]).groupby(level=0).mean(); dmean.columns = [f"DDK__g_{c}" for c in gcols]
    G = pd.concat(list(wide.values()) + [vmean, dmean], axis=1)
    s6 = sub.merge(G, left_on="subject", right_index=True, how="left")
    rows = []
    protos = {f"/{TASK_SHORT[t].lower()}/ only": [f"{TASK_SHORT[t]}__g_{c}" for c in gcols] for t in VOWEL_TASKS + DDK_TASKS}
    protos["All 8 tasks"] = list(vmean.columns) + list(dmean.columns)
    protos_pros = {k + " (prosodic only)": [c for c in v if c.split("_g_", 1)[1] in pros] for k, v in protos.items()}
    for name, fl in {**protos, **protos_pros}.items():
        X = s6[fl + ["age_num", "sex_m"]].to_numpy(float)
        r = cv_auc(X, y, 5, args.repeats, args.seed, 0.3)
        lo, hi = pooled_auc_ci(y, r["oof"], args.boot, rng)
        rows.append({"protocol": name, "n_features": len(fl), "auc_pooled": r["auc_pooled"], "ci_lo": lo, "ci_hi": hi, "brier": r["brier"]})
        print(f"  R6 {name:32s} AUC={r['auc_pooled']:.3f} [{lo:.3f},{hi:.3f}]")
    r6 = pd.DataFrame(rows); r6.to_csv(out / "tables" / "table14_generic_reference.csv", index=False)

    (out / "summary_revision.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\nwrote revision tables to", out / "tables")


if __name__ == "__main__":
    main()
