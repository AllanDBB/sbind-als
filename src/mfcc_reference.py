#!/usr/bin/env python3
"""
Non-interpretable reference model: MFCC statistics per task + regularised logistic regression,
evaluated with the same repeated stratified CV as the interpretable measures. This tells us whether
the subclinical ceiling comes from our hand-crafted measures or from the data itself.

Usage: python mfcc_reference.py --data data/raw --features results/features.csv --out results/analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import discover, VOWEL_TASKS, DDK_TASKS, TASK_SHORT  # noqa: E402
from analyze_subclinical import cv_auc, pooled_auc_ci, resolve_columns  # noqa: E402


def mfcc_stats(path: Path, n_mfcc: int = 13) -> np.ndarray:
    import parselmouth
    snd = parselmouth.Sound(str(path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    # Praat MFCC: 13 coefficients (c0 excluded here), 25 ms window, 10 ms step, up to 4 kHz
    mfcc = snd.to_mfcc(number_of_coefficients=n_mfcc, window_length=0.025, time_step=0.01,
                       firstFilterFrequency=100.0, distance_between_filters=100.0, maximum_frequency=4000.0)
    m = mfcc.to_array()  # (n_mfcc+1, n_frames); row 0 = c0 (energy)
    inten = snd.to_intensity(minimum_pitch=60.0, time_step=0.01).values[0]
    thr = np.nanpercentile(inten, 95) - 25
    keep = np.zeros(m.shape[1], dtype=bool)
    n = min(len(inten), m.shape[1])
    keep[:n] = inten[:n] > thr                     # drop silence frames
    if keep.sum() < 10:
        keep[:] = True
    c = m[1:, keep]
    d = np.diff(c, axis=1) if c.shape[1] > 1 else np.zeros((c.shape[0], 1))
    return np.concatenate([c.mean(1), c.std(1), np.abs(d).mean(1)])  # 39-d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path, help="features.csv (for groups, age, sex)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    files = discover(args.data)
    cache = args.out / "mfcc_stats.csv"
    if cache.exists():
        mf = pd.read_csv(cache)
    else:
        rows = []
        for i, f in enumerate(files, 1):
            v = mfcc_stats(f.path)
            rows.append({"subject": f.subject, "task": f.task, **{f"m{j}": x for j, x in enumerate(v)}})
            if i % 200 == 0:
                print(f"  {i}/{len(files)}", flush=True)
        mf = pd.DataFrame(rows)
        args.out.mkdir(parents=True, exist_ok=True)
        mf.to_csv(cache, index=False)

    df = pd.read_csv(args.features)
    cols = resolve_columns(df)
    cat = df[cols["category"]].astype(str).str.upper()
    speech = pd.to_numeric(df[cols["speech"]], errors="coerce")
    df["is_als"] = cat.str.startswith("ALS").astype(int)
    df["group"] = np.where(df["is_als"] == 0, "HC", np.where(speech >= 4, "ALS-ND", "ALS-D"))
    df["sex_m"] = (df[cols["sex"]].astype(str).str.upper().str[0] == "M").astype(float)
    df["age_num"] = pd.to_numeric(df[cols["age"]], errors="coerce")

    mcols = [c for c in mf.columns if c.startswith("m")]
    wide = {}
    for task, g in mf.groupby("task"):
        gg = g.set_index("subject")[mcols]
        gg.columns = [f"{TASK_SHORT[task]}__{c}" for c in mcols]
        wide[task] = gg
    # vowel-mean and ddk-mean representations
    vmean = pd.concat([wide[t].set_axis(mcols, axis=1) for t in VOWEL_TASKS]).groupby(level=0).mean()
    vmean.columns = [f"V__{c}" for c in mcols]
    dmean = pd.concat([wide[t].set_axis(mcols, axis=1) for t in DDK_TASKS]).groupby(level=0).mean()
    dmean.columns = [f"DDK__{c}" for c in mcols]
    X_all = pd.concat(list(wide.values()) + [vmean, dmean], axis=1)
    df = df.merge(X_all, left_on="subject", right_index=True, how="left")

    sets = {f"/{TASK_SHORT[t].lower()}/ only": list(wide[t].columns) for t in VOWEL_TASKS + DDK_TASKS}
    sets["5 vowels (mean)"] = list(vmean.columns)
    sets["3 DDK (mean)"] = list(dmean.columns)
    sets["All 8 tasks"] = list(vmean.columns) + list(dmean.columns)

    recs = []
    for contrast, mask in [("HC vs ALS-ND", df.group.isin(["HC", "ALS-ND"])), ("HC vs all ALS", df.is_als >= 0),
                           ("HC vs ALS-D", df.group.isin(["HC", "ALS-D"]))]:
        sub = df[mask]
        y = sub["is_als"].to_numpy()
        for name, fl in sets.items():
            X = sub[fl + ["age_num", "sex_m"]].to_numpy(dtype=float)
            r = cv_auc(X, y, args.folds, args.repeats, args.seed, args.C)
            lo, hi = pooled_auc_ci(y, r["oof"], args.boot, rng)
            recs.append({"contrast": contrast, "protocol": name, "representation": "MFCC-39 stats", "n": len(y),
                         "auc_pooled": r["auc_pooled"], "ci_lo": lo, "ci_hi": hi, "brier": r["brier"]})
            print(f"  {contrast:14s} {name:18s} MFCC AUC={r['auc_pooled']:.3f} [{lo:.3f},{hi:.3f}]")
    out = pd.DataFrame(recs)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "tables" / "table3b_mfcc_reference.csv", index=False)


if __name__ == "__main__":
    main()
