#!/usr/bin/env python3
"""
Ablation of the generic descriptor set, age/sex floor and permutation null (second-round review).

A1  Age + sex only (floor for every CV model).
A2  Generic set split into blocks: F0 functionals (semitone mean, SD, 10-90 range, slope), voice quality
    (jitter, shimmer, HNR), intensity/duration/voicing (intensity mean, SD, range, voiced fraction, duration);
    each block alone and all 12, per task and for all tasks; summary as median over the 8 single tasks
    (the best single task is a post-hoc choice among 16 combinations).
A3  Permutation null of the cross-validated AUC: labels shuffled, same CV protocol; distribution of the
    pooled out-of-fold AUC for each subsystem composite and for the generic all-task model. Gives the
    reference against which 'below 0.5' composites are read (CV has a negative bias under the null).

Usage: python analyze_ablation.py --features results/features.csv --out results/analysis [--perms 200]
Needs results/analysis/tables/revision_generic_per_file.csv (written by analyze_revision.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_subclinical import resolve_columns, cv_auc, pooled_auc_ci, TRACKS  # noqa: E402
from extract_features import VOWEL_TASKS, DDK_TASKS, TASK_SHORT  # noqa: E402

BLOCKS = {
    "F0 functionals (4)": ["f0_st_mean", "f0_st_sd", "f0_st_range", "f0_st_slope"],
    "Voice quality (3)": ["jitter", "shimmer", "hnr"],
    "Intensity/duration/voicing (5)": ["int_mean", "int_sd", "int_range", "voiced_frac", "dur"],
}
ALL12 = [c for b in BLOCKS.values() for c in b]
BLOCKS["All but F0 (8)"] = BLOCKS["Voice quality (3)"] + BLOCKS["Intensity/duration/voicing (5)"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = args.out
    df = pd.read_csv(args.features)
    cols = resolve_columns(df)
    sp = pd.to_numeric(df[cols["speech"]], errors="coerce")
    df["is_als"] = df[cols["category"]].astype(str).str.upper().str.startswith("ALS").astype(int)
    df["group"] = np.where(df.is_als == 0, "HC", np.where(sp >= 4, "ALS-ND", "ALS-D"))
    df["sex_m"] = (df[cols["sex"]].astype(str).str.upper().str[0] == "M").astype(float)
    df["age_num"] = pd.to_numeric(df[cols["age"]], errors="coerce")
    sub = df[df.group.isin(["HC", "ALS-ND"])].reset_index(drop=True)
    y = (sub.group == "ALS-ND").to_numpy().astype(int)

    gf = pd.read_csv(out / "tables" / "revision_generic_per_file.csv")
    wide = {}
    for t, g in gf.groupby("task"):
        gg = g.set_index("subject")[ALL12]
        gg.columns = [f"{TASK_SHORT[t]}__g_{c}" for c in ALL12]
        wide[t] = gg
    vmean = pd.concat([wide[t].set_axis(ALL12, axis=1) for t in VOWEL_TASKS]).groupby(level=0).mean(); vmean.columns = [f"V__g_{c}" for c in ALL12]
    dmean = pd.concat([wide[t].set_axis(ALL12, axis=1) for t in DDK_TASKS]).groupby(level=0).mean(); dmean.columns = [f"DDK__g_{c}" for c in ALL12]
    G = pd.concat(list(wide.values()) + [vmean, dmean], axis=1)
    s = sub.merge(G, left_on="subject", right_index=True, how="left")

    def run(cols_, label, with_cov=True):
        X = s[cols_ + (["age_num", "sex_m"] if with_cov else [])].to_numpy(float) if cols_ else s[["age_num", "sex_m"]].to_numpy(float)
        r = cv_auc(X, y, 5, args.repeats, args.seed, 0.3)
        lo, hi = pooled_auc_ci(y, r["oof"], args.boot, rng)
        return {"model": label, "n_features": len(cols_), "auc_pooled": r["auc_pooled"], "ci_lo": lo, "ci_hi": hi, "brier": r["brier"]}, r["oof"]

    rows = []
    # A1 floor
    rec, _ = run([], "age + sex only"); rows.append(rec); print("A1", rec)
    # A2 blocks per task and pooled
    tasks = {f"/{TASK_SHORT[t].lower()}/": TASK_SHORT[t] for t in VOWEL_TASKS + DDK_TASKS}
    for bname, bcols in list(BLOCKS.items()) + [("All 12", ALL12)]:
        per_task = []
        for tl, short in tasks.items():
            rec, _ = run([f"{short}__g_{c}" for c in bcols], f"{bname} | {tl}"); rec["block"] = bname; rec["task"] = tl; rows.append(rec); per_task.append(rec["auc_pooled"])
        rec, _ = run([f"V__g_{c}" for c in bcols] + [f"DDK__g_{c}" for c in bcols], f"{bname} | all 8 tasks (means)"); rec["block"] = bname; rec["task"] = "all"; rows.append(rec)
        rows.append({"model": f"{bname} | median over 8 single tasks", "block": bname, "task": "median", "n_features": len(bcols),
                     "auc_pooled": float(np.median(per_task)), "ci_lo": float(np.min(per_task)), "ci_hi": float(np.max(per_task)), "brier": np.nan})
        print(f"A2 {bname:32s} median over tasks {np.median(per_task):.3f} (min {np.min(per_task):.3f}, max {np.max(per_task):.3f}); all tasks {rec['auc_pooled']:.3f}")
    abl = pd.DataFrame(rows)
    abl.to_csv(out / "tables" / "table15_generic_ablation.csv", index=False)

    if args.perms <= 0:
        print("wrote table15 (permutation null skipped; summary_ablation.json left unchanged)")
        return
    # A3 permutation null for the subsystem composites and the generic all-task model
    null = {}
    comp = {t: [f for f in fs if f in sub.columns] for t, fs in TRACKS.items()}
    comp["Generic all tasks (24)"] = [f"V__g_{c}" for c in ALL12] + [f"DDK__g_{c}" for c in ALL12]
    comp["Age + sex only"] = []
    prng = np.random.default_rng(args.seed + 1)
    for name, fl in comp.items():
        X = s[fl + ["age_num", "sex_m"]].to_numpy(float) if fl else s[["age_num", "sex_m"]].to_numpy(float)
        obs = cv_auc(X, y, 5, args.repeats, args.seed, 0.3)["auc_pooled"]
        vals = []
        for i in range(args.perms):
            yp = prng.permutation(y)
            vals.append(cv_auc(X, yp, 5, 3, args.seed + i, 0.3)["auc_pooled"])
        vals = np.array(vals)
        null[name] = {"observed": float(obs), "null_median": float(np.median(vals)), "null_q025": float(np.percentile(vals, 2.5)),
                      "null_q975": float(np.percentile(vals, 97.5)), "p_perm": float((np.sum(vals >= obs) + 1) / (len(vals) + 1)), "n_perm": int(len(vals))}
        print(f"A3 {name:28s} observed {obs:.3f}  null median {np.median(vals):.3f} [{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]  p_perm {null[name]['p_perm']:.3f}")
    (out / "summary_ablation.json").write_text(json.dumps(null, indent=1))
    print("wrote table15 and summary_ablation.json")


if __name__ == "__main__":
    main()
