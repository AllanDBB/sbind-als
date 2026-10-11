#!/usr/bin/env python3
"""
Third-round checks (reviewer round 3).

T1  Tracker-independent subharmonic measure: for EVERY syllable nucleus of the DDK tasks, the SHR is computed at
    half of the octave-folded nucleus F0 (nucleus F0 moved by whole octaves to within +-6 st of the speaker's
    median), and, as a variant, at half of the speaker's median F0 (no per-nucleus pitch decision at all).
    Per speaker: share of nuclei with SHR > -6 dB (all nuclei; and unflagged nuclei only, which tests whether the
    'control false positives' of the main analysis are really subharmonics the tracker did not flag).
T2  Position of octave-level events within each uninterrupted run (runs split where the inter-syllable interval
    exceeds 2.5 x the median, as for 'syllables per run'): rate of events in the final third vs the first two
    thirds, and the group contrast repeated after removing the last 3 syllables of every run.
T3  Same-track folding: folded syllable F0 SD and octave share on the main 10 ms track (the exploratory family
    uses a 2 ms track), with bootstrap CIs.
T4  Which generic intensity/duration/voicing descriptor drives the block: univariate AUC of each descriptor
    (vowel and DDK means) and leave-one-out cross-validated AUC of the block.

Usage: python analyze_round3.py --data data/raw --features results/features.csv --analysis results/analysis [--jobs 2]
Writes results/analysis/tables/table16_round3.csv, table17_idv_descriptors.csv, round3_per_file.csv (cache),
and results/analysis/summary_round3.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import discover, detect_syllables, DDK_TASKS, PITCH_FLOOR, PITCH_CEILING, PITCH_STEP  # noqa: E402
from analyze_revision import shr_db, nucleus_f0  # noqa: E402
from analyze_subclinical import auc_mwu, auc_boot_ci, resolve_columns, cv_auc, pooled_auc_ci  # noqa: E402

SHR_THR = -6.0
OCT_ST = 6.0
DROP_LAST = 3


def runs_of(pk_t):
    """Run index of each syllable; a new run starts after an interval > 2.5 x the median interval."""
    if len(pk_t) < 3:
        return np.zeros(len(pk_t), dtype=int)
    ioi = np.diff(pk_t)
    is_pause = ioi > 2.5 * np.median(ioi)
    return np.concatenate([[0], np.cumsum(is_pause)])


def sd_st(f):
    f = f[np.isfinite(f)]
    if len(f) < 5:
        return np.nan
    return float((12 * np.log2(f / np.median(f))).std(ddof=1))


def fold_st(st):
    return st - 12 * np.round(st / 12)


def process(af):
    import parselmouth
    from scipy.io import wavfile
    sr, y = wavfile.read(str(af.path))
    is_int = np.issubdtype(y.dtype, np.integer)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64) / (32768.0 if is_int else 1.0)
    pk_t, _, _ = detect_syllables(y, sr)
    p = parselmouth.Sound(y, sampling_frequency=sr).to_pitch_cc(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    med = nucleus_f0(p.selected_array["frequency"], p.xs(), pk_t)
    rows = []
    ok = np.isfinite(med)
    if ok.sum() < 5:
        return rows
    ref = float(np.median(med[ok]))
    run = runs_of(pk_t)
    for i, t in enumerate(pk_t):
        r_idx = run == run[i]
        pos_in_run = np.where(np.where(r_idx)[0] == i)[0][0]
        n_run = int(r_idx.sum())
        from_end = n_run - 1 - pos_in_run
        st = 12 * np.log2(med[i] / ref) if ok[i] else np.nan
        f_fold = ref * 2 ** (fold_st(st) / 12) if ok[i] else np.nan
        rows.append({"subject": af.subject, "task": af.task, "i": i, "run": int(run[i]), "n_run": n_run,
                     "pos_rel": pos_in_run / max(n_run - 1, 1), "from_end": int(from_end), "f0": med[i], "ref": ref,
                     "st": st, "octave": bool(ok[i] and abs(st) > OCT_ST),
                     "shr_fold": shr_db(y, sr, t, f_fold / 2) if ok[i] else np.nan,
                     "shr_med": shr_db(y, sr, t, ref / 2)})
    return rows


def process_reaper(af):
    """Every-nucleus SHR at half the octave-folded nucleus F0, with nucleus F0 from REAPER instead of Praat."""
    from scipy.io import wavfile
    from analyze_revision import reaper_f0
    sr, y = wavfile.read(str(af.path))
    is_int = np.issubdtype(y.dtype, np.integer)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64) / (32768.0 if is_int else 1.0)
    pk_t, _, _ = detect_syllables(y, sr)
    rp = reaper_f0(af.path)
    if rp is None:
        return []
    med = nucleus_f0(rp[1], rp[0], pk_t)
    ok = np.isfinite(med)
    if ok.sum() < 5:
        return []
    ref = float(np.median(med[ok]))
    out = []
    for i, t in enumerate(pk_t):
        if not ok[i]:
            continue
        st = 12 * np.log2(med[i] / ref)
        f_fold = ref * 2 ** (fold_st(st) / 12)
        out.append({"subject": af.subject, "task": af.task, "i": i, "shr_fold_rpr": shr_db(y, sr, t, f_fold / 2)})
    return out


def per_speaker(nf):
    """Speaker-level measures from the nucleus table (tasks pooled for shares; SDs per task then averaged)."""
    out = {}
    for sid, g in nf.groupby("subject"):
        d = {}
        v = g[np.isfinite(g.shr_fold)]
        d["subh_share_fold"] = float((v.shr_fold > SHR_THR).mean()) if len(v) else np.nan
        d["subh_share_med"] = float((g.shr_med > SHR_THR).mean())
        d["shr_fold_median"] = float(v.shr_fold.median()) if len(v) else np.nan
        un = v[~v.octave]
        d["subh_share_unflagged"] = float((un.shr_fold > SHR_THR).mean()) if len(un) else np.nan
        voiced = g[np.isfinite(g.st)]
        d["oct_share"] = float(voiced.octave.mean()) if len(voiced) else np.nan
        keep = voiced[voiced.from_end >= DROP_LAST]
        d["oct_share_trim"] = float(keep.octave.mean()) if len(keep) else np.nan
        early = voiced[voiced.pos_rel < 2 / 3]; late = voiced[voiced.pos_rel >= 2 / 3]
        d["oct_share_first2thirds"] = float(early.octave.mean()) if len(early) else np.nan
        d["oct_share_last3rd"] = float(late.octave.mean()) if len(late) else np.nan
        sds, sdf, sdt, sdtf = [], [], [], []
        for _, gt in voiced.groupby("task"):
            f = gt.f0.to_numpy()
            sds.append(sd_st(f))
            st = 12 * np.log2(f / np.median(f)) if len(f) else f
            sdf.append(float(fold_st(st).std(ddof=1)) if len(f) >= 5 else np.nan)
            ft = gt[gt.from_end >= DROP_LAST].f0.to_numpy()
            sdt.append(sd_st(ft))
            stt = 12 * np.log2(ft / np.median(ft)) if len(ft) else ft
            sdtf.append(float(fold_st(stt).std(ddof=1)) if len(ft) >= 5 else np.nan)
        d["syl_sd"] = float(np.nanmean(sds)) if np.isfinite(sds).any() else np.nan
        d["syl_sd_fold10"] = float(np.nanmean(sdf)) if np.isfinite(sdf).any() else np.nan
        d["syl_sd_trim"] = float(np.nanmean(sdt)) if np.isfinite(sdt).any() else np.nan
        d["syl_sd_fold_trim"] = float(np.nanmean(sdtf)) if np.isfinite(sdtf).any() else np.nan
        d["n_nuclei"] = int(len(g))
        out[sid] = d
    return pd.DataFrame.from_dict(out, orient="index")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    T = args.analysis / "tables"
    df = pd.read_csv(args.features)
    cols = resolve_columns(df)
    sp = pd.to_numeric(df[cols["speech"]], errors="coerce")
    is_als = df[cols["category"]].astype(str).str.upper().str.startswith("ALS")
    df["group"] = np.where(~is_als, "HC", np.where(sp >= 4, "ALS-ND", "ALS-D"))
    grp = df.set_index("subject").group

    cache = T / "round3_per_file.csv"
    if cache.exists():
        nf = pd.read_csv(cache)
    else:
        from joblib import Parallel, delayed
        files = [f for f in discover(args.data) if f.task in DDK_TASKS]
        res = Parallel(n_jobs=args.jobs)(delayed(process)(f) for f in files)
        nf = pd.DataFrame([r for rr in res for r in rr])
        nf.to_csv(cache, index=False)
    spk = per_speaker(nf)
    cache_r = T / "round3_per_file_reaper.csv"
    if cache_r.exists():
        nr = pd.read_csv(cache_r)
    else:
        from joblib import Parallel, delayed
        files = [f for f in discover(args.data) if f.task in DDK_TASKS]
        res = Parallel(n_jobs=args.jobs)(delayed(process_reaper)(f) for f in files)
        nr = pd.DataFrame([r for rr in res for r in rr])
        nr.to_csv(cache_r, index=False)
    if len(nr):
        spk["subh_share_fold_reaper"] = nr.groupby("subject").shr_fold_rpr.apply(lambda v: float((v > SHR_THR).mean()))
    # single-run recordings (how often a 'run' is the whole syllable train)
    nruns = nf.groupby(["subject", "task"]).run.max() + 1
    single_run_share = float((nruns == 1).mean())
    spk["group"] = grp.reindex(spk.index)

    rows = []
    measures = ["subh_share_fold", "subh_share_fold_reaper", "subh_share_med", "shr_fold_median", "subh_share_unflagged", "oct_share", "oct_share_trim",
                "oct_share_first2thirds", "oct_share_last3rd", "syl_sd", "syl_sd_fold10", "syl_sd_trim", "syl_sd_fold_trim"]
    for m in measures:
        a = spk.loc[spk.group == "HC", m].to_numpy(float)
        for g in ["ALS-ND", "ALS-D"]:
            b = spk.loc[spk.group == g, m].to_numpy(float)
            auc, p = auc_mwu(a, b)
            lo, hi = auc_boot_ci(a, b, args.boot, rng)
            rows.append({"measure": m, "contrast": f"HC vs {g}", "auc": auc, "ci_lo": lo, "ci_hi": hi, "p": p,
                         "median_hc": float(np.nanmedian(a)), "median_grp": float(np.nanmedian(b))})
    t16 = pd.DataFrame(rows)
    t16.to_csv(T / "table16_round3.csv", index=False)
    print(t16[t16.contrast == "HC vs ALS-ND"].round(3).to_string(index=False))

    # T2 position of octave events: pooled event rate by run third and group
    nf["group"] = nf.subject.map(grp)
    v = nf[np.isfinite(nf.st)].copy()
    v["third"] = np.where(v.pos_rel < 1 / 3, "first", np.where(v.pos_rel < 2 / 3, "middle", "last"))
    pos = v.groupby(["group", "third"]).octave.agg(["mean", "sum", "count"]).reset_index()
    ev = v[v.octave]
    pos_summary = {}
    for g in ["HC", "ALS-ND", "ALS-D"]:
        e = ev[ev.group == g]
        pos_summary[g] = {"n_events": int(len(e)), "share_events_last3rd": float((e.pos_rel >= 2 / 3).mean()) if len(e) else np.nan,
                          "share_events_last3syll": float((e.from_end < DROP_LAST).mean()) if len(e) else np.nan,
                          "share_nuclei_last3syll": float((v[v.group == g].from_end < DROP_LAST).mean()),
                          "rate_by_third": {r.third: float(r["mean"]) for _, r in pos[pos.group == g].iterrows()}}
    print(json.dumps(pos_summary, indent=1))

    # control false-positive rate re-read: SHR at half F0 of unflagged nuclei, by group
    un = v[~v.octave & np.isfinite(v.shr_fold)]
    fp = {g: float((un[un.group == g].shr_fold > SHR_THR).mean()) for g in ["HC", "ALS-ND", "ALS-D"]}
    allrate = {g: float((v[(v.group == g) & np.isfinite(v.shr_fold)].shr_fold > SHR_THR).mean()) for g in ["HC", "ALS-ND", "ALS-D"]}
    print("unflagged positive rate by group", fp, "all nuclei", allrate)

    # T4 intensity/duration/voicing descriptors
    gf = pd.read_csv(T / "revision_generic_per_file.csv")
    gf["is_vowel"] = ~gf.task.isin(DDK_TASKS)
    idv = ["int_mean", "int_sd", "int_range", "voiced_frac", "dur"]
    vm = gf[gf.is_vowel].groupby("subject")[idv].mean().add_prefix("V_")
    dm = gf[~gf.is_vowel].groupby("subject")[idv].mean().add_prefix("D_")
    G = pd.concat([vm, dm], axis=1)
    sub = df[df.group.isin(["HC", "ALS-ND"])].reset_index(drop=True)
    s = sub.merge(G, left_on="subject", right_index=True, how="left")
    y = (s.group == "ALS-ND").to_numpy(int)
    s["sex_m"] = (s[cols["sex"]].astype(str).str.upper().str[0] == "M").astype(float)
    s["age_num"] = pd.to_numeric(s[cols["age"]], errors="coerce")
    rows = []
    for c in G.columns:
        x = s[c].to_numpy(float)
        auc, p = auc_mwu(x[y == 0], x[y == 1])
        lo, hi = auc_boot_ci(x[y == 0], x[y == 1], args.boot, rng)
        rows.append({"descriptor": c, "kind": "univariate", "auc": auc, "ci_lo": lo, "ci_hi": hi, "p": p,
                     "median_hc": float(np.nanmedian(x[y == 0])), "median_nd": float(np.nanmedian(x[y == 1]))})
    allc = list(G.columns)
    X = s[allc + ["age_num", "sex_m"]].to_numpy(float)
    full = cv_auc(X, y, 5, 20, args.seed, 0.3)["auc_pooled"]
    rows.append({"descriptor": "all 10 (block)", "kind": "cv", "auc": full})
    for base in idv:
        keep = [c for c in allc if not c.endswith(base)]
        X = s[keep + ["age_num", "sex_m"]].to_numpy(float)
        r = cv_auc(X, y, 5, 20, args.seed, 0.3)
        rows.append({"descriptor": f"block without {base}", "kind": "cv", "auc": r["auc_pooled"]})
    t17 = pd.DataFrame(rows)
    t17.to_csv(T / "table17_idv_descriptors.csv", index=False)
    print(t17.round(3).to_string(index=False))

    summ = {"position": pos_summary, "unflagged_positive_rate": fp, "all_nuclei_positive_rate": allrate,
            "n_nuclei_total": int(len(nf)), "n_nuclei_voiced": int(len(v)), "drop_last": DROP_LAST,
            "single_run_share": single_run_share}
    (args.analysis / "summary_round3.json").write_text(json.dumps(summ, indent=1))
    print("wrote table16, table17, summary_round3.json")


if __name__ == "__main__":
    main()
