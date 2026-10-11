#!/usr/bin/env python3
"""
Subclinical bulbar involvement from short smartphone speech tasks (VOC-ALS).

Groups
  HC      healthy controls
  ALS-ND  ALS with normal-rated speech (ALSFRS-R speech item = 4)      <- primary contrast vs HC
  ALS-D   ALS with dysarthria (speech item < 4)

Analyses
  1. Per-feature effects: AUC / Cliff's delta with bootstrap CI, Mann-Whitney p, BH-FDR,
     age- and sex-adjusted AUC (residualised), for HC vs ALS-ND and HC vs ALS-D.
  2. Task efficiency (minimal protocol): repeated stratified CV of an L2-logistic model per
     single task (/a/, /e/, /i/, /o/, /u/, /pa/, /ta/, /ka/) and per protocol (vowels, DDK, all).
  3. Clinical anchoring within ALS: Spearman correlations with bulbar / respiratory clinical
     measures (PUMNS bulbar, MRC head, ALSFRS-R bulbar items, FVC%), bulbar vs spinal onset.
  4. Severity gradient: Spearman of each feature with ordinal level HC < ND < mild < moderate < severe.

Usage:
  python analyze_subclinical.py --features results/features.csv --out results/analysis
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------------------

TRACKS = {
    "Phonation": ["V__f0_sd_st", "V__f0_sd_st_all", "V__f0_slope_st", "V__f0_tremor_idx", "V__int_tremor_idx",
                  "V__unvoiced_frac_run", "V__n_voice_breaks",
                  "V__jitter_local", "V__jitter_rap", "V__jitter_ppq5",
                  "V__shimmer_local", "V__shimmer_apq3", "V__shimmer_apq5", "V__shimmer_apq11",
                  "V__hnr", "V__cpps", "V__alpha_ratio_db", "V__int_sd",
                  "DDK__syl_f0_sd_st", "DDK__f0_sd_st_all"],
    "Articulation": ["ART__vsa", "ART__fcr", "ART__f2_ratio", "V__f1_iqr", "V__f2_iqr",
                     "DDK__ddk_rate", "DDK__ioi_cv", "DDK__ioi_rcv", "DDK__ioi_slope", "DDK__amp_cv",
                     "DDK__pause_ratio", "DDK__voiced_frac"],
    "Respiratory support": ["V__mpt_proxy_max", "V__voiced_dur", "V__int_slope", "V__int_drop",
                            "DDK__syll_per_run", "DDK__pause_time_frac", "DDK__amp_slope_db"],
}
PRETTY = {
    "f0_sd_st_all": "F0 SD, whole file (st)", "f0_slope_st": "F0 drift", "f0_tremor_idx": "F0 tremor index",
    "int_tremor_idx": "Intensity tremor index", "unvoiced_frac_run": "Voice-break fraction",
    "n_voice_breaks": "Voice breaks (n)", "alpha_ratio_db": "Spectral balance (alpha)",
    "syl_f0_sd_st": "DDK syllable F0 SD (st)",
    "f0_cv": "F0 CV", "f0_sd_st": "F0 SD, steady (st)", "jitter_local": "Jitter (local)", "jitter_rap": "Jitter (RAP)",
    "jitter_ppq5": "Jitter (PPQ5)", "shimmer_local": "Shimmer (local)", "shimmer_apq3": "Shimmer (APQ3)",
    "shimmer_apq5": "Shimmer (APQ5)", "shimmer_apq11": "Shimmer (APQ11)", "hnr": "HNR", "cpps": "CPPS",
    "int_sd": "Intensity SD", "vsa": "Vowel space area", "fcr": "FCR", "f2_ratio": "F2 i/u ratio",
    "f1_iqr": "F1 IQR", "f2_iqr": "F2 IQR", "ddk_rate": "DDK rate", "ioi_mean": "DDK interval",
    "ioi_cv": "DDK interval CV", "ioi_rcv": "DDK interval robust CV", "ioi_slope": "DDK slowing",
    "amp_cv": "DDK amplitude CV", "voiced_frac": "DDK voiced fraction", "mpt_proxy_max": "Longest phonation",
    "voiced_dur": "Voiced duration", "int_slope": "Intensity slope", "int_drop": "Intensity drop",
    "n_syll": "Syllables (total)", "syll_per_run": "Syllables per run", "pause_time_frac": "DDK pause time fraction",
    "ddk_span": "DDK train duration", "amp_slope_db": "DDK amplitude decay", "pause_ratio": "DDK gap ratio",
}
VOWEL_TASK_FEATS = ["f0_sd_st", "f0_sd_st_all", "f0_tremor_idx", "int_tremor_idx", "unvoiced_frac_run",
                    "jitter_local", "jitter_ppq5", "shimmer_local", "shimmer_apq11", "hnr",
                    "cpps", "alpha_ratio_db", "int_sd", "int_slope", "mpt_proxy", "f1_iqr", "f2_iqr"]
DDK_TASK_FEATS = ["ddk_rate", "ioi_cv", "ioi_rcv", "ioi_slope", "amp_cv", "amp_slope_db", "syll_per_run",
                  "pause_time_frac", "pause_ratio", "voiced_frac", "f0_sd_st_all", "syl_f0_sd_st"]

# clinical columns: canonical name -> regex candidates (matched on lower-case, alphanumerics only)
CLINICAL = {
    "category": [r"^category$", r"^group$", r"^class$"],
    "age": [r"^age"],
    "sex": [r"^sex$", r"^gender$"],
    "speech": [r"speechsubscore", r"^speech$", r"alsfrsr?speech", r"item1"],
    "salivation": [r"salivation"],
    "swallowing": [r"swallow"],
    "alsfrs_total": [r"totalscore", r"alsfrsr?total"],
    "fvc": [r"^fvc"],
    "pumns_bulbar": [r"pumns.*bulbar"],
    "mrc_head": [r"mrc.*head"],
    "onset": [r"onset.*region", r"^onset"],
    "duration": [r"diseaseduration"],
    "progression": [r"progressionrate"],
    "king": [r"kingclinicalstage", r"^king"],
    "dyspnea": [r"dyspnea"],
    "orthopnea": [r"orthopnea"],
    "breathing": [r"breathinginsufficiency", r"respiratoryinsufficiency"],
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_columns(df: pd.DataFrame) -> dict:
    found = {}
    cols = {c: norm(c) for c in df.columns}
    for key, pats in CLINICAL.items():
        for pat in pats:
            hit = [c for c, n in cols.items() if re.search(pat, n) and not c.startswith(("V__", "DDK__", "ART__"))
                   and "__" not in c]
            if hit:
                found[key] = hit[0]
                break
    return found


def pretty(col: str) -> str:
    prefix, base = (col.split("__", 1) if "__" in col else ("", col))
    name = PRETTY.get(base, base)
    if base == "f0_sd_st_all":  # same measure on two task families
        name = "F0 SD, vowel (st)" if prefix == "V" else "F0 SD, DDK (st)"
    return name


# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------

def auc_mwu(a: np.ndarray, b: np.ndarray):
    """AUC = P(b > a) (prob. that the ALS value exceeds the HC value), with MWU p-value."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u = stats.mannwhitneyu(b, a, alternative="two-sided")
    return float(u.statistic / (len(a) * len(b))), float(u.pvalue)


def auc_boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        aa = rng.choice(a, len(a), replace=True)
        bb = rng.choice(b, len(b), replace=True)
        vals[i] = stats.mannwhitneyu(bb, aa).statistic / (len(a) * len(b))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(q, 1.0)
    out[ok] = res
    return out


def residualise(x: np.ndarray, covars: np.ndarray, rank: bool = True) -> np.ndarray:
    """Residuals of x on covariates (with intercept); NaNs preserved.
    rank=True (default): x and the covariates are rank-transformed first, so the adjustment is not
    driven by outliers of skewed measures (jitter, octave rates) and matches the rank-based AUC."""
    ok = np.isfinite(x) & np.all(np.isfinite(covars), axis=1)
    out = np.full_like(x, np.nan, dtype=float)
    if ok.sum() < covars.shape[1] + 3:
        return out
    xx = stats.rankdata(x[ok]) if rank else x[ok]
    Z = covars[ok]
    if rank:
        Z = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])])
    X = np.column_stack([np.ones(ok.sum()), Z])
    beta, *_ = np.linalg.lstsq(X, xx, rcond=None)
    out[ok] = xx - X @ beta
    return out


def pooled_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int, rng: np.random.Generator):
    from sklearn.metrics import roc_auc_score
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------------------
# Task-efficiency models
# --------------------------------------------------------------------------------------

def protocol_feature_sets(df: pd.DataFrame) -> dict:
    sets = {}
    for v in "AEIOU":
        sets[f"/{v.lower()}/ only"] = [f"{v}__{f}" for f in VOWEL_TASK_FEATS if f"{v}__{f}" in df.columns]
    for d in ["PA", "TA", "KA"]:
        sets[f"/{d.lower()}/ only"] = [f"{d}__{f}" for f in DDK_TASK_FEATS if f"{d}__{f}" in df.columns]
    main39 = [f for fs in TRACKS.values() for f in fs if f in df.columns]
    sets["5 vowels (mean)"] = [f for f in main39 if f.startswith(("V__", "ART__"))]
    sets["3 DDK (mean)"] = [f for f in main39 if f.startswith("DDK__")]
    sets["All 8 tasks"] = main39
    # parsimonious protocols
    sets["Phonation only (all tasks)"] = [c for c in TRACKS["Phonation"] if c in df.columns]
    for d in ["PA", "TA", "KA"]:
        if f"{d}__syl_f0_sd_st" in df.columns:
            sets[f"/{d.lower()}/: syllable F0 SD only"] = [f"{d}__syl_f0_sd_st"]
    if "DDK__syl_f0_sd_st" in df.columns:
        sets["3 DDK: syllable F0 SD only"] = ["DDK__syl_f0_sd_st"]
    return {k: v for k, v in sets.items() if len(v) >= 1}


def cv_auc(X: np.ndarray, y: np.ndarray, folds: int, repeats: int, seed: int, C: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    oof = np.zeros((repeats, len(y)))
    aucs = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        r = i // folds
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=C, penalty="l2", max_iter=5000, class_weight="balanced"))
        model.fit(X[tr], y[tr])
        oof[r, te] = model.predict_proba(X[te])[:, 1]
        if (i + 1) % folds == 0:
            aucs.append(roc_auc_score(y, oof[r]))
    p_mean = oof.mean(axis=0)
    return {
        "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan,
        "auc_pooled": float(roc_auc_score(y, p_mean)), "brier": float(brier_score_loss(y, p_mean)),
        "oof": p_mean,
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "grey": "#7F7F7F", "red": "#D55E00"}


COMPACT = {  # curated subset for the 4-page figure (full set goes to the repository / supplement)
    "Phonation": ["DDK__syl_f0_sd_st", "DDK__f0_sd_st_all", "V__f0_sd_st_all", "V__jitter_ppq5",
                  "V__unvoiced_frac_run", "V__hnr", "V__cpps"],
    "Articulation": ["DDK__ddk_rate", "DDK__ioi_cv", "ART__fcr", "ART__vsa"],
    "Respiratory support": ["DDK__syll_per_run", "V__mpt_proxy_max", "V__int_slope"],
}


def fig_forest(eff: pd.DataFrame, path: Path, subset: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tracks = subset if subset is not None else TRACKS
    rows = []
    for track, feats in tracks.items():
        for f in feats:
            if f in set(eff["feature"]):
                rows.append((track, f))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.5, 0.26 * n + 1.6) if subset is None else (3.4, 0.14 * n + 0.72))
    ypos = []
    y = 0
    last_track = None
    labels = []
    for track, f in rows:
        if track != last_track:
            y -= 0.6
            ax.text(0.02, y, track, fontsize=8.5 if subset is None else 7.5, fontweight="bold", va="center", transform=ax.get_yaxis_transform(),
                    zorder=5, bbox=dict(fc="white", ec="none", pad=0.4))
            y -= 1
            last_track = track
        r_nd = eff[(eff.feature == f) & (eff.contrast == "HC vs ALS-ND")].iloc[0]
        r_d = eff[(eff.feature == f) & (eff.contrast == "HC vs ALS-D")]
        ax.plot([r_nd.ci_lo, r_nd.ci_hi], [y, y], color=OKABE["blue"], lw=1.4)
        ax.plot(r_nd.auc, y, "o", color=OKABE["blue"], ms=4.5, zorder=3)
        if len(r_d):
            r_d = r_d.iloc[0]
            ax.plot([r_d.ci_lo, r_d.ci_hi], [y - 0.35, y - 0.35], color=OKABE["orange"], lw=1.0, alpha=0.9)
            ax.plot(r_d.auc, y - 0.35, "s", mfc="white", mec=OKABE["orange"], ms=4, zorder=3)
        mark = ("*" if r_nd.q < 0.05 else "") + ("†" if r_nd.q_adj < 0.05 else "")
        labels.append((y - 0.17, pretty(f) + mark))
        y -= 1.1
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yticks([p for p, _ in labels])
    fs = 8 if subset is None else 7
    ax.set_yticklabels([l for _, l in labels], fontsize=fs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC (0.5 = no difference)" if subset else "AUC  (P[ALS value > HC value]; 0.5 = no difference)", fontsize=fs + 0.5)
    ax.tick_params(axis="x", labelsize=fs)
    if subset is None:
        ax.plot([], [], "o-", color=OKABE["blue"], label="HC vs ALS-ND (speech rated normal)")
        ax.plot([], [], "s-", color=OKABE["orange"], mfc="white", label="HC vs ALS-D (dysarthria)")
        ax.legend(fontsize=fs - 0.5, loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.4 / (0.26 * n + 1.6)), frameon=False, ncol=2)
    else:
        ax.plot([], [], "o-", color=OKABE["blue"], label="HC vs ALS-ND")
        ax.plot([], [], "s-", color=OKABE["orange"], mfc="white", label="HC vs ALS-D")
        ax.legend(fontsize=fs - 0.5, loc="lower center", bbox_to_anchor=(0.35, 1.0), frameon=False, ncol=2, handlelength=1.5, columnspacing=1.0)
    ax.set_ylim(y, 0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_tasks(task_df: pd.DataFrame, path: Path, mfcc_df: pd.DataFrame | None = None, null_band=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    protocols = [p for p in dict.fromkeys(task_df["protocol"]) if p not in ("/ta/: syllable F0 SD only", "/ka/: syllable F0 SD only", "/pa/: syllable F0 SD only", "5 vowels (mean)")]
    contrasts = ["HC vs ALS-ND", "HC vs ALS-D"]
    fig, ax = plt.subplots(figsize=(3.4, 0.165 * len(protocols) + 1.0))
    if null_band is not None:
        ax.axvspan(null_band[0], null_band[1], color="0.92", lw=0, zorder=0)
        ax.axvline(null_band[2], color="0.6", lw=0.8, ls=":", zorder=0)
    for j, c in enumerate(contrasts):
        sub = task_df[task_df.contrast == c].set_index("protocol").reindex(protocols)
        ys = np.arange(len(protocols))[::-1] + (0.2 if j == 0 else -0.2)
        col = OKABE["blue"] if j == 0 else OKABE["orange"]
        ax.errorbar(sub["auc_pooled"], ys, xerr=[sub["auc_pooled"] - sub["ci_lo"], sub["ci_hi"] - sub["auc_pooled"]],
                    fmt="o" if j == 0 else "s", color=col, mfc=col if j == 0 else "white", ms=4, lw=1.1, capsize=1.8,
                    label=c + " (interpretable)")
    if mfcc_df is not None and len(mfcc_df):
        # generic prosodic reference (table14, rows "<protocol> (prosodic only)") when available, else MFCC reference
        if "protocol" in mfcc_df.columns and mfcc_df["protocol"].str.contains("prosodic only").any():
            g = mfcc_df[mfcc_df.protocol.str.endswith("(prosodic only)")].copy()
            g["protocol"] = g.protocol.str.replace(" (prosodic only)", "", regex=False)
            sub = g.set_index("protocol").reindex(protocols)
            lab = "HC vs ALS-ND (generic set, 12/task)"
        else:
            sub = mfcc_df[mfcc_df.contrast == "HC vs ALS-ND"].set_index("protocol").reindex(protocols)
            lab = "HC vs ALS-ND (MFCC reference)"
        ys = np.arange(len(protocols))[::-1] + 0.2
        ax.plot(sub["auc_pooled"], ys, "x", color=OKABE["grey"], ms=5, mew=1.2, label=lab)
    ax.set_yticks(np.arange(len(protocols))[::-1])
    ax.set_yticklabels(protocols, fontsize=6.5)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Cross-validated AUC (95% bootstrap CI)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(fontsize=5.6, frameon=False, loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_roc_calib(curves: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    colors = [OKABE["blue"], OKABE["orange"], OKABE["green"], OKABE["grey"]]
    for (name, (y, p)), col in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=col, lw=1.3, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set_xlabel("1 - specificity", fontsize=8.5); axes[0].set_ylabel("sensitivity", fontsize=8.5)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    name, (y, p) = list(curves.items())[-1]
    frac, mean_pred = calibration_curve(y, p, n_bins=6, strategy="quantile")
    axes[1].plot(mean_pred, frac, "o-", color=OKABE["blue"], lw=1.2, ms=4, label=name)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[1].set_xlabel("predicted probability", fontsize=8.5); axes[1].set_ylabel("observed fraction ALS", fontsize=8.5)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    for ax in axes:
        ax.tick_params(labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--C", type=float, default=0.3, help="L2 logistic regression inverse regularisation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from saved tables (no recomputation)")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = args.out
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        eff = pd.read_csv(out / "tables" / "table2_feature_effects.csv")
        fig_forest(eff, out / "figures" / "figS1_forest_all")
        fig_forest(eff, out / "figures" / "fig1_forest", subset=COMPACT)
        task_df = pd.read_csv(out / "tables" / "table3_task_efficiency.csv")
        gen_path = out / "tables" / "table14_generic_reference.csv"
        mfcc_path = out / "tables" / "table3b_mfcc_reference.csv"
        ref = pd.read_csv(gen_path) if gen_path.exists() else (pd.read_csv(mfcc_path) if mfcc_path.exists() else None)
        nb = None
        sa = out / "summary_ablation.json"
        if sa.exists():
            import json as _json
            g = _json.loads(sa.read_text()).get("Generic all tasks (24)")
            if g:
                nb = (g["null_q025"], g["null_q975"], g["null_median"])
        fig_tasks(task_df, out / "figures" / "fig2_tasks", ref, null_band=nb)
        oof_path = out / "tables" / "oof_predictions.csv"
        if oof_path.exists():
            oof = pd.read_csv(oof_path)
            curves = {c: (g["y"].to_numpy(), g["p"].to_numpy()) for c, g in oof.groupby("protocol", sort=False)}
            fig_roc_calib(curves, out / "figures" / "fig3_roc_calibration")
        print("figures redrawn")
        return

    df = pd.read_csv(args.features)
    cols = resolve_columns(df)
    print("Resolved clinical columns:", json.dumps(cols, indent=1))
    for need in ["category", "speech", "age", "sex"]:
        if need not in cols:
            raise SystemExit(f"Could not find a '{need}' column in {args.features}; columns: {list(df.columns)[:40]}...")

    # ---- groups
    cat = df[cols["category"]].astype(str).str.strip().str.upper()
    speech = pd.to_numeric(df[cols["speech"]], errors="coerce")
    df["is_als"] = cat.str.startswith("ALS").astype(int)
    df["group"] = np.where(df["is_als"] == 0, "HC", np.where(speech >= 4, "ALS-ND", "ALS-D"))
    df["severity"] = np.where(df["is_als"] == 0, 0, 5 - speech.fillna(4))  # HC 0, ND 1, mild 2, mod 3, severe 4
    df["sex_m"] = (df[cols["sex"]].astype(str).str.strip().str.upper().str[0] == "M").astype(float)
    df["age_num"] = pd.to_numeric(df[cols["age"]], errors="coerce")
    counts = df["group"].value_counts().to_dict()
    print("Groups:", counts)

    # ---- Table 1: cohort
    t1 = df.groupby("group").agg(n=("subject", "size"), age_mean=("age_num", "mean"), age_sd=("age_num", "std"),
                                 male_frac=("sex_m", "mean")).reindex(["HC", "ALS-ND", "ALS-D"])
    t1.to_csv(out / "tables" / "table1_cohort.csv")

    # ---- 1. per-feature effects
    hc = df[df.group == "HC"]
    nd = df[df.group == "ALS-ND"]
    dys = df[df.group == "ALS-D"]
    feats = [f for fs in TRACKS.values() for f in fs if f in df.columns]
    track_of = {f: t for t, fs in TRACKS.items() for f in fs}
    recs = []
    for contrast, grp in [("HC vs ALS-ND", nd), ("HC vs ALS-D", dys)]:
        if len(grp) < 5:
            continue
        both = pd.concat([hc, grp])
        covars = both[["age_num", "sex_m"]].to_numpy(dtype=float)
        is_grp = (both.group != "HC").to_numpy()
        for f in feats:
            a, b = hc[f].to_numpy(float), grp[f].to_numpy(float)
            auc, p = auc_mwu(a, b)
            lo, hi = auc_boot_ci(a, b, args.boot, rng)
            res = residualise(both[f].to_numpy(float), covars)
            auc_adj, p_adj = auc_mwu(res[~is_grp], res[is_grp])
            recs.append({"contrast": contrast, "track": track_of[f], "feature": f, "label": pretty(f),
                         "n_hc": int(np.isfinite(a).sum()), "n_grp": int(np.isfinite(b).sum()),
                         "median_hc": float(np.nanmedian(a)), "median_grp": float(np.nanmedian(b)),
                         "auc": auc, "ci_lo": lo, "ci_hi": hi, "cliffs_delta": 2 * auc - 1 if np.isfinite(auc) else np.nan,
                         "p": p, "auc_adj": auc_adj, "p_adj": p_adj})
    eff = pd.DataFrame(recs)
    for c in eff.contrast.unique():
        m = eff.contrast == c
        eff.loc[m, "q"] = bh_fdr(eff.loc[m, "p"].to_numpy())
        eff.loc[m, "q_adj"] = bh_fdr(eff.loc[m, "p_adj"].to_numpy())
    eff["abs_effect"] = (eff["auc"] - 0.5).abs()
    eff = eff.sort_values(["contrast", "abs_effect"], ascending=[True, False])
    eff.to_csv(out / "tables" / "table2_feature_effects.csv", index=False)
    fig_forest(eff, out / "figures" / "figS1_forest_all")
    fig_forest(eff, out / "figures" / "fig1_forest", subset=COMPACT)

    # ---- 2. task efficiency
    sets = protocol_feature_sets(df)
    trecs = []
    curves = {}
    for contrast, mask in [("HC vs ALS-ND", df.group.isin(["HC", "ALS-ND"])),
                           ("HC vs all ALS", df.group.isin(["HC", "ALS-ND", "ALS-D"])),
                           ("HC vs ALS-D", df.group.isin(["HC", "ALS-D"]))]:
        sub = df[mask]
        y = sub["is_als"].to_numpy()
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < args.folds:
            continue
        for name, fl in sets.items():
            X = sub[fl + ["age_num", "sex_m"]].to_numpy(dtype=float)
            r = cv_auc(X, y, args.folds, args.repeats, args.seed, args.C)
            lo, hi = pooled_auc_ci(y, r["oof"], args.boot, rng)
            trecs.append({"contrast": contrast, "protocol": name, "n_features": len(fl), "n": len(y),
                          "auc_mean_repeats": r["auc_mean"], "auc_sd_repeats": r["auc_sd"],
                          "auc_pooled": r["auc_pooled"], "ci_lo": lo, "ci_hi": hi, "brier": r["brier"]})
            if contrast == "HC vs ALS-ND" and name in ("3 DDK: syllable F0 SD only", "/ka/ only", "Phonation only (all tasks)", "All 8 tasks"):
                curves[name] = (y, r["oof"])
            print(f"  {contrast:14s} {name:18s} AUC={r['auc_pooled']:.3f} [{lo:.3f},{hi:.3f}] Brier={r['brier']:.3f}")
    task_df = pd.DataFrame(trecs)
    task_df.to_csv(out / "tables" / "table3_task_efficiency.csv", index=False)
    mfcc_path = out / "tables" / "table3b_mfcc_reference.csv"
    mfcc_df = pd.read_csv(mfcc_path) if mfcc_path.exists() else None
    if len(task_df):
        fig_tasks(task_df, out / "figures" / "fig2_tasks", mfcc_df)
    if curves:
        fig_roc_calib(curves, out / "figures" / "fig3_roc_calibration")
        pd.concat([pd.DataFrame({"protocol": k, "y": y, "p": p}) for k, (y, p) in curves.items()]).to_csv(
            out / "tables" / "oof_predictions.csv", index=False)

    # ---- 3. clinical anchoring within ALS
    if "pumns_bulbar" in cols:  # one released value (7) exceeds the 0-4 range of the PUMNS bulbar sub-score: treated as missing
        pu_all = pd.to_numeric(df[cols["pumns_bulbar"]], errors="coerce")
        df[cols["pumns_bulbar"]] = pu_all.where(pu_all <= 4)
    als = df[df.is_als == 1]
    clin_vars = {k: cols[k] for k in ["pumns_bulbar", "mrc_head", "salivation", "swallowing", "fvc", "alsfrs_total",
                                       "duration", "progression", "dyspnea", "orthopnea", "breathing"] if k in cols}
    if "speech" in cols and "salivation" in cols and "swallowing" in cols:
        als = als.assign(bulbar_sum=pd.to_numeric(als[cols["speech"]], errors="coerce")
                         + pd.to_numeric(als[cols["salivation"]], errors="coerce")
                         + pd.to_numeric(als[cols["swallowing"]], errors="coerce"))
        clin_vars["bulbar_sum"] = "bulbar_sum"
    crecs = []
    for subset_name, subset in [("all ALS", als), ("ALS-ND", als[als.group == "ALS-ND"])]:
        for ck, cc in clin_vars.items():
            cv = pd.to_numeric(subset[cc], errors="coerce").to_numpy(float)
            for f in feats:
                fv = subset[f].to_numpy(float)
                ok = np.isfinite(cv) & np.isfinite(fv)
                if ok.sum() < 10 or len(np.unique(cv[ok])) < 3:
                    continue
                rho, p = stats.spearmanr(fv[ok], cv[ok])
                crecs.append({"subset": subset_name, "clinical": ck, "feature": f, "label": pretty(f),
                              "track": track_of[f], "n": int(ok.sum()), "rho": float(rho), "p": float(p)})
    clin = pd.DataFrame(crecs)
    if len(clin):
        for (s, c), g in clin.groupby(["subset", "clinical"]):
            clin.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
        clin.sort_values(["subset", "clinical", "p"]).to_csv(out / "tables" / "table4_clinical_correlations.csv", index=False)
    # bulbar vs spinal onset (within the subset where both exist; in VOC-ALS all ALS-ND are spinal onset)
    if "onset" in cols:
        for subset_name, subset in [("ALS-ND", als[als.group == "ALS-ND"]), ("ALS-D", als[als.group == "ALS-D"]), ("all ALS", als)]:
            onset = subset[cols["onset"]].astype(str).str.lower()
            bul, spi = subset[onset.str.contains("bulb")], subset[onset.str.contains("spin")]
            if len(bul) < 4 or len(spi) < 4:
                print(f"  onset contrast skipped for {subset_name}: bulbar={len(bul)}, spinal={len(spi)}")
                continue
            orecs = []
            for f in feats:
                auc, p = auc_mwu(spi[f].to_numpy(float), bul[f].to_numpy(float))
                orecs.append({"subset": subset_name, "feature": f, "label": pretty(f), "track": track_of[f],
                              "n_spinal": len(spi), "n_bulbar": len(bul), "auc_bulbar_gt_spinal": auc, "p": p})
            onset_df = pd.DataFrame(orecs)
            onset_df["q"] = bh_fdr(onset_df["p"].to_numpy())
            onset_df.sort_values("p").to_csv(out / "tables" / f"table5_onset_{subset_name.replace(' ', '_')}.csv", index=False)
    # spinal-onset ALS-D vs ALS-ND: does dysarthria rating track acoustic change within spinal onset?
    if "onset" in cols:
        onset_all = als[cols["onset"]].astype(str).str.lower()
        spi_d = als[(als.group == "ALS-D") & onset_all.str.contains("spin")]
        if len(spi_d) >= 5:
            srecs2 = []
            for f in feats:
                auc, p = auc_mwu(hc[f].to_numpy(float), spi_d[f].to_numpy(float))
                srecs2.append({"contrast": "HC vs spinal-onset ALS-D", "feature": f, "label": pretty(f),
                               "track": track_of[f], "n_hc": len(hc), "n_grp": len(spi_d), "auc": auc, "p": p})
            sd = pd.DataFrame(srecs2)
            sd["q"] = bh_fdr(sd["p"].to_numpy())
            sd.sort_values("p").to_csv(out / "tables" / "table5b_HC_vs_spinal_ALS-D.csv", index=False)

    # ---- 4. severity gradient
    srecs = []
    for f in feats:
        fv, sv = df[f].to_numpy(float), df["severity"].to_numpy(float)
        ok = np.isfinite(fv) & np.isfinite(sv)
        rho, p = stats.spearmanr(fv[ok], sv[ok])
        med = df[ok].groupby("severity")[f].median().to_dict()
        srecs.append({"feature": f, "label": pretty(f), "track": track_of[f], "rho_severity": float(rho), "p": float(p),
                      **{f"median_sev{int(k)}": v for k, v in med.items()}})
    sev = pd.DataFrame(srecs)
    sev["q"] = bh_fdr(sev["p"].to_numpy())
    sev.sort_values("p").to_csv(out / "tables" / "table6_severity_gradient.csv", index=False)

    # ---- summary
    summary = {"groups": counts, "clinical_columns": cols, "n_features": len(feats),
               "top_subclinical": eff[eff.contrast == "HC vs ALS-ND"].head(8)[["label", "auc", "ci_lo", "ci_hi", "q"]].round(3).to_dict("records"),
               "tasks": task_df[task_df.contrast == "HC vs ALS-ND"][["protocol", "auc_pooled", "ci_lo", "ci_hi", "brier"]].round(3).to_dict("records") if len(task_df) else []}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\nTop subclinical features (HC vs ALS-ND):")
    print(eff[eff.contrast == "HC vs ALS-ND"].head(10)[["label", "track", "auc", "ci_lo", "ci_hi", "q", "auc_adj"]].round(3).to_string(index=False))
    print(f"\nWrote tables and figures to {out}")


if __name__ == "__main__":
    main()
