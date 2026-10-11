#!/usr/bin/env python3
"""
Extended analyses (signal processing + neurology-inspired):
  E1  time-scale decomposition of laryngeal instability (vowel modulation bands; DDK micro vs macro)
  E2  voice re-initiation dynamics in the DDK task
  E5  upper- vs lower-motor-neuron signature: partial Spearman correlations with PUMNS bulbar (UMN)
      controlling MRC head (LMN) and vice versa, within patients
  E6  progression: association with the ALSFRS-R progression rate, fast vs slow progressors

Usage: python analyze_extended.py --features results/features.csv --extended results/extended.csv --out results/analysis
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
from analyze_subclinical import (auc_mwu, auc_boot_ci, bh_fdr, residualise, resolve_columns, OKABE, pretty)  # noqa: E402

# pre-specified extended measures (families for FDR)
E1_VOWEL = {
    "V__f0_b1_wander_abs": "F0 power 0.25-1 Hz (wander)", "V__f0_b2_slow_abs": "F0 power 1-4 Hz",
    "V__f0_b3_tremor_abs": "F0 power 4-8 Hz (tremor)", "V__f0_b4_flutter_abs": "F0 power 8-15 Hz (flutter)",
    "V__f0_b1_wander_rel": "F0 share 0.25-1 Hz", "V__f0_b3_tremor_rel": "F0 share 4-8 Hz",
    "V__f0_sd_st_fold": "F0 SD, octave-folded", "V__f0_oct_frac": "Octave-level frames (vowel)",
    "V__int_b1_wander_abs": "Intensity power 0.25-1 Hz", "V__int_b3_tremor_abs": "Intensity power 4-8 Hz",
}
E1_DDK = {
    "DDK__macro_f0_sd_st": "Across-syllable F0 SD (folded)", "DDK__macro_f0_sd_detr_st": "Across-syllable F0 SD (detrended)",
    "DDK__macro_f0_lag1_ac": "Across-syllable lag-1 autocorr.", "DDK__micro_f0_sd_st": "Within-nucleus F0 SD",
    "DDK__macro_oct_frac": "Octave-level nuclei (share)",
}
E2 = {
    "DDK__onset_dev_sd": "F0 at voicing onset: SD", "DDK__onset_slope_sd": "F0 onset slope SD",
    "DDK__onset_dev_mean": "F0 onset deviation, mean |st|", "DDK__rise_time_mean": "Amplitude rise time",
    "DDK__rise_time_sd": "Amplitude rise time SD", "DDK__vot_mean": "Burst-to-voicing delay",
    "DDK__vot_sd": "VOT proxy SD",
}
FAMILIES = {"E1 vowel time-scale": E1_VOWEL, "E1 DDK micro/macro": E1_DDK, "E2 re-initiation": E2}

# measures for the UMN/LMN and progression analyses: all 39 main + 22 extended (labels for the key ones)
PATHWAY_LABELS = {
    "DDK__syl_f0_sd_st": "DDK syllable F0 SD", "V__f0_sd_st_fold": "Vowel F0 SD (folded)", "DDK__macro_oct_frac": "Octave-level nuclei",
    "V__jitter_ppq5": "Jitter (PPQ5)", "V__shimmer_apq11": "Shimmer (APQ11)", "V__hnr": "HNR", "V__cpps": "CPPS",
    "V__alpha_ratio_db": "Spectral balance", "V__f0_b3_tremor_abs": "F0 tremor power", "DDK__onset_dev_sd": "F0 onset dev. SD",
    "DDK__rise_time_mean": "Amplitude rise time", "DDK__vot_mean": "VOT proxy", "DDK__ddk_rate": "DDK rate",
    "DDK__ioi_cv": "DDK interval CV", "ART__fcr": "FCR", "ART__vsa": "Vowel space area", "DDK__voiced_frac": "DDK voiced fraction",
    "DDK__syll_per_run": "Syllables per run", "V__mpt_proxy_max": "Longest phonation", "V__int_slope": "Intensity slope",
    "V__f0_mean": "Mean F0",
}


def pathway_measures(df):
    from analyze_subclinical import TRACKS
    main = [f for fs in TRACKS.values() for f in fs if f in df.columns]
    ext = [f for fam in FAMILIES.values() for f in fam if f in df.columns]
    out = {}
    for f in main + ext:
        if f not in out:
            out[f] = PATHWAY_LABELS.get(f, pretty(f) if f in main else {**E1_VOWEL, **E1_DDK, **E2}.get(f, f))
    return out


def partial_spearman(x, y, z):
    """Spearman correlation between x and y controlling for z (rank-based, residual method)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    if ok.sum() < 12:
        return np.nan, np.nan, int(ok.sum())
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    rz = np.column_stack([stats.rankdata(z[ok][:, j]) for j in range(z.shape[1])])
    X = np.column_stack([np.ones(ok.sum()), rz])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p), int(ok.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--extended", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = args.out
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features).merge(pd.read_csv(args.extended), on="subject", how="left")
    cols = resolve_columns(df)
    cat = df[cols["category"]].astype(str).str.upper()
    speech = pd.to_numeric(df[cols["speech"]], errors="coerce")
    df["is_als"] = cat.str.startswith("ALS").astype(int)
    df["group"] = np.where(df.is_als == 0, "HC", np.where(speech >= 4, "ALS-ND", "ALS-D"))
    df["sex_m"] = (df[cols["sex"]].astype(str).str.upper().str[0] == "M").astype(float)
    df["age_num"] = pd.to_numeric(df[cols["age"]], errors="coerce")
    pu_all = pd.to_numeric(df[cols["pumns_bulbar"]], errors="coerce")
    df[cols["pumns_bulbar"]] = pu_all.where(pu_all <= 4)  # one released value (7) exceeds the 0-4 range: treated as missing
    hc, nd, dys = df[df.group == "HC"], df[df.group == "ALS-ND"], df[df.group == "ALS-D"]

    # ---------------- E1/E2: effect sizes with FDR per family
    recs = []
    for fam, meas in FAMILIES.items():
        for contrast, grp in [("HC vs ALS-ND", nd), ("HC vs ALS-D", dys)]:
            both = pd.concat([hc, grp])
            covars = both[["age_num", "sex_m"]].to_numpy(float)
            is_grp = (both.group != "HC").to_numpy()
            for f, lab in meas.items():
                if f not in df.columns:
                    continue
                a, b = hc[f].to_numpy(float), grp[f].to_numpy(float)
                auc, p = auc_mwu(a, b)
                lo, hi = auc_boot_ci(a, b, args.boot, rng)
                res = residualise(both[f].to_numpy(float), covars)
                auc_adj, p_adj = auc_mwu(res[~is_grp], res[is_grp])
                recs.append({"family": fam, "contrast": contrast, "feature": f, "label": lab,
                             "n_hc": int(np.isfinite(a).sum()), "n_grp": int(np.isfinite(b).sum()),
                             "median_hc": float(np.nanmedian(a)), "median_grp": float(np.nanmedian(b)),
                             "auc": auc, "ci_lo": lo, "ci_hi": hi, "p": p, "auc_adj": auc_adj, "p_adj": p_adj})
    eff = pd.DataFrame(recs)
    for c, g in eff.groupby("contrast"):  # one family: all 22 extended measures
        eff.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
        eff.loc[g.index, "q_adj"] = bh_fdr(g["p_adj"].to_numpy())
    eff.to_csv(out / "tables" / "table7_extended_effects.csv", index=False)
    print(eff[eff.contrast == "HC vs ALS-ND"][["family", "label", "auc", "ci_lo", "ci_hi", "q", "auc_adj", "q_adj"]].round(3).to_string(index=False))

    # ---------------- E5: UMN vs LMN partial correlations within ALS
    als = df[df.is_als == 1]
    pum = pd.to_numeric(als[cols["pumns_bulbar"]], errors="coerce").to_numpy(float)
    mrc = pd.to_numeric(als[cols["mrc_head"]], errors="coerce").to_numpy(float)
    age = als["age_num"].to_numpy(float)
    PATHWAY = pathway_measures(df)
    prec = []
    for f, lab in PATHWAY.items():
        if f not in als.columns:
            continue
        x = als[f].to_numpy(float)
        r_u, p_u, n = partial_spearman(x, pum, np.column_stack([mrc, age]))
        r_l, p_l, _ = partial_spearman(x, mrc, np.column_stack([pum, age]))
        ru0 = stats.spearmanr(x, pum, nan_policy="omit")
        rl0 = stats.spearmanr(x, mrc, nan_policy="omit")
        prec.append({"feature": f, "label": lab, "n": n, "rho_umn_partial": r_u, "p_umn": p_u, "rho_lmn_partial": r_l, "p_lmn": p_l,
                     "rho_umn_raw": float(ru0.statistic), "rho_lmn_raw": float(rl0.statistic)})
    path = pd.DataFrame(prec)
    path["q_umn"] = bh_fdr(path["p_umn"].to_numpy())
    path["q_lmn"] = bh_fdr(path["p_lmn"].to_numpy())
    path.to_csv(out / "tables" / "table8_umn_lmn_partial.csv", index=False)
    print("\nUMN (PUMNS bulbar | MRC head, age) vs LMN (MRC head | PUMNS, age) partial Spearman (top by |rho_lmn|):")
    print(path.reindex(path.rho_lmn_partial.abs().sort_values(ascending=False).index).head(14)[["label", "n", "rho_umn_partial", "q_umn", "rho_lmn_partial", "q_lmn"]].round(3).to_string(index=False))
    print("PUMNS vs MRC themselves: rho=%.2f" % stats.spearmanr(pum, mrc, nan_policy="omit").statistic)

    # ---------------- E6: progression
    prog = pd.to_numeric(als[cols["progression"]], errors="coerce").to_numpy(float)
    dur = pd.to_numeric(als[cols["duration"]], errors="coerce").to_numpy(float)
    precs = []
    for subset_name, mask in [("ALS-ND", (als.group == "ALS-ND").to_numpy()), ("all ALS", np.ones(len(als), bool))]:
        pr = prog[mask]
        fast = pr >= 0.5
        for f, lab in PATHWAY.items():
            if f not in als.columns:
                continue
            x = als[f].to_numpy(float)[mask]
            ok = np.isfinite(x) & np.isfinite(pr)
            if ok.sum() < 12:
                continue
            rho, p = stats.spearmanr(x[ok], pr[ok])
            # partial for disease duration (progression rate = (48-ALSFRS)/duration is duration-dependent)
            rho_d, p_d, _ = partial_spearman(x, pr, dur[mask][:, None])
            a_fast, p_fast = auc_mwu(x[ok & ~fast], x[ok & fast])
            precs.append({"subset": subset_name, "feature": f, "label": lab, "n": int(ok.sum()), "n_fast": int((ok & fast).sum()),
                          "rho_progression": float(rho), "p": float(p), "rho_partial_duration": rho_d, "p_partial": p_d,
                          "auc_fast_vs_slow": a_fast, "p_fast": p_fast})
    pg = pd.DataFrame(precs)
    for s_, g in pg.groupby("subset"):
        pg.loc[g.index, "q"] = bh_fdr(g["p"].to_numpy())
        pg.loc[g.index, "q_fast"] = bh_fdr(g["p_fast"].to_numpy())
    pg.to_csv(out / "tables" / "table9_progression.csv", index=False)
    print("\nProgression (ALS-ND):")
    print(pg[pg.subset == "ALS-ND"].sort_values("p")[["label", "n", "n_fast", "rho_progression", "q", "rho_partial_duration", "auc_fast_vs_slow", "q_fast"]].round(3).head(12).to_string(index=False))
    print("\nProgression (all ALS):")
    print(pg[pg.subset == "all ALS"].sort_values("p")[["label", "n", "n_fast", "rho_progression", "q", "rho_partial_duration", "auc_fast_vs_slow", "q_fast"]].round(3).head(12).to_string(index=False))

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 4: time-scale decomposition + re-initiation (AUC with CI, ND filled / D hollow)
    rows = [("Vowel: F0 modulation and folded SD", ["V__f0_b1_wander_abs", "V__f0_b2_slow_abs", "V__f0_b3_tremor_abs", "V__f0_b4_flutter_abs", "V__f0_sd_st_fold"]),
            ("DDK: within vs across syllables", ["DDK__micro_f0_sd_st", "DDK__macro_f0_sd_st", "DDK__macro_f0_sd_detr_st", "DDK__macro_f0_lag1_ac", "DDK__macro_oct_frac"]),
            ("DDK: voice re-initiation", ["DDK__onset_dev_sd", "DDK__vot_mean", "DDK__rise_time_mean"])]
    labels_all = {**E1_VOWEL, **E1_DDK, **E2}
    n = sum(len(r[1]) for r in rows)
    fig, ax = plt.subplots(figsize=(3.4, 0.135 * n + 0.85))
    y = 0
    ticks = []
    e_nd = eff[eff.contrast == "HC vs ALS-ND"].set_index("feature")
    e_d = eff[eff.contrast == "HC vs ALS-D"].set_index("feature")
    titles = []
    for title, fl in rows:
        y -= 0.5
        ticks.append((y, title))
        titles.append(title)
        y -= 1.0
        for f in fl:
            if f not in e_nd.index:
                continue
            r, rd = e_nd.loc[f], e_d.loc[f]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=OKABE["blue"], lw=1.3)
            ax.plot(r.auc, y, "o", color=OKABE["blue"], ms=4, zorder=3)
            ax.plot([rd.ci_lo, rd.ci_hi], [y - 0.35, y - 0.35], color=OKABE["orange"], lw=1.0)
            ax.plot(rd.auc, y - 0.35, "s", mfc="white", mec=OKABE["orange"], ms=3.8, zorder=3)
            mark = ("*" if r.q < 0.05 else "") + ("\u2020" if r.q_adj < 0.05 else "")
            ticks.append((y - 0.17, labels_all[f] + mark))
            y -= 1.1
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yticks([p for p, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=6.5)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in titles:
            lbl.set_fontweight("bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(y, 0.4)
    ax.set_xlabel("AUC (0.5 = no difference)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.plot([], [], "o-", color=OKABE["blue"], label="HC vs ALS-ND")
    ax.plot([], [], "s-", color=OKABE["orange"], mfc="white", label="HC vs ALS-D")
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.35, 1.0), frameon=False, ncol=2, handlelength=1.5, columnspacing=1.0)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig4_timescale.pdf")
    fig.savefig(out / "figures" / "fig4_timescale.png", dpi=200)
    plt.close(fig)

    # Fig 5: UMN vs LMN partial-correlation map
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    for _, r in path.iterrows():
        col = OKABE["blue"] if (r.q_umn < 0.05 or r.q_lmn < 0.05) else OKABE["grey"]
        ax.plot(r.rho_umn_partial, r.rho_lmn_partial, "o", color=col, ms=4)
        ax.annotate(r.label, (r.rho_umn_partial, r.rho_lmn_partial), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("partial $\\rho$ with UMN burden (PUMNS bulbar | MRC, age)", fontsize=7)
    ax.set_ylabel("partial $\\rho$ with LMN strength (MRC head | PUMNS, age)", fontsize=7)
    ax.tick_params(labelsize=7)
    lim = max(0.45, float(np.nanmax(np.abs(path[["rho_umn_partial", "rho_lmn_partial"]].to_numpy()))) + 0.05)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig5_umn_lmn.pdf")
    fig.savefig(out / "figures" / "fig5_umn_lmn.png", dpi=200)
    plt.close(fig)

    summary = {"n_extended_measures": int(eff[eff.contrast == "HC vs ALS-ND"].shape[0]),
               "n_sig_q_nd": int((eff[eff.contrast == "HC vs ALS-ND"].q < 0.05).sum()),
               "n_sig_qadj_nd": int((eff[eff.contrast == "HC vs ALS-ND"].q_adj < 0.05).sum())}
    (out / "summary_extended.json").write_text(json.dumps(summary, indent=1))
    print("\n", summary)


if __name__ == "__main__":
    main()
