#!/usr/bin/env python3
"""Generate LaTeX table fragments and a numbers.tex macro file from results/analysis, so that every
number in the paper is traceable to the analysis output.

Usage: python make_paper_tables.py --features results/features.csv --analysis results/analysis --out paper
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent))


def fmt(x, nd=2):
    return "--" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"


def macro(name, value):
    name = re.sub(r"[^A-Za-z]", "", name)
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    T = a.analysis / "tables"
    (a.out / "tables").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.features)
    sp = pd.to_numeric(df["ALSFRS-R_SpeechSubscore"], errors="coerce")
    df["group"] = np.where(df["Category"].str.upper() == "HC", "HC", np.where(sp >= 4, "ALS-ND", "ALS-D"))
    eff = pd.read_csv(T / "table2_feature_effects.csv")
    task = pd.read_csv(T / "table3_task_efficiency.csv")
    mfcc = pd.read_csv(T / "table3b_mfcc_reference.csv")
    clin = pd.read_csv(T / "table4_clinical_correlations.csv")

    # ------------------------------------------------------------------ numbers.tex
    nums = ""
    g = df.groupby("group")
    nums += macro("nHC", int((df.group == "HC").sum()))
    nums += macro("nND", int((df.group == "ALS-ND").sum()))
    nums += macro("nD", int((df.group == "ALS-D").sum()))
    nums += macro("nFeat", int(eff[eff.contrast == "HC vs ALS-ND"].shape[0]))
    nd = eff[eff.contrast == "HC vs ALS-ND"].set_index("feature")
    dd = eff[eff.contrast == "HC vs ALS-D"].set_index("feature")
    top = nd.loc["DDK__syl_f0_sd_st"]
    nums += macro("topAUC", fmt(top.auc)) + macro("topLo", fmt(top.ci_lo)) + macro("topHi", fmt(top.ci_hi))
    nums += macro("topQ", fmt(top.q, 3)) + macro("topAdj", fmt(top.auc_adj)) + macro("topQadj", fmt(top.q_adj, 3))
    nums += macro("topMedHC", fmt(top.median_hc)) + macro("topMedND", fmt(top.median_grp)) + macro("topMedD", fmt(dd.loc["DDK__syl_f0_sd_st"].median_grp))
    nums += macro("nSigRaw", int((nd.q < 0.05).sum())) + macro("nSigAdj", int((nd.q_adj < 0.05).sum()))
    for key, col in [("ddkF", "DDK__f0_sd_st_all"), ("jitP", "V__jitter_ppq5"), ("vbreak", "V__unvoiced_frac_run"),
                     ("ddkrate", "DDK__ddk_rate"), ("ioicv", "DDK__ioi_cv"), ("vsa", "ART__vsa"), ("fcr", "ART__fcr"),
                     ("sylrun", "DDK__syll_per_run"), ("mpt", "V__mpt_proxy_max"), ("intslope", "V__int_slope"),
                     ("vowF", "V__f0_sd_st_all"), ("hnr", "V__hnr"), ("cpps", "V__cpps")]:
        r = nd.loc[col]
        nums += macro(key + "AUC", fmt(r.auc)) + macro(key + "Lo", fmt(r.ci_lo)) + macro(key + "Hi", fmt(r.ci_hi))
        nums += macro(key + "Adj", fmt(r.auc_adj)) + macro(key + "Qadj", fmt(r.q_adj, 3)) + macro(key + "Q", fmt(r.q, 3))
        rd = dd.loc[col]
        nums += macro(key + "DAUC", fmt(rd.auc)) + macro(key + "DLo", fmt(rd.ci_lo)) + macro(key + "DHi", fmt(rd.ci_hi))
    # task efficiency
    tnd = task[task.contrast == "HC vs ALS-ND"].set_index("protocol")
    td = task[task.contrast == "HC vs ALS-D"].set_index("protocol")
    tm = mfcc[mfcc.contrast == "HC vs ALS-ND"].set_index("protocol")
    tmd = mfcc[mfcc.contrast == "HC vs ALS-D"].set_index("protocol")
    for key, prot in [("ka", "/ka/ only"), ("pa", "/pa/ only"), ("ta", "/ta/ only"), ("a", "/a/ only"), ("i", "/i/ only"),
                      ("o", "/o/ only"), ("u", "/u/ only"), ("vow", "5 vowels (mean)"), ("ddk", "3 DDK (mean)"),
                      ("all", "All 8 tasks"), ("phon", "Phonation only (all tasks)"), ("one", "/pa/: syllable F0 SD only"),
                      ("taone", "/ta/: syllable F0 SD only"), ("kaone", "/ka/: syllable F0 SD only"), ("ddkone", "3 DDK: syllable F0 SD only")]:
        r = tnd.loc[prot]
        nums += macro("t" + key + "AUC", fmt(r.auc_pooled)) + macro("t" + key + "Lo", fmt(r.ci_lo)) + macro("t" + key + "Hi", fmt(r.ci_hi)) + macro("t" + key + "Brier", fmt(r.brier))
        r = td.loc[prot]
        nums += macro("t" + key + "DAUC", fmt(r.auc_pooled)) + macro("t" + key + "DLo", fmt(r.ci_lo)) + macro("t" + key + "DHi", fmt(r.ci_hi))
        if prot in tm.index:
            r = tm.loc[prot]
            nums += macro("m" + key + "AUC", fmt(r.auc_pooled)) + macro("m" + key + "Lo", fmt(r.ci_lo)) + macro("m" + key + "Hi", fmt(r.ci_hi))
            r = tmd.loc[prot]
            nums += macro("m" + key + "DAUC", fmt(r.auc_pooled)) + macro("m" + key + "DLo", fmt(r.ci_lo)) + macro("m" + key + "DHi", fmt(r.ci_hi))
    nums += macro("mBestAUC", fmt(tm.auc_pooled.max()))
    nums += macro("nRep", 20) + macro("CVal", "0.3")
    # clinical anchoring numbers
    def rho(subset, clinical, feature):
        r = clin[(clin.subset == subset) & (clin.clinical == clinical) & (clin.feature == feature)]
        return (float(r.rho.iloc[0]), float(r.q.iloc[0]), int(r.n.iloc[0])) if len(r) else (np.nan, np.nan, 0)
    for key, (s, c, f) in {"fvcSyl": ("all ALS", "fvc", "DDK__syll_per_run"), "mrcSyl": ("all ALS", "mrc_head", "DDK__syll_per_run"),
                           "bulSyl": ("all ALS", "bulbar_sum", "DDK__syll_per_run"), "pumnsSyl": ("all ALS", "pumns_bulbar", "DDK__syll_per_run"),
                           "ndDyspVb": ("ALS-ND", "dyspnea", "V__unvoiced_frac_run"), "ndDyspJit": ("ALS-ND", "dyspnea", "V__jitter_ppq5"),
                           "ndMrcSyl": ("ALS-ND", "mrc_head", "DDK__syll_per_run"), "ndMrcInt": ("ALS-ND", "mrc_head", "V__int_slope"),
                           "ndDurSyl": ("ALS-ND", "duration", "DDK__syll_per_run"),
                           "pumnsTrem": ("all ALS", "pumns_bulbar", "V__f0_tremor_idx"), "pumnsRate": ("all ALS", "pumns_bulbar", "DDK__ddk_rate"),
                           "fvcF": ("all ALS", "fvc", "DDK__f0_sd_st_all"), "bulF": ("all ALS", "bulbar_sum", "DDK__f0_sd_st_all"),
                           "ndallDysp": ("all ALS", "dyspnea", "DDK__syl_f0_sd_st")}.items():
        r, q, n = rho(s, c, f)
        nums += macro(key + "Rho", fmt(r)) + macro(key + "Q", fmt(q, 3)) + macro(key + "N", n)
    # ---- extended analyses (E1/E2/E5/E6)
    ext_path = T / "table7_extended_effects.csv"
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        xnd = ext[ext.contrast == "HC vs ALS-ND"].set_index("feature")
        xd = ext[ext.contrast == "HC vs ALS-D"].set_index("feature")
        for key, col in [("eWander", "V__f0_b1_wander_abs"), ("eSlow", "V__f0_b2_slow_abs"), ("eTremor", "V__f0_b3_tremor_abs"),
                         ("eFlutter", "V__f0_b4_flutter_abs"), ("eWanderRel", "V__f0_b1_wander_rel"), ("eTremorRel", "V__f0_b3_tremor_rel"),
                         ("eFold", "V__f0_sd_st_fold"), ("eOctV", "V__f0_oct_frac"), ("eMacro", "DDK__macro_f0_sd_st"),
                         ("eMacroDetr", "DDK__macro_f0_sd_detr_st"), ("eLag", "DDK__macro_f0_lag1_ac"), ("eMicro", "DDK__micro_f0_sd_st"),
                         ("eOctD", "DDK__macro_oct_frac"), ("eOnset", "DDK__onset_dev_sd"), ("eVot", "DDK__vot_mean"),
                         ("eRise", "DDK__rise_time_mean"), ("eRiseSd", "DDK__rise_time_sd"), ("eIntWander", "V__int_b1_wander_abs")]:
            if col in xnd.index:
                r, rd = xnd.loc[col], xd.loc[col]
                nums += macro(key + "AUC", fmt(r.auc)) + macro(key + "Lo", fmt(r.ci_lo)) + macro(key + "Hi", fmt(r.ci_hi))
                nums += macro(key + "Q", fmt(r.q, 3)) + macro(key + "Adj", fmt(r.auc_adj)) + macro(key + "Qadj", fmt(r.q_adj, 3))
                nums += macro(key + "DAUC", fmt(rd.auc)) + macro(key + "MedHC", fmt(r.median_hc, 3)) + macro(key + "MedND", fmt(r.median_grp, 3))
        nums += macro("nExt", int(xnd.shape[0])) + macro("nExtSigRaw", int((xnd.q < 0.05).sum())) + macro("nExtSigAdj", int((xnd.q_adj < 0.05).sum()))
    pth = T / "table8_umn_lmn_partial.csv"
    if pth.exists():
        pw = pd.read_csv(pth).set_index("feature")
        for key, col in [("pSyl", "DDK__syll_per_run"), ("pMpt", "V__mpt_proxy_max"), ("pInt", "V__int_slope"), ("pRate", "DDK__ddk_rate"),
                         ("pRise", "DDK__rise_time_mean"), ("pFold", "V__f0_sd_st_fold"), ("pTrem", "V__f0_b3_tremor_abs"), ("pSylF", "DDK__syl_f0_sd_st")]:
            if col in pw.index:
                r = pw.loc[col]
                nums += macro(key + "Umn", fmt(r.rho_umn_partial)) + macro(key + "UmnQ", fmt(r.q_umn, 2)) + macro(key + "Lmn", fmt(r.rho_lmn_partial)) + macro(key + "LmnQ", fmt(r.q_lmn, 3))
        nums += macro("pUmnMinQ", fmt(pw.q_umn.min(), 2)) + macro("pUmnMaxAbs", fmt(pw.rho_umn_partial.abs().max())) + macro("pNLmnSig", int((pw.q_lmn < 0.05).sum())) + macro("pN", int(pw.shape[0]))
    pgp = T / "table9_progression.csv"
    if pgp.exists():
        pg = pd.read_csv(pgp)
        nd_ = pg[pg.subset == "ALS-ND"]
        al_ = pg[pg.subset == "all ALS"].set_index("feature")
        nums += macro("gNdMinQ", fmt(nd_.q.min(), 2)) + macro("gNdAucLo", fmt(nd_.auc_fast_vs_slow.min())) + macro("gNdAucHi", fmt(nd_.auc_fast_vs_slow.max())) + macro("gNdNfast", int(nd_.n_fast.max())) + macro("gNdN", int(nd_.n.max()))
        for key, col in [("gInt", "V__int_slope"), ("gMpt", "V__mpt_proxy_max"), ("gFold", "V__f0_sd_st_fold")]:
            if col in al_.index:
                r = al_.loc[col]
                nums += macro(key + "Rho", fmt(r.rho_progression)) + macro(key + "Q", fmt(r.q, 3)) + macro(key + "RhoDur", fmt(r.rho_partial_duration)) + macro(key + "AucFast", fmt(r.auc_fast_vs_slow)) + macro(key + "QFast", fmt(r.q_fast, 3))
    # ---- revision analyses (R1-R6)
    import json as _json
    r1p = T / "table10_subsystem_comparison.csv"
    if r1p.exists():
        r1 = pd.read_csv(r1p)
        def row(prefix):
            m = r1[r1.comparison.str.startswith(prefix)]
            return m.iloc[0] if len(m) else None
        for key, prefix in [("cPhon", "composite: Phonation ("), ("cArt", "composite: Articulation ("), ("cResp", "composite: Respiratory support (")]:
            r = row(prefix)
            if r is not None:
                nums += macro(key + "AUC", fmt(r.auc1)) + macro(key + "Lo", fmt(r.ci_lo)) + macro(key + "Hi", fmt(r.ci_hi))
        for key, prefix in [("dPhonArt", "composite: Phonation vs Articulation"), ("dPhonResp", "composite: Phonation vs Respiratory support")]:
            r = row(prefix)
            if r is not None:
                nums += macro(key + "Diff", fmt(r["diff"])) + macro(key + "Lo", fmt(r.ci_lo)) + macro(key + "Hi", fmt(r.ci_hi)) + macro(key + "P", fmt(r.p_boot_le0, 3))
        singles = r1[r1.comparison.str.startswith("single:")]
        for i, (_, r) in enumerate(singles.iterrows()):
            key = ["sPhonArt", "sPhonResp"][i]
            nums += macro(key + "Diff", fmt(r["diff"])) + macro(key + "Lo", fmt(r.ci_lo)) + macro(key + "Hi", fmt(r.ci_hi)) + macro(key + "AUCtwo", fmt(r.auc2))
    r2p = T / "table11_sensitivity_bulbar_normal.csv"
    if r2p.exists():
        r2 = pd.read_csv(r2p)
        k = r2[r2.feature == "DDK__syl_f0_sd_st"].set_index("subset")
        for key, sname in [("sensA", "ALS-ND (speech item = 4)"), ("sensB", "+ salivation = swallowing = 4"), ("sensC", "+ MRC head = 15"), ("sensD", "+ PUMNS bulbar = 0")]:
            if sname in k.index:
                nums += macro(key + "AUC", fmt(k.loc[sname, "auc"])) + macro(key + "N", int(k.loc[sname, "n"]))
                if "ci_lo" in k.columns:
                    nums += macro(key + "Lo", fmt(k.loc[sname, "ci_lo"])) + macro(key + "Hi", fmt(k.loc[sname, "ci_hi"])) + macro(key + "P", fmt(k.loc[sname, "p"], 3))
        k2 = r2[r2.feature == "DDK__macro_oct_frac"].set_index("subset")
        for key, sname in [("sensOctB", "+ salivation = swallowing = 4"), ("sensOctC", "+ MRC head = 15"), ("sensOctD", "+ PUMNS bulbar = 0")]:
            if sname in k2.index:
                nums += macro(key + "AUC", fmt(k2.loc[sname, "auc"]))
                if "ci_lo" in k2.columns:
                    nums += macro(key + "Lo", fmt(k2.loc[sname, "ci_lo"])) + macro(key + "Hi", fmt(k2.loc[sname, "ci_hi"]))
        k3 = r2[r2.feature == "V__f0_sd_st_fold"].set_index("subset")
        for key, sname in [("sensVowB", "+ salivation = swallowing = 4"), ("sensVowC", "+ MRC head = 15"), ("sensVowD", "+ PUMNS bulbar = 0")]:
            if sname in k3.index:
                nums += macro(key + "AUC", fmt(k3.loc[sname, "auc"]))
    r3p = T / "table12_tracker_robustness.csv"
    if r3p.exists():
        r3 = pd.read_csv(r3p)
        for key, meas in [("trk", "syl_sd"), ("trkFold", "syl_sd_fold"), ("trkOct", "oct_frac")]:
            m = r3[(r3.measure == meas) & (~r3.tracker.isin(["cc60", "reaper"]))]  # alternative Praat settings only
            nums += macro(key + "Min", fmt(m.auc.min())) + macro(key + "Max", fmt(m.auc.max())) + macro(key + "RhoMin", fmt(m.rho_vs_main.min())) + macro(key + "RhoMax", fmt(m.rho_vs_main.max()))
            if "rho_same_measure" in m.columns:
                nums += macro(key + "SameMin", fmt(m.rho_same_measure.min())) + macro(key + "SameMax", fmt(m.rho_same_measure.max()))
            rp = r3[(r3.measure == meas) & (r3.tracker == "reaper")]
            if len(rp):
                r = rp.iloc[0]
                nums += macro(key + "Rpr", fmt(r.auc)) + macro(key + "RprP", fmt(r.p, 3)) + macro(key + "RprRho", fmt(r.rho_vs_main))
                if "rho_same_measure" in rp.columns:
                    nums += macro(key + "RprSame", fmt(r.rho_same_measure))
    sj = a.analysis / "summary_revision.json"
    if sj.exists():
        S = _json.loads(sj.read_text())
        shr = S.get("R3_shr", {})
        if shr:
            nums += macro("shrFlagged", int(shr["flagged_nuclei_total"])) + macro("shrGenuine", int(shr["genuine_by_SHR"])) + macro("shrGenuinePct", f"{100 * shr['genuine_by_SHR'] / shr['flagged_nuclei_total']:.0f}")
            nums += macro("shrFlagMed", fmt(shr["shr_flag_median_db"], 1)) + macro("shrCtrlMed", fmt(shr["shr_ctrl_median_db"], 1))
            nums += macro("shrGenAUC", fmt(shr["auc_oct_genuine_frac"])) + macro("shrGenP", fmt(shr["p_oct_genuine_frac"], 4)) + macro("shrSpurAUC", fmt(shr["auc_oct_spurious_frac"])) + macro("shrSpurP", fmt(shr["p_oct_spurious_frac"], 3))
            if "median_nd_oct_frac" in shr:
                nums += macro("octMedNDPct", f"{100 * shr['median_nd_oct_frac']:.1f}")
            if "ctrl_fp_rate" in shr:
                nums += macro("shrCtrlFP", f"{100 * shr['ctrl_fp_rate']:.0f}") + macro("shrCtrlN", int(shr["ctrl_nuclei_tested"])) + macro("shrCtrlAbove", int(shr["ctrl_above_threshold"]))
            if "flagged_down" in shr:
                nums += macro("shrDown", int(shr["flagged_down"])) + macro("shrUp", int(shr["flagged_up"])) + macro("shrDownGenuine", int(shr["flagged_down_genuine"]))
                nums += macro("shrDownPct", f"{100 * shr['flagged_down_genuine'] / max(shr['flagged_down'], 1):.0f}") + macro("shrDownMed", fmt(shr["shr_down_median_db"], 1))
                nums += macro("shrDownGenAUC", fmt(shr["auc_oct_down_genuine_frac"])) + macro("shrDownGenP", fmt(shr["p_oct_down_genuine_frac"], 4)) + macro("shrDownSpurAUC", fmt(shr["auc_oct_down_spurious_frac"])) + macro("shrUpAUC", fmt(shr["auc_oct_up_frac"]))
        r4 = S.get("R4_marker_adjusted", {})
        if r4:
            pv = r4["p_adj_full"]
            nums += macro("covAdjAUC", fmt(r4["auc_adj_age_sex_snr_level_rate_nsyll"])) + macro("covAdjP", ("<10^{-4}" if pv < 1e-4 else fmt(pv, 4)))
        r5 = S.get("R5_repeatability", {})
        if r5:
            nums += macro("iccAll", fmt(r5["icc21_across_syllables_all"])) + macro("iccHC", fmt(r5["icc21_across_syllables_HC"])) + macro("iccN", int(r5["n_subjects"]))
            sb = [v["spearman_brown"] for v in r5["split_half"].values()]; sh = [v["spearman_odd_even"] for v in r5["split_half"].values()]
            nums += macro("shMin", fmt(min(sh))) + macro("shMax", fmt(max(sh))) + macro("sbMin", fmt(min(sb))) + macro("sbMax", fmt(max(sb)))
            bt = r5["spearman_between_tasks"]; nums += macro("btMin", fmt(min(bt.values()))) + macro("btMax", fmt(max(bt.values())))
    r4p = T / "table13_covariates.csv"
    if r4p.exists():
        r4t = pd.read_csv(r4p).set_index("covariate")
        for key, c in [("covSnr", "snr_db"), ("covLevel", "level_db"), ("covRate", "ddk_rate"), ("covNsyl", "n_syll"), ("covF0", "f0_mean_hz")]:
            if c in r4t.index:
                nums += macro(key + "AUC", fmt(r4t.loc[c, "auc_group_difference"])) + macro(key + "RhoHC", fmt(r4t.loc[c, "rho_with_marker_HC"])) + macro(key + "Rho", fmt(r4t.loc[c, "rho_with_marker_all"]))
    r6p = T / "table14_generic_reference.csv"
    if r6p.exists():
        r6 = pd.read_csv(r6p).set_index("protocol")
        for key, prot in [("gAll", "All 8 tasks (prosodic only)"), ("gI", "/i/ only (prosodic only)"), ("gA", "/a/ only (prosodic only)"), ("gPa", "/pa/ only (prosodic only)"), ("gKa", "/ka/ only (prosodic only)"),
                          ("gAllM", "All 8 tasks"), ("gIM", "/i/ only")]:
            if prot in r6.index:
                r = r6.loc[prot]; nums += macro(key + "AUC", fmt(r.auc_pooled)) + macro(key + "Lo", fmt(r.ci_lo)) + macro(key + "Hi", fmt(r.ci_hi)) + macro(key + "N", int(r.n_features))
        pros = r6[r6.index.str.endswith("(prosodic only)") & r6.index.str.contains("only (", regex=False)]
        best = pros.auc_pooled.idxmax(); nums += macro("gBestTask", best.replace(" only (prosodic only)", "")) + macro("gBestAUC", fmt(pros.auc_pooled.max()))
        nums += macro("gSingleMin", fmt(pros.auc_pooled.min())) + macro("gSingleMax", fmt(pros.auc_pooled.max()))
        worst = pros.auc_pooled.idxmin(); nums += macro("gWorstTask", worst.replace(" only (prosodic only)", ""))
    # F0 SD released with the database (Praat, Hz, whole recording): HC vs ALS-ND
    from scipy import stats as _st
    for key, c in [("relPa", "stdevF0Hz_PA"), ("relTa", "stdevF0Hz_TA"), ("relKa", "stdevF0Hz_KA")]:
        if c in df.columns:
            xa = pd.to_numeric(df.loc[df.group == "HC", c], errors="coerce").dropna().to_numpy()
            xb = pd.to_numeric(df.loc[df.group == "ALS-ND", c], errors="coerce").dropna().to_numpy()
            mw = _st.mannwhitneyu(xb, xa)
            nums += macro(key + "AUC", fmt(mw.statistic / (len(xa) * len(xb)))) + macro(key + "P", fmt(mw.pvalue, 3))
    # round 3: tracker-independent subharmonic share, run position (end-of-breath creak), same-track folding,
    # intensity/duration/voicing descriptors, and the FDR status of the REAPER versions in the main family
    r3p = T / "table16_round3.csv"
    if r3p.exists():
        r3t = pd.read_csv(r3p)
        r3t = r3t[r3t.contrast == "HC vs ALS-ND"].set_index("measure")
        for key, m in [("rSubh", "subh_share_fold"), ("rSubhRpr", "subh_share_fold_reaper"), ("rSubhMed", "subh_share_med"), ("rUnfl", "subh_share_unflagged"), ("rOct", "oct_share"),
                       ("rOctTrim", "oct_share_trim"), ("rOctEarly", "oct_share_first2thirds"), ("rOctLate", "oct_share_last3rd"),
                       ("rSd", "syl_sd"), ("rFold", "syl_sd_fold10"), ("rSdTrim", "syl_sd_trim"), ("rFoldTrim", "syl_sd_fold_trim")]:
            if m not in r3t.index:
                continue
            r = r3t.loc[m]
            pv = r.p
            nums += macro(key + "AUC", fmt(r.auc)) + macro(key + "Lo", fmt(r.ci_lo)) + macro(key + "Hi", fmt(r.ci_hi))
            nums += macro(key + "P", "<0.001" if pv < 0.001 else fmt(pv, 3))
            if m.startswith("subh") or m.startswith("oct"):
                nums += macro(key + "MedHC", f"{100 * r.median_hc:.1f}") + macro(key + "MedND", f"{100 * r.median_grp:.1f}")
    r3j = a.analysis / "summary_round3.json"
    if r3j.exists():
        R3 = _json.loads(r3j.read_text())
        for key, g in [("HC", "HC"), ("ND", "ALS-ND")]:
            pz = R3["position"][g]
            nums += macro(f"rPos{key}Last", f"{100 * pz['share_events_last3rd']:.0f}") + macro(f"rPos{key}LastThree", f"{100 * pz['share_events_last3syll']:.0f}")
            nums += macro(f"rUnflRate{key}", f"{100 * R3['unflagged_positive_rate'][g]:.1f}") + macro(f"rPos{key}N", int(pz["n_events"]))
        if "single_run_share" in R3:
            nums += macro("rSingleRun", f"{100 * R3['single_run_share']:.0f}")
        nums += macro("rNucLastThree", f"{100 * R3['position']['ALS-ND']['share_nuclei_last3syll']:.0f}") + macro("rDropLast", int(R3["drop_last"]))
    r4p = T / "table17_idv_descriptors.csv"
    if r4p.exists():
        r4t = pd.read_csv(r4p).set_index("descriptor")
        for key, d in [("idvIntV", "V_int_mean"), ("idvVoicedD", "D_voiced_frac"), ("idvDurV", "V_dur"), ("idvDurD", "D_dur"), ("idvIntD", "D_int_mean")]:
            r = r4t.loc[d]
            nums += macro(key + "AUC", fmt(r.auc)) + macro(key + "P", fmt(r.p, 3)) + macro(key + "MedHC", fmt(r.median_hc, 1)) + macro(key + "MedND", fmt(r.median_nd, 1))
        nums += macro("idvBlock", fmt(r4t.loc["all 10 (block)", "auc"])) + macro("idvNoInt", fmt(r4t.loc["block without int_mean", "auc"])) + macro("idvNoVoiced", fmt(r4t.loc["block without voiced_frac", "auc"]))
        nums += macro("idvNoDur", fmt(r4t.loc["block without dur", "auc"]))
    # BH q of the REAPER versions if they replaced the Praat marker in the 39-measure family
    if r3p.exists() and (T / "table12_tracker_robustness.csv").exists():
        e = pd.read_csv(T / "table2_feature_effects.csv")
        e = e[e.contrast == "HC vs ALS-ND"].set_index("feature")
        r12 = pd.read_csv(T / "table12_tracker_robustness.csv")
        from analyze_subclinical import bh_fdr as _bh
        for key, trk, meas in [("trkRprQ", "reaper", "syl_sd"), ("trkFoldRprQ", "reaper", "syl_sd_fold"), ("trkFoldCcQ", "cc60", "syl_sd_fold"),
                               ("trkOctRprQ", "reaper", "oct_frac")]:
            pr = float(r12[(r12.tracker == trk) & (r12.measure == meas)].p.iloc[0])
            pv = e.p.copy(); pv.loc["DDK__syl_f0_sd_st"] = pr
            q = pd.Series(_bh(pv.to_numpy()), index=pv.index).loc["DDK__syl_f0_sd_st"]
            nums += macro(key, fmt(q))
    abp = T / "table15_generic_ablation.csv"
    if abp.exists():
        ab = pd.read_csv(abp)
        fl = ab[ab.model == "age + sex only"].iloc[0]
        nums += macro("abFloorAUC", fmt(fl.auc_pooled)) + macro("abFloorLo", fmt(fl.ci_lo)) + macro("abFloorHi", fmt(fl.ci_hi))
        blocks = [("abFz", "F0 functionals (4)"), ("abVq", "Voice quality (3)"), ("abIdv", "Intensity/duration/voicing (5)"), ("abNf", "All but F0 (8)"), ("abAll", "All 12")]
        blocks = [(k_, b_) for k_, b_ in blocks if (ab.model == f"{b_} | all 8 tasks (means)").any()]
        for key, b in blocks:
            per = ab[ab.model.str.startswith(f"{b} | /")]
            ddk = per[per.task.isin(["/pa/", "/ta/", "/ka/"])].auc_pooled; vow = per[~per.task.isin(["/pa/", "/ta/", "/ka/"])].auc_pooled
            nums += macro(key + "DdkMin", fmt(ddk.min())) + macro(key + "DdkMax", fmt(ddk.max())) + macro(key + "VowMin", fmt(vow.min())) + macro(key + "VowMax", fmt(vow.max()))
        for key, b in blocks:
            med = ab[ab.model == f"{b} | median over 8 single tasks"].iloc[0]
            al = ab[ab.model == f"{b} | all 8 tasks (means)"].iloc[0]
            nums += macro(key + "Med", fmt(med.auc_pooled)) + macro(key + "Min", fmt(med.ci_lo)) + macro(key + "Max", fmt(med.ci_hi))
            nums += macro(key + "All", fmt(al.auc_pooled)) + macro(key + "AllLo", fmt(al.ci_lo)) + macro(key + "AllHi", fmt(al.ci_hi))
            for tk, tl in [("Pa", "/pa/"), ("Ta", "/ta/"), ("Ka", "/ka/"), ("I", "/i/"), ("A", "/a/"), ("E", "/e/")]:
                r = ab[ab.model == f"{b} | {tl}"]
                if len(r):
                    nums += macro(key + tk, fmt(r.iloc[0].auc_pooled))
        # Table 4: ablation
        t4 = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nBlock & /pa/, /ta/, /ka/ & 5 vowels & All tasks [95\\% CI] \\\\\n\\midrule\n"
        for key, b in blocks:
            per = ab[ab.model.str.startswith(f"{b} | /")]
            ddk = per[per.task.isin(["/pa/", "/ta/", "/ka/"])].auc_pooled; vow = per[~per.task.isin(["/pa/", "/ta/", "/ka/"])].auc_pooled
            al = ab[ab.model == f"{b} | all 8 tasks (means)"].iloc[0]
            lab = b.replace("F0", "$F_0$").replace("Intensity/duration/voicing", "Intens., dur., voicing")
            t4 += f"{lab} & {ddk.min():.2f}--{ddk.max():.2f} & {vow.min():.2f}--{vow.max():.2f} & {al.auc_pooled:.2f} [{al.ci_lo:.2f}, {al.ci_hi:.2f}] \\\\\n"
        t4 += f"\\midrule\nAge + sex only & -- & -- & {fl.auc_pooled:.2f} [{fl.ci_lo:.2f}, {fl.ci_hi:.2f}] \\\\\n"
        sa = a.analysis / "summary_ablation.json"
        if sa.exists():
            P = _json.loads(sa.read_text())
            for key, name in [("permPhon", "Phonation"), ("permArt", "Articulation"), ("permResp", "Respiratory support"), ("permGen", "Generic all tasks (24)"), ("permFloor", "Age + sex only")]:
                if name in P:
                    v = P[name]
                    nums += macro(key + "Obs", fmt(v["observed"])) + macro(key + "Null", fmt(v["null_median"])) + macro(key + "Lo", fmt(v["null_q025"])) + macro(key + "Hi", fmt(v["null_q975"])) + macro(key + "P", fmt(v["p_perm"], 3))
                    nums += macro(key + "Excess", fmt(v["observed"] - v["null_median"]))
            nums += macro("permN", int(P["Phonation"]["n_perm"]))
            g = P["Generic all tasks (24)"]
            t4 += f"Permutation null, all 12 & -- & -- & {g['null_median']:.2f} [{g['null_q025']:.2f}, {g['null_q975']:.2f}]$^\\ddagger$ \\\\\n"
        t4 += "\\bottomrule\n\\end{tabular}\n"
        (a.out / "tables" / "table4_ablation.tex").write_text(t4)
    (a.out / "numbers.tex").write_text(nums)

    # ------------------------------------------------------------------ Table 1: cohort
    rows = []
    for grp in ["HC", "ALS-ND", "ALS-D"]:
        s = df[df.group == grp]
        age = pd.to_numeric(s["Age (years)"], errors="coerce")
        onset = s["OnsetRegion"].astype(str).str.lower()
        king = s["KingClinicalStage"].astype(str)
        tot = pd.to_numeric(s["ALSFRS-R_TotalScore"], errors="coerce")
        fvc = pd.to_numeric(s["FVC%"], errors="coerce")
        rows.append([grp, len(s), f"{age.mean():.1f} ({age.std():.1f})", f"{100 * (s['Sex'].str.upper().str[0] == 'M').mean():.0f}",
                     "--" if grp == "HC" else f"{int(onset.str.contains('bulb').sum())}/{int(onset.str.contains('spin').sum())}",
                     "--" if grp == "HC" else f"{int(tot.median())} ({int(tot.quantile(.25))}--{int(tot.quantile(.75))})",
                     "--" if grp == "HC" else f"{fvc.median():.0f}",
                     "--" if grp == "HC" else f"{int((king.str.startswith('3') | king.str.startswith('4')).sum())}"])
    # cohort macros (for the in-text cohort description)
    cm = ""
    for grp, key in [("HC", "cHC"), ("ALS-ND", "cND"), ("ALS-D", "cD")]:
        s_ = df[df.group == grp]
        age = pd.to_numeric(s_["Age (years)"], errors="coerce")
        onset = s_["OnsetRegion"].astype(str).str.lower()
        tot = pd.to_numeric(s_["ALSFRS-R_TotalScore"], errors="coerce")
        fvc = pd.to_numeric(s_["FVC%"], errors="coerce")
        cm += macro(key + "N", len(s_)) + macro(key + "Age", f"{age.mean():.1f}") + macro(key + "AgeSD", f"{age.std():.1f}") + macro(key + "Male", f"{100 * (s_['Sex'].str.upper().str[0] == 'M').mean():.0f}")
        if grp != "HC":
            cm += macro(key + "Bulb", int(onset.str.contains('bulb').sum())) + macro(key + "Spin", int(onset.str.contains('spin').sum()))
            cm += macro(key + "Frs", int(tot.median())) + macro(key + "FrsQa", int(tot.quantile(.25))) + macro(key + "FrsQb", int(tot.quantile(.75))) + macro(key + "Fvc", f"{fvc.median():.0f}")
    with open(a.out / "numbers.tex", "a") as fh:
        fh.write(cm)
    t1 = "\\begin{tabular}{@{}lrccccc@{}}\n\\toprule\nGroup & $n$ & Age & Male\\% & Onset B/S & ALSFRS-R & FVC\\% \\\\\n\\midrule\n"
    for r in rows:
        t1 += " & ".join(str(x) for x in r[:7]) + " \\\\\n"
    t1 += "\\bottomrule\n\\end{tabular}\n"
    (a.out / "tables" / "table1_cohort.tex").write_text(t1)

    # ------------------------------------------------------------------ Table 2: subclinical effects (top 12 + reference rows)
    order = ["DDK__syl_f0_sd_st", "DDK__f0_sd_st_all", "V__f0_sd_st_all", "V__jitter_ppq5",
             "V__unvoiced_frac_run", "V__shimmer_apq11", "V__hnr", "V__cpps",
             "ART__fcr", "DDK__ddk_rate", "ART__vsa", "DDK__syll_per_run", "V__mpt_proxy_max", "V__int_slope"]
    t2 = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{HC vs ALS-ND} & ALS-D \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-5}\nMeasure & AUC [95\\% CI] & $q$ & AUC$_{\\rm adj}$ ($q_{\\rm adj}$) & AUC \\\\\n\\midrule\n"
    last_track = None
    for f in order:
        r, rd = nd.loc[f], dd.loc[f]
        q = f"{r.q:.3f}" if r.q >= 0.001 else "$<$0.001"
        qa = f"{r.q_adj:.3f}" if r.q_adj >= 0.001 else "$<$0.001"
        lab = r.label.replace("F0", "$F_0$").replace("(st)", "(st)")
        bold = r.q_adj < 0.05
        cell = lambda s: f"\\textbf{{{s}}}" if bold else s
        if r.track != last_track:
            t2 += f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{r.track}}}}} \\\\\n"
            last_track = r.track
        t2 += f"\\;{cell(lab)} & {cell(f'{r.auc:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]')} & {q} & {r.auc_adj:.2f} ({qa}) & {rd.auc:.2f} \\\\\n"
    t2 += "\\bottomrule\n\\end{tabular}\n"
    (a.out / "tables" / "table2_effects.tex").write_text(t2)

    # ------------------------------------------------------------------ Table 3: clinical anchoring (all ALS)
    feats = [("DDK__syl_f0_sd_st", "DDK syllable $F_0$ SD"), ("DDK__f0_sd_st_all", "$F_0$ SD, DDK"), ("V__jitter_ppq5", "Jitter (PPQ5)"),
             ("DDK__ddk_rate", "DDK rate"), ("ART__fcr", "FCR"),
             ("DDK__syll_per_run", "Syllables per run"), ("V__mpt_proxy_max", "Longest phonation"), ("V__int_slope", "Intensity slope")]
    clins = [("pumns_bulbar", "PUMNS"), ("mrc_head", "MRC"), ("bulbar_sum", "Bulbar"), ("fvc", "FVC"), ("dyspnea", "Dyspnea")]
    t3 = "\\begin{tabular}{@{}l" + "c" * len(clins) + "@{}}\n\\toprule\nMeasure & " + " & ".join(c for _, c in clins) + " \\\\\n\\midrule\n"
    for f, lab in feats:
        cells = []
        for c, _ in clins:
            r, q, n = rho("all ALS", c, f)
            s = fmt(r)
            if np.isfinite(q) and q < 0.05:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        t3 += lab + " & " + " & ".join(cells) + " \\\\\n"
    t3 += "\\bottomrule\n\\end{tabular}\n"
    (a.out / "tables" / "table3_clinical.tex").write_text(t3)
    print("wrote numbers.tex and 3 table fragments to", a.out)


if __name__ == "__main__":
    main()
