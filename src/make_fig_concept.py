#!/usr/bin/env python3
"""
Figure 1 of the paper.
(a) Motor-pathway view that organises the measures: upper motor neuron -> lower-motor-neuron pool -> speech
    subsystem -> acoustic measures, annotated with the cross-validated AUC of each subsystem composite
    (HC vs ALS-ND) against its permutation null (results/analysis/summary_ablation.json).
(b) The marker on real data: /pa/ repetition of one control and one patient with normal-rated speech:
    energy envelope with syllable peaks, nucleus F0 per syllable (semitones re the median nucleus F0),
    and the 50 ms spectrum of one SHR-confirmed octave-down nucleus (the one whose half-harmonic peaks align
    best with the local F0, i.e. a clear example), with harmonics and half-harmonics of the local F0 marked.

Usage: python make_fig_concept.py --data data/raw --hc CT063 --als PZ012 --analysis results/analysis \
       --out results/analysis/figures/fig0_concept
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import detect_syllables, PITCH_FLOOR, PITCH_CEILING, PITCH_STEP  # noqa: E402
from extract_extended import semitones_folded  # noqa: E402
from analyze_revision import shr_db  # noqa: E402

BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#E69F00", "#009E73", "#6F6F6F", "#D55E00"
FS = 7.0   # base font size (pt); the figure is printed at full text width (7.1 in)


def nucleus_f0(path):
    import parselmouth
    from scipy.io import wavfile
    sr, y = wavfile.read(str(path))
    y = y.astype(np.float64) / 32768.0
    pk_t, pk_db, (te, env) = detect_syllables(y, sr)
    p = parselmouth.Sound(y, sampling_frequency=sr).to_pitch_cc(time_step=PITCH_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0, tt = p.selected_array["frequency"], p.xs()
    med = []
    for pt in pk_t:
        w = (tt >= pt - 0.04) & (tt <= pt + 0.04) & (f0 > 0)
        med.append(float(np.median(f0[w])) if w.sum() >= 2 else np.nan)
    return y, sr, pk_t, np.array(med), te, env


def local_spectrum(y, sr, t, f_nuc, win=0.05):
    """50 ms Hann spectrum at t; local F0 = argmax of the mean level at k*f (k=1..4) for f in [1.7, 2.3] x f_nuc;
    dev_sub = largest relative distance between (k+1/2) x F0 (k=1..3) and the nearest spectral peak."""
    from scipy.signal import find_peaks
    n = int(win * sr); c = int(t * sr)
    if c - n // 2 < 0 or c + n // 2 > len(y):
        return None
    seg = y[c - n // 2: c + n // 2] * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg, 8192)); fr = np.fft.rfftfreq(8192, 1 / sr)
    db = 20 * np.log10(spec / spec.max() + 1e-9)
    cands = np.arange(1.7 * f_nuc, 2.3 * f_nuc, 0.5)
    hs = [np.mean([db[np.argmin(abs(fr - k * f))] for k in range(1, 5)]) for f in cands]
    f0 = float(cands[int(np.argmax(hs))])
    pk, _ = find_peaks(db[fr < 5 * f0], prominence=4)
    if len(pk) < 3:
        return None
    pf = fr[pk]
    dev_sub = max(float(np.min(np.abs(pf - (k + 0.5) * f0))) / f0 for k in range(1, 4))
    return {"f0": f0, "fr": fr, "db": db, "dev_sub": dev_sub}


def panel_a(ax, null):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(-0.12, 10.05); ax.set_ylim(0, 10); ax.axis("off")

    def comp(name):
        v = null.get(name)
        if not v:
            return ""
        rel = ">" if v["p_perm"] < 0.05 else "≈"
        return f"AUC {v['observed']:.2f} {rel} null"

    rows = [
        ("spinal motoneurons\nC3–T12 (phrenic,\nintercostal nerves)", "Respiratory\nsupport", "longest phonation,\nintensity decline,\nsyllables per run", comp("Respiratory support"), GREY),
        ("nucleus ambiguus\n→ vagus (X):\nlaryngeal nerves", "Laryngeal\n(phonation)", "F0 spread across\nre-onsets, octave\nevents, jitter, HNR", comp("Phonation"), BLUE),
        ("hypoglossal (XII),\nfacial (VII),\ntrigeminal (V)", "Articulatory\n(oral)", "DDK rate and rhythm,\nvowel space, FCR,\nformant stability", comp("Articulation"), GREY),
    ]
    ax.text(0.0, 9.72, "upper motor neuron (corticobulbar / corticospinal tracts)", fontsize=FS - 0.4, color=GREY, style="italic", ha="left", va="center")
    ax.annotate("", xy=(1.55, 8.75), xytext=(1.55, 9.45), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8))
    ys = [7.45, 4.85, 2.25]
    cols = [(0.05, 3.0, False), (3.4, 2.6, True), (6.4, 3.55, False)]   # x0, width, filled
    for (pool, sub, meas, auc, col), yc in zip(rows, ys):
        for (x0, w, fill), txt in zip(cols, [pool, sub, meas]):
            ax.add_patch(FancyBboxPatch((x0, yc - 1.05), w, 2.1, boxstyle="round,pad=0.03,rounding_size=0.15",
                                        fc=col if fill else "white", ec=col, lw=1.0, alpha=0.16 if fill else 1.0))
            ax.add_patch(FancyBboxPatch((x0, yc - 1.05), w, 2.1, boxstyle="round,pad=0.03,rounding_size=0.15",
                                        fc="none", ec=col, lw=1.0))
            ax.text(x0 + w / 2, yc + (0.28 if fill else 0), txt, ha="center", va="center", fontsize=FS - 0.5, color="black", linespacing=1.15)
        ax.text(cols[1][0] + cols[1][1] / 2, yc - 0.68, auc, ha="center", va="center", fontsize=FS - 0.6, color=col, fontweight="bold")
        ax.annotate("", xy=(3.4, yc), xytext=(3.08, yc), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9))
        ax.annotate("", xy=(6.4, yc), xytext=(6.03, yc), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9))
    for x, lab in [(1.55, "lower motor neuron"), (4.7, "subsystem"), (8.17, "acoustic measures")]:
        ax.text(x, 0.55, lab, fontsize=FS - 0.4, color=GREY, ha="center")


def panel_b(ax_env, ax_f0, ax_sp, data_dir, hc_id, als_id, n_show=45, n_env=15):
    for sid, col, lab in [(hc_id, GREEN, "control"), (als_id, BLUE, "ALS-ND")]:
        y, sr, pk_t, med, te, env = nucleus_f0(data_dir / f"{sid}_rhythmPA.wav")
        ok = np.isfinite(med)
        ref = float(np.nanmedian(med))
        st = 12 * np.log2(med / ref)
        fold, octv = semitones_folded(med[ok], ref)
        k = np.arange(1, len(med) + 1)
        m_show = k <= n_show
        ax_f0.plot(k[ok & m_show], st[ok & m_show], "o-", color=col, ms=2.4, lw=0.7,
                   label=f"{lab} {sid}: SD {np.std(st[ok], ddof=1):.1f} st (folded {fold.std(ddof=1):.1f}), {octv.mean() * 100:.0f}% octave")
        if sid != als_id:
            continue
        oc = ok & (np.abs(st) > 6) & m_show
        ax_f0.plot(k[oc], st[oc], "o", mfc="none", mec=RED, ms=6.0, mew=0.9, label="octave-level re-onset (|dev| > 6 st)")
        # envelope of the first ~6 s with the detected syllable peaks
        t_end = min(te[-1], pk_t[min(n_env - 1, len(pk_t) - 1)] + 0.3)
        m = te <= t_end
        ax_env.plot(te[m], env[m], color=col, lw=0.7)
        ax_env.plot(pk_t[pk_t <= t_end], np.interp(pk_t[pk_t <= t_end], te, env), ".", color=RED, ms=4)
        ax_env.set_xlim(te[m][0], t_end)
        ax_env.set_ylabel("dB", fontsize=FS - 0.3, labelpad=1)
        ax_env.set_xlabel("time (s)", fontsize=FS - 0.3, labelpad=0.5)
        ax_env.set_title(f"/pa/, {sid}: envelope, peaks", fontsize=FS - 0.4, loc="left", pad=2)
        # spectrum of an SHR-confirmed octave-down nucleus: the local F0 is estimated from the spectrum itself
        # (harmonic-sum search around twice the tracked nucleus F0); among this speaker's confirmed nuclei the one
        # whose half-harmonic peaks align best with (k+1/2) x local F0 is shown (a clear example, as stated in the caption)
        # period-doubling pattern: SHR-confirmed (> -6 dB) but subharmonics weaker than harmonics (< 0 dB)
        idx = [i for i in np.where(oc & (st < 0))[0] if -6 < shr_db(y, sr, pk_t[i], med[i]) < 0]
        best = None
        for i in idx:
            r = local_spectrum(y, sr, pk_t[i], med[i])
            if r is not None and (best is None or r["dev_sub"] < best[1]["dev_sub"]):
                best = (i, r)
        if best is not None:
            i0, r = best
            f0l, fr, db = r["f0"], r["fr"], r["db"]
            mm = fr <= 4.7 * f0l
            ax_sp.plot(fr[mm] / f0l, db[mm], color="k", lw=0.7)
            for h in range(1, 5):
                ax_sp.axvline(h, color=BLUE, lw=0.8, ls="--")
            for h in [0.5, 1.5, 2.5, 3.5, 4.5]:
                ax_sp.axvline(h, color=RED, lw=0.8, ls=":")
            ax_sp.set_xticks([0.5, 1, 2, 3, 4])
            ax_sp.set_xticklabels(["½", "1", "2", "3", "4"], fontsize=FS - 0.4)
            for tl, c_ in zip(ax_sp.get_xticklabels(), [RED, BLUE, BLUE, BLUE, BLUE]):
                tl.set_color(c_)
            ax_sp.set_xlim(0, 4.7)
            ax_sp.set_xlabel("frequency / local F0", fontsize=FS - 0.3, labelpad=0.5)
            ax_sp.text(0.03, 0.985, "dashed: harmonics\ndotted: (k+½)F0", transform=ax_sp.transAxes, fontsize=FS - 1.0,
                       ha="left", va="top", color=GREY, linespacing=1.1, bbox=dict(fc="white", ec="none", pad=0.6))
            ax_sp.set_ylim(-50, 3)
            ax_sp.set_ylabel("dB re max", fontsize=FS - 0.3, labelpad=1)
            shr0 = shr_db(y, sr, pk_t[i0], med[i0])
            ax_sp.set_title(f"syllable {i0 + 1}, 50 ms:\nF0 {f0l:.0f} Hz, SHR {shr0:+.1f} dB", fontsize=FS - 0.4, loc="left", pad=2)
            ax_f0.annotate("", xy=(i0 + 1, st[i0] - 0.8), xytext=(i0 + 1, st[i0] - 4.5), arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8))
    for lim in [6, -6]:
        ax_f0.axhline(lim, color=RED, lw=0.5, ls=":")
    ax_f0.axhline(0, color="k", lw=0.5)
    ax_f0.set_ylim(-15, 23)
    ax_f0.set_xlim(0, n_show + 1)
    ax_f0.set_xlabel("syllable number", fontsize=FS - 0.3, labelpad=0.5)
    ax_f0.set_ylabel("nucleus F0 (st)", fontsize=FS - 0.3, labelpad=1)
    ax_f0.set_yticks([-12, -6, 0, 6])
    leg = ax_f0.legend(fontsize=FS - 1.2, frameon=True, loc="upper left", handlelength=1.2, borderaxespad=0.05, labelspacing=0.15, borderpad=0.2)
    leg.get_frame().set_edgecolor("none"); leg.get_frame().set_alpha(1.0)
    for ax in (ax_env, ax_f0, ax_sp):
        ax.tick_params(labelsize=FS - 0.8, length=2, pad=1.5)
        for s_ in ["top", "right"]:
            ax.spines[s_].set_visible(False)


def main(argv=None):
    global FS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--hc", default="CT063")
    ap.add_argument("--als", default="PZ012")
    ap.add_argument("--analysis", type=Path, default=Path("results/analysis"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--with-diagram", action="store_true", help="two-column version with the motor-pathway diagram (panel a)")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    if args.with_diagram:
        null = json.loads((args.analysis / "summary_ablation.json").read_text())
        fig = plt.figure(figsize=(7.1, 2.45))
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.36], wspace=0.11, left=0.004, right=0.995, top=0.9, bottom=0.125)
        ax_a = fig.add_subplot(outer[0])
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], width_ratios=[1.75, 1.0], height_ratios=[0.62, 1.38],
                                                 hspace=0.62, wspace=0.27)
        ax_env = fig.add_subplot(inner[0, 0]); ax_f0 = fig.add_subplot(inner[1, 0]); ax_sp = fig.add_subplot(inner[:, 1])
        panel_a(ax_a, null)
        panel_b(ax_env, ax_f0, ax_sp, args.data, args.hc, args.als)
        fig.text(0.004, 0.962, "(a)", fontsize=FS + 1.0, fontweight="bold"); fig.text(0.445, 0.962, "(b)", fontsize=FS + 1.0, fontweight="bold")
    else:
        # single-column version used in the paper: nucleus F0 trajectory on top, envelope and spectrum below
        fig = plt.figure(figsize=(3.45, 2.75))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.25, 1.0], hspace=0.62, wspace=0.38,
                               left=0.13, right=0.985, top=0.975, bottom=0.115)
        ax_f0 = fig.add_subplot(gs[0, :]); ax_env = fig.add_subplot(gs[1, 0]); ax_sp = fig.add_subplot(gs[1, 1])
        FS = 6.6
        panel_b(ax_env, ax_f0, ax_sp, args.data, args.hc, args.als, n_env=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf")); fig.savefig(args.out.with_suffix(".png"), dpi=220)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
