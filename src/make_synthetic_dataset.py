#!/usr/bin/env python3
"""
Generate a small synthetic VOC-ALS-like dataset (8 kHz wav + xlsx) to smoke-test the pipeline.
Not for science: only to check that the extraction code runs end to end and that the
feature values move in the expected direction (e.g. higher jitter -> higher measured jitter).

Usage: python make_synthetic_dataset.py OUT_DIR [--n-hc 3 --n-als 3 --seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import lfilter

SR = 8000
VOWEL_FORMANTS = {  # rough adult formants (Hz)
    "A": (750, 1200), "E": (500, 1900), "I": (300, 2300), "O": (500, 900), "U": (320, 800),
}


def resonator(y, f, bw, sr=SR):
    """Simple 2-pole resonator (formant filter)."""
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * f / sr
    a = [1, -2 * r * np.cos(theta), r ** 2]
    b = [1 - r]
    return lfilter(b, a, y)


def glottal_source(dur, f0, jitter, shimmer, rng, sr=SR):
    """Pulse train with cycle-to-cycle period and amplitude perturbation."""
    n = int(dur * sr)
    y = np.zeros(n)
    t = 0.0
    while True:
        period = (1.0 / f0) * (1 + jitter * rng.standard_normal())
        amp = 1.0 + shimmer * rng.standard_normal()
        i = int(round(t * sr))
        if i >= n:
            break
        # Rosenberg-like pulse: short decaying exponential
        L = int(0.6 * period * sr)
        k = np.arange(min(L, n - i))
        y[i:i + len(k)] += amp * np.exp(-k / (0.15 * period * sr)) * np.sin(np.pi * k / max(L, 1))
        t += period
    return y


def synth_vowel(vowel, dur, f0, jitter, shimmer, noise, decay_db, rng):
    src = glottal_source(dur, f0, jitter, shimmer, rng)
    f1, f2 = VOWEL_FORMANTS[vowel]
    y = resonator(src, f1, 80) + 0.5 * resonator(src, f2, 110) + 0.2 * resonator(src, 2900, 200)
    y += noise * rng.standard_normal(len(y)) * np.abs(y).max()
    n = len(y)
    env = np.ones(n)
    ramp = int(0.15 * SR)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    env *= 10 ** (np.linspace(0, decay_db, n) / 20)
    y = y * env
    # add leading/trailing silence
    pad = np.zeros(int(0.4 * SR))
    y = np.concatenate([pad, y, pad])
    return 0.6 * y / np.abs(y).max()


def synth_ddk(n_syll, rate, ioi_cv, f0, jitter, shimmer, decay_db, rng):
    ioi = 1.0 / rate
    syl_dur = 0.55 * ioi
    total = n_syll * ioi + 1.0
    y = np.zeros(int(total * SR))
    t = 0.4
    for k in range(n_syll):
        src = glottal_source(syl_dur, f0, jitter, shimmer, rng)
        s = resonator(src, 700, 90) + 0.5 * resonator(src, 1250, 120)
        burst = rng.standard_normal(int(0.012 * SR)) * 0.4  # stop release burst
        s = np.concatenate([burst, s])
        env = np.ones(len(s)); r = int(0.02 * SR)
        env[:r] = np.linspace(0, 1, r); env[-r:] = np.linspace(1, 0, r)
        s = s * env * 10 ** ((decay_db * k / max(n_syll - 1, 1)) / 20)
        i = int(t * SR)
        y[i:i + len(s)] += s[: len(y) - i]
        t += ioi * (1 + ioi_cv * rng.standard_normal())
    y += 0.003 * rng.standard_normal(len(y))
    return 0.6 * y / np.abs(y).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-hc", type=int, default=3)
    ap.add_argument("--n-als-nd", type=int, default=2, help="ALS with normal-rated speech (subclinical, small shifts)")
    ap.add_argument("--n-als-d", type=int, default=1, help="ALS with dysarthria (large shifts)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    subjects = [("CS", i, "HC") for i in range(1, a.n_hc + 1)]
    subjects += [("PZ", i, "ALS-ND") for i in range(1, a.n_als_nd + 1)]
    subjects += [("PZ", a.n_als_nd + i, "ALS-D") for i in range(1, a.n_als_d + 1)]
    for prefix, i, grp in subjects:
        cat = "HC" if grp == "HC" else "ALS"
        sid = f"{prefix}{i:03d}"
        sex = rng.choice(["M", "F"])
        f0 = 110 if sex == "M" else 200
        j = lambda lo, hi: float(rng.uniform(lo, hi))  # per-subject variability
        if grp == "HC":
            jit, shim, noise, decay, rate, icv, nsyl = j(0.003, 0.006), j(0.02, 0.05), 0.002, j(-2, 0), j(5.5, 6.5), j(0.04, 0.08), int(rng.integers(18, 24))
            speech = 4
        elif grp == "ALS-ND":  # subtle shifts, speech rated normal
            jit, shim, noise, decay, rate, icv, nsyl = j(0.004, 0.009), j(0.03, 0.07), 0.005, j(-3, 0), j(4.8, 6.2), j(0.06, 0.12), int(rng.integers(15, 22))
            speech = 4
        else:
            jit, shim, noise, decay, rate, icv, nsyl = j(0.01, 0.03), j(0.08, 0.15), 0.02, j(-8, -3), j(2.5, 4.5), j(0.15, 0.30), int(rng.integers(8, 15))
            speech = int(rng.choice([3, 3, 2, 1]))
        for v in "AEIOU":
            d = a.out / f"phonation{v}"; d.mkdir(parents=True, exist_ok=True)
            y = synth_vowel(v, 5.5 + rng.uniform(-0.5, 1.0), f0 * (1 + 0.05 * rng.standard_normal()), jit, shim, noise, decay, rng)
            wavfile.write(d / f"{sid}_phonation{v}.wav", SR, (y * 32767).astype(np.int16))
        for s in ["PA", "TA", "KA"]:
            d = a.out / f"rhythm{s}"; d.mkdir(parents=True, exist_ok=True)
            y = synth_ddk(nsyl, rate, icv, f0, jit, shim, decay, rng)
            wavfile.write(d / f"{sid}_rhythm{s}.wav", SR, (y * 32767).astype(np.int16))
        als = cat == "ALS"
        rows.append({"ID": sid, "age": int(rng.integers(45, 80)), "sex": sex, "category": cat,
                     "OnsetRegion": (rng.choice(["spinal", "bulbar"], p=[0.75, 0.25]) if als else np.nan),
                     "DiseaseDuration": float(rng.uniform(3, 60)) if als else np.nan,
                     "FVC%": float(rng.uniform(50, 110)) if als else np.nan,
                     "ALSFRS-R_TotalScore": int(rng.integers(20, 48)) if als else np.nan,
                     "ALSFRS-R_SpeechSubscore": speech if als else np.nan,
                     "ALSFRS-R_SalivationSubscore": int(rng.integers(2, 5)) if als else np.nan,
                     "ALSFRS-R_SwallowingSubscore": int(rng.integers(2, 5)) if als else np.nan,
                     "KingClinicalStage": rng.choice(["1", "2", "3", "4A", "4B"]) if als else np.nan,
                     "MRC_HeadMuscles": int(rng.integers(10, 16)) if als else np.nan,
                     "PUMNS_BulbarSubscore": int(rng.integers(0, 5)) if als else np.nan})
    pd.DataFrame(rows).to_excel(a.out / "VOC-ALS_synthetic.xlsx", index=False)
    print(f"wrote {len(subjects)} subjects x 8 files to {a.out}")


if __name__ == "__main__":
    main()
