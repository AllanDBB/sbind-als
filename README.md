# Which speech measures differ when ALS speech is rated normal?

Between-syllable laryngeal instability — analysis code and paper for
SBiND 2027 (Speech Biomarkers for Neurodegenerative Diseases, ICASSP 2027 satellite workshop).

Everything in the paper is generated from the public **VOC-ALS** database
(Dubbioso et al., *Scientific Data* 2024; Synapse [syn53009474](https://doi.org/10.7303/syn53009474)).
The audio is **not** redistributed here; download it from Synapse (free registered account) and
place the `.wav` files and `VOC-ALS.xlsx` under `data/raw/`.
`results/features.csv` (subject-level measures merged with the clinical columns of the database) is not
included for the same reason; step 1 below regenerates it from the Synapse download.

## Pipeline

```bash
pip install -r requirements.txt           # numpy, scipy, pandas, scikit-learn, matplotlib, openpyxl, praat-parselmouth, joblib

# 1. Acoustic measures (Praat/Parselmouth): ~12 min for 1224 files on 2 cores
python src/extract_features.py --data data/raw --xlsx data/raw/VOC-ALS.xlsx \
       --out results/features.csv --jobs 2 --debug-plots results/ddk_plots --debug-n 40

# 2. Subclinical analysis: effect sizes, task efficiency (repeated CV), clinical anchoring, figures
python src/analyze_subclinical.py --features results/features.csv --out results/analysis --repeats 20 --boot 2000

# 3. Non-interpretable reference (MFCC statistics + the same CV protocol)
python src/mfcc_reference.py --data data/raw --features results/features.csv --out results/analysis

# 4. Extended signal-processing / neurology analyses (time-scale decomposition, voice re-initiation,
#    UMN vs LMN partial correlations, progression): ~2 min extraction + ~1 min analysis
python src/extract_extended.py --data data/raw --out results/extended.csv --jobs 2 --debug-plots results/onset_plots --debug-n 16
python src/analyze_extended.py --features results/features.csv --extended results/extended.csv --out results/analysis

# 5. Revision analyses: paired-bootstrap subsystem comparison, sensitivity tiers (normal bulbar examination, with CIs),
#    tracker robustness (Praat cc/ac, 60/75 Hz floor, and REAPER if tools/reaper exists) + SHR validation of octave
#    events with a control false-positive rate, recording/task covariates, within-session consistency (ICC, split-half),
#    generic prosodic (eGeMAPS-like) reference: ~8 min
python src/analyze_revision.py --data data/raw --features results/features.csv --extended results/extended.csv --out results/analysis --jobs 2

# 6. Chance references: age+sex floor, block ablation of the generic set, permutation null of the CV AUC (~10 min)
python src/analyze_ablation.py --features results/features.csv --out results/analysis --perms 200

# 7. Figures (incl. the concept/example figure), then LaTeX tables + number macros for the paper
python src/analyze_subclinical.py --features results/features.csv --out results/analysis --figures-only
python src/analyze_round3.py --data data/raw --features results/features.csv --analysis results/analysis --jobs 2
python src/make_fig_concept.py --data data/raw --hc CT063 --als PZ012 --analysis results/analysis --out results/analysis/figures/fig0_example
python src/make_paper_tables.py --features results/features.csv --analysis results/analysis --out paper

# 8. Paper (ICASSP style: spconf.sty + IEEEbib.bst)
cd paper && latexmk -pdf main.tex
```

`run_all.sh` runs the steps in order. All random procedures use a fixed seed (`--seed 0`).
The independent pitch tracker is Google's REAPER: `git clone https://github.com/google/REAPER && cd REAPER && mkdir build && cd build && cmake .. && make`,
then copy the `reaper` binary to `tools/reaper` (step 5 skips it when absent).

## What the scripts do

| Script | Purpose |
| --- | --- |
| `src/extract_features.py` | One row per recording and one row per subject. Phonation (F0 spread in semitones, jitter, shimmer, HNR, CPPS, tremor index, voice breaks, spectral balance), articulation (formants, vowel space area, FCR, DDK rate and rhythm) and respiratory support (longest phonation, intensity decline, syllables per breath run). Validated against the acoustic columns released with VOC-ALS (Spearman 0.95–0.99 for jitter, shimmer, HNR and mean F0). |
| `src/analyze_subclinical.py` | Groups: HC, ALS-ND (ALSFRS-R speech item = 4) and ALS-D (< 4). Per-measure AUC with bootstrap CI, BH-FDR, age/sex-adjusted AUC; per-task L2-logistic models under 20×5-fold stratified CV with calibration; Spearman correlations with PUMNS bulbar, MRC head, ALSFRS-R items, FVC%; onset and severity analyses. Writes `results/analysis/tables/*.csv` and `figures/*`. |
| `src/mfcc_reference.py` | 39 MFCC statistics per task with the same CV protocol, as a non-interpretable reference. |
| `src/extract_extended.py` | Time-scale decomposition of the F0 contour (modulation bands 0.25–1, 1–4, 4–8, 8–15 Hz, octave-folded), within- vs across-syllable F0 variation in the DDK task, octave-level event rates, and voice re-initiation dynamics (F0 at voicing onset, burst-to-voicing delay, amplitude rise time). |
| `src/analyze_revision.py` | Reviewer-requested checks: paired bootstrap of AUC differences between subsystems (single measures and CV composites), sensitivity tiers on ALS-ND subsets with a normal bulbar examination (bootstrap CIs), pitch-tracker robustness (Praat cc/ac, 60/75 Hz floor, REAPER) with subharmonic-to-harmonic-ratio validation of octave-level nuclei and its control false-positive rate, SNR/level/rate covariates, within-session consistency (ICC(2,1) across /pa/,/ta/,/ka/; split-half), and a generic prosodic reference (12 eGeMAPS-like functionals per task, with/without MFCC). Writes `table10–14` and `summary_revision.json`. |
| `src/analyze_ablation.py` | Chance references: age+sex-only model, block ablation of the generic set (F0 functionals / voice quality / intensity-duration-voicing, per task and pooled), permutation null of the cross-validated AUC for every composite. Writes `table15_generic_ablation.csv` and `summary_ablation.json`. |
| `src/analyze_round3.py` | Third-round checks: SHR at every syllable nucleus at half its octave-folded F0 (Praat and REAPER) or half the speaker's median; positive share among unflagged nuclei; position of octave events within pause-delimited runs and the contrast without the last 3 syllables of each run; same-track folding; which intensity/duration/voicing descriptor drives the generic block. Writes `table16`, `table17`, `summary_round3.json`. |
| `src/make_fig_concept.py` | Fig. 1 of the paper: a real /pa/ example (nucleus F0 per syllable, envelope peaks, 50 ms spectrum of an SHR-confirmed nucleus marked at its local F0). `--with-diagram` adds the motor-pathway diagram (two-column version, not used in the paper). |
| `src/analyze_extended.py` | Effect sizes with family-wise FDR for the extended measures; partial Spearman correlations with PUMNS bulbar (UMN) controlling MRC head (LMN) and vice versa; association with progression rate and fast-vs-slow progressors. Writes `table7–9` and `fig4_timescale`, `fig5_umn_lmn`. |
| `src/make_paper_tables.py` | Generates `paper/numbers.tex` (every number quoted in the text) and `paper/tables/*.tex` from the CSVs, so the paper is traceable to the analysis. |
| `src/make_synthetic_dataset.py` | Synthetic 8 kHz vowels and syllable trains to smoke-test the extraction without the real data. |

## Main result (VOC-ALS, n = 153)

* In ALS patients whose speech the neurologist rates as normal (n = 53, all spinal onset), the only one of 39
  interpretable measures that passes false-discovery control is the semitone spread of F0 across the syllable
  nuclei of the /pa/, /ta/, /ka/ tasks (AUC 0.70). The whole-recording F0 SD released with the database (Hz) already
  separates the groups on /pa/ (AUC 0.67), so the contribution is what the variability consists of.
* On the same 10 ms track it splits into an octave-folded spread (0.66) and octave-level events (0.70); 62% of
  octave-down nuclei pass the subharmonic-to-harmonic-ratio criterion (7% of unflagged nuclei, which are themselves
  subharmonic more often in patients). The SHR at every nucleus (half the folded F0) gives 0.64 (p = 0.016).
* Octave events are commoner at the end of pause-delimited runs (end-of-breath creak), but the contrast holds without
  the last 3 syllables of each run (octave share 0.70, spread 0.69).
* Tracker dependence: with REAPER the unfolded spread falls to 0.61 (p = 0.049), the octave share to 0.59 and the
  spectral subharmonic share to 0.58; the folded spread is stable (0.66 Praat vs 0.64 REAPER, rho = 0.80) but only
  nominal (q = 0.13-0.24 if placed in the 39-measure family).
* The phonatory composite (cross-validated AUC 0.64) exceeds its permutation null; articulatory (0.54) and
  respiratory (0.40) composites do not, but the phonatory-articulatory difference is uncertain (0.10 [-0.01, 0.21]).
* Generic descriptors show a crossover: F0 functionals carry the syllable-task signal, voice quality and mean level
  the vowel signal; the intensity/duration/voicing block is driven by mean vowel intensity and DDK voiced fraction
  (task execution), not by duration or respiratory support.
* With dysarthria every subsystem changes; respiratory and articulatory measures track MRC head strength,
  ALSFRS-R bulbar items and FVC% and align with head-muscle strength rather than the PUMNS bulbar score.

## Citation

If you use this code, cite the paper (to appear, SBiND 2027) and the VOC-ALS database:
Dubbioso R. et al., "Voice signals database of ALS patients with different dysarthria severity and healthy
controls", *Scientific Data* 11:800, 2024, doi:10.1038/s41597-024-03597-2.

## License

Code: MIT. The VOC-ALS data are governed by the Synapse terms of use.
