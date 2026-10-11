#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper from the raw VOC-ALS data.
set -euo pipefail
cd "$(dirname "$0")"
python src/extract_features.py --data data/raw --xlsx data/raw/VOC-ALS.xlsx --out results/features.csv --jobs 2 --debug-plots results/ddk_plots --debug-n 40
python src/analyze_subclinical.py --features results/features.csv --out results/analysis --repeats 20 --boot 2000
python src/mfcc_reference.py --data data/raw --features results/features.csv --out results/analysis
python src/extract_extended.py --data data/raw --out results/extended.csv --jobs 2 --debug-plots results/onset_plots --debug-n 16
python src/analyze_extended.py --features results/features.csv --extended results/extended.csv --out results/analysis
python src/analyze_revision.py --data data/raw --features results/features.csv --extended results/extended.csv --out results/analysis --jobs 2   # uses tools/reaper if present
python src/analyze_ablation.py --features results/features.csv --out results/analysis --perms 200
python src/analyze_subclinical.py --features results/features.csv --out results/analysis --figures-only
python src/analyze_round3.py --data data/raw --features results/features.csv --analysis results/analysis --jobs 2   # uses tools/reaper if present
python src/make_fig_concept.py --data data/raw --hc CT063 --als PZ012 --analysis results/analysis --out results/analysis/figures/fig0_example
python src/make_paper_tables.py --features results/features.csv --analysis results/analysis --out paper
(cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
echo "done: paper/main.pdf"
