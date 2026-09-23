#!/usr/bin/env bash
# Rebuild every number, figure and the PDF from the cached API responses (no network calls needed).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { python3 -m venv .venv; .venv/bin/pip install -q -r requirements.txt; }
$PY experiments/analyze_e1e3.py
$PY experiments/e4_selected_ci.py     # PySCF recomputation; Jev answers come from data/cache
$PY experiments/analyze_e4.py
$PY experiments/figures.py
(cd paper && latexmk -pdf -interaction=nonstopmode -quiet main.tex)
echo "done: paper/main.pdf"
