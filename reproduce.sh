#!/usr/bin/env bash
# Rebuild every number, table, figure and the PDF from the cached API responses and saved labels.
# JEV_OFFLINE=1 makes any missing cache entry an error instead of a (billed) API call.
# Pass --full to also recompute PySCF-heavy steps (v2 labels, E4 v2 selected CI); these take tens of minutes.
set -euo pipefail
cd "$(dirname "$0")"
export JEV_OFFLINE=1
PY=.venv/bin/python
[ -x "$PY" ] || { python3 -m venv .venv; .venv/bin/pip install -q -r requirements.txt; }
if [ "${1:-}" = "--full" ]; then
  $PY experiments/qc_labels_v2.py
  $PY experiments/e4v2_selected_ci.py
  $PY experiments/e2_hamiltonian.py
fi
for s in analyze_e1e3 analyze_r1 analyze_e2 analyze_e8 analyze_n5 analyze_e9 analyze_e3v2 analyze_e5_lomo analyze_e4v2 analyze_calib make_tables e5_analyze e6_analyze e7_analyze figures e6_figure e7_figure; do
  echo "== $s"; $PY experiments/$s.py
done
(cd paper && latexmk -pdf -interaction=nonstopmode -quiet main.tex)
echo "done: paper/main.pdf"
