"""E5 re-analysed with exactly the E3 pipeline (leave-one-molecule-out models, molecule-clustered CIs)."""
import json, sys
import numpy as np
sys.path.insert(0, "experiments")
from analyze_e3v2 import run
lab = json.load(open("experiments/e5_labels_v2.json")); raw = json.load(open("experiments/e5_raw.json"))
rows = []
for r in lab:
    r = dict(r)
    if not r["rhf_converged"]: r["mr"] = r["t1_flag"] = r["rhf_unstable"] = None
    if r.get("cas_ss", 0) > 0.1: r["mr"] = None
    if not r["ccsd_converged"]: r["t1_flag"] = None
    if not np.isfinite(r["uhf_drop_eh"]): r["rhf_unstable"] = None
    rows.append(r)
M = len(rows)
P = {k: np.array([np.mean([x["p"][k] for x in raw["full"] if x["case"] == i]) for i in range(M)]) for k in ["mr", "t1", "unstable"]}
geo = {k: np.array([np.mean([x["p"][k] for x in raw["geo"] if x["case"] == i]) for i in range(M)]) for k in ["mr", "t1", "unstable"]}
json.dump(run(rows, P, {"jev_geometry_only": geo}), open("experiments/e5_lomo_results.json", "w"), indent=1)
