"""E4 summary: area under the log10 error curve (lower is better), paired across systems."""
import json
import numpy as np
from scipy.stats import wilcoxon
d = json.load(open("experiments/e4_raw.json"))
M = ["oracle", "pt1", "excitation", "jev", "diag", "random"]
area = {m: np.array([np.mean(np.log10(np.maximum(s["curves"][m], 1e-3))) for s in d]) for m in M}
for m in M: print(f"{m:10s} mean log10-error area {area[m].mean():+.3f}")
for b in ["random", "excitation", "pt1", "diag"]:
    diff = area["jev"] - area[b]
    print(f"jev - {b:10s}: {diff.mean():+.3f}  (jev better in {np.sum(diff<0)}/10, Wilcoxon p={wilcoxon(diff).pvalue:.3f})")
zeros = [np.mean([np.mean([v == 0 for v in p.values()]) for p in s["jev_probs"]]) for s in d]
ties = [len(set(np.round(np.mean([[p.get(k, 0) for k in s["jev_probs"][0]] for p in s["jev_probs"]], 0), 4))) for s in d]
print("fraction of 99 options Jev returns as exactly 0:", np.round(zeros, 2), "distinct mean-prob values:", ties)
json.dump({"area": {m: v.tolist() for m, v in area.items()}, "zero_frac": zeros}, open("experiments/e4_results.json", "w"), indent=1)
