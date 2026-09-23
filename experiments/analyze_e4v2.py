"""E4 v2 summary: log-error area over the whole curve, paired Wilcoxon, question-aligned metrics, template similarity."""
import itertools, json
import numpy as np
from scipy.stats import wilcoxon, spearmanr
d = json.load(open("experiments/e4v2_raw.json"))
M = ["oracle", "cipsi", "en_pt2", "first_order", "jev_top1", "jev_noul", "excitation", "random"]
area = {m: np.array([np.mean(np.log10(np.maximum(s["curves"][m], 1e-3))) for s in d]) for m in M}
out = {"area": {m: v.tolist() for m, v in area.items()}, "tests": {}}
for m in M: print(f"{m:12s} mean log10-error area {area[m].mean():+.3f}")
for j in ["jev_top1", "jev_noul"]:
    for b in ["random", "excitation", "en_pt2", "first_order", "cipsi"]:
        diff = area[j] - area[b]; p = wilcoxon(diff).pvalue
        out["tests"][f"{j}-{b}"] = [float(diff.mean()), int(np.sum(diff < 0)), float(p)]
        print(f"{j} - {b:11s}: {diff.mean():+.3f}  better in {np.sum(diff<0)}/10  p={p:.3f}")
out["top1_match"] = int(sum(s["top1_match"] for s in d)); out["oracle_rank"] = [s["oracle_rank_under_jev"] for s in d]
out["n_families"] = [s["n_families"] for s in d]; out["jev_top_labels"] = [s["jev_top_label"] for s in d]
out["rho_top1"] = [s["rho_top1_c2"] for s in d]; out["rho_noul"] = [s["rho_noul_c2"] for s in d]
same = [s for s in d if s["cas"] == "(6e,5o)"]
sims = [spearmanr(a["top1"], b["top1"]).statistic for a, b in itertools.combinations(same, 2)]
simc = [spearmanr(np.array(json.load(open("experiments/e4v2_raw.json"))[0]["top1"]) * 0 + 1, [0]).statistic] if False else None
# exact weights across the same systems, for contrast
out["template_similarity_jev"] = [float(np.median(sims)), float(min(sims)), len(same)]
print("top1 match", out["top1_match"], "/10; oracle rank under Jev", out["oracle_rank"], "of", out["n_families"])
print("Jev top-1 labels", out["jev_top_labels"])
print(f"cross-system Spearman of Jev top1 distributions over {len(same)} CAS(6e,5o) systems: median {np.median(sims):.2f}, min {min(sims):.2f}")
json.dump(out, open("experiments/e4v2_results.json", "w"), indent=1)
