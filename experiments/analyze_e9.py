"""E9: positive control (stated population), fact uptake, and E3 ablations (no stretch note / no name)."""
import itertools, json
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
cl = lambda p: np.clip(p, .005, .995); lg = lambda p: np.log(cl(p) / (1 - cl(p)))
d = json.load(open("experiments/e9_raw.json")); out = {}
# positive control
A = ["A", "B", "C"]; idx = {"A": 0, "B": 1, "C": 2}
g = defaultdict(list); counts = {}
for x in d:
    if x["kind"] != "e9a": continue
    counts[x["case"]] = x["counts"]
    for q, p in x["p"].items(): g[(x["case"], tuple(x["cond"]) if x["cond"] else None, q)].append(p)
m = {k: np.mean(v) for k, v in g.items()}
def exact(c, q, cond):
    num = den = 0
    for k, n in c.items():
        if cond and (k[idx[cond[0]]] == "1") != cond[1]: continue
        den += n; num += n * (k[idx[q]] == "1")
    return num / den
asym_j, asym_x, err, pcJ = [], [], [], []
for t, c in counts.items():
    for a, b in itertools.combinations(A, 2):
        J = lambda f, x, y: lg(f(x, (y, True))) - lg(f(x, (y, False)))
        fj = lambda x, cond: m[(t, cond, x)]; fx = lambda x, cond: exact(c, x, cond)
        asym_j.append(float(abs(J(fj, a, b) - J(fj, b, a)))); asym_x.append(abs(J(fx, a, b) - J(fx, b, a)))
        pcJ.append([float(J(fj, a, b)), float(J(fj, b, a))])
    for (tt, cond, q), p in m.items():
        if tt == t: err.append(abs(p - exact(c, q, cond)))
out["pos_control_asym"] = asym_j; out["pos_control_J"] = pcJ
out["pos_control"] = {"jev_median_asym": float(np.median(asym_j)), "exact_median_asym": float(np.median(asym_x)),
                      "frac_asym_gt_null95": float(np.mean(np.array(asym_j) > 0.36)), "mae_prob_vs_exact": float(np.mean(err)),
                      "median_abs_err": float(np.median(err))}
print("positive control:", {k: round(v, 3) for k, v in out["pos_control"].items()})
# fact uptake
up = defaultdict(list)
for x in d:
    if x["kind"] == "e9b":
        k, v = x["cond"]; up[(k, v)].append(x["p"][k])
out["uptake"] = {f"{k}_{v}": float(np.mean(p)) for (k, v), p in up.items()}
print("fact uptake (mean p(s) when s stated TRUE / FALSE):", {k: round(v, 2) for k, v in out["uptake"].items()})
# E3 ablations
rows = json.load(open("experiments/qc_labels_v2.json"))
lab = {"mr": [r["mr"] for r in rows], "t1": [r["t1_flag"] for r in rows], "unstable": [r["rhf_unstable"] for r in rows]}
for kind, name in [("e9c", "name+geometry, no stretch note"), ("e9d", "stretch note+geometry, no name")]:
    out[kind] = {}
    for k, y in lab.items():
        p = np.array([np.mean([x["p"][k] for x in d if x["kind"] == kind and x["case"] == i]) for i in range(len(rows))])
        ok = np.array([v is not None for v in y]); yy = np.array([bool(v) for v in y])[ok].astype(int)
        out[kind][k] = [float(roc_auc_score(yy, p[ok])), float(np.mean((p[ok] - yy) ** 2))]
    print(name, {k: np.round(v, 3) for k, v in out[kind].items()})
json.dump(out, open("experiments/e9_results.json", "w"), indent=1)
