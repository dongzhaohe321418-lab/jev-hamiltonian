"""Round-2 N5: does a symmetric pairwise Ising energy predict held-out two-fact conditionals better than an asymmetric
additive-logit model (a dependency network, J_ab != J_ba) built from the same single-fact data?
Asymmetric additive: logit p(x | y=v, z=w) = logit p(x) + [logit p(x|y=v) - logit p(x)] + [logit p(x|z=w) - logit p(x)].
Also an asymmetric model with the same fitted form as the Ising one (per-target bias + per-(target,source) weights)."""
import json, sys
import numpy as np
sys.argv = ["x"]
exec(open("experiments/analyze_e8.py").read().split("# ---- E1c")[0])   # loads D (means), S, helpers
out = {"ising": [], "asym_additive": [], "ignore_second": []}
import itertools
from scipy.optimize import least_squares
st5 = np.array(list(itertools.product([0, 1], repeat=K))); pairs = list(itertools.combinations(range(K), 2))
F = np.hstack([st5, np.array([[s[a] * s[b] for a, b in pairs] for s in st5])])
def model_cond(theta, x, facts):
    e = F @ theta; w = np.exp(e - e.max()); m = np.ones(len(st5), bool)
    for y, v in facts: m &= st5[:, S.index(y)] == int(v)
    return w[m & (st5[:, S.index(x)] == 1)].sum() / w[m].sum()
for i in range(M):
    Di = D[i]
    one = [(x, f, p) for f, d in Di.items() if len(f) <= 1 for x, p in d.items()]
    two = [(x, f, p) for f, d in Di.items() if len(f) == 2 for x, p in d.items()]
    th = least_squares(lambda t: np.array([lg(model_cond(t, x, f)) - lg(p) for x, f, p in one]), np.zeros(F.shape[1]), method="lm").x
    e_is, e_as, e_ig = [], [], []
    for x, f, p in two:
        base = lg(Di[()][x]); d1 = lg(Di[(f[0],)][x]) - base; d2 = lg(Di[(f[1],)][x]) - base
        e_is.append((lg(model_cond(th, x, f)) - lg(p)) ** 2); e_as.append((base + d1 + d2 - lg(p)) ** 2); e_ig.append((base + d1 - lg(p)) ** 2)
    out["ising"].append(float(np.sqrt(np.mean(e_is)))); out["asym_additive"].append(float(np.sqrt(np.mean(e_as)))); out["ignore_second"].append(float(np.sqrt(np.mean(e_ig))))
for k, v in out.items(): print(k, "median RMS", round(np.median(v), 3))
d = np.array(out["ising"]) - np.array(out["asym_additive"])
print("ising better than asym-additive in", int(np.sum(d < 0)), "of", M, "cases; median diff", round(np.median(d), 3))
from scipy.stats import wilcoxon; print("Wilcoxon p", wilcoxon(d).pvalue)
out["wilcoxon_p"] = float(wilcoxon(d).pvalue); out["n_ising_better"] = int(np.sum(d < 0))
json.dump(out, open("experiments/n5_results.json", "w"), indent=1)
