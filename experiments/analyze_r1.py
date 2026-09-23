"""Revisions after review round 1:
- per-call noise from the unbiased repeat variance
- matched coherent null per (case, pair): the joint implied by Jev's own p(b), p(a|b=1), p(a|b=0), measured with Jev's
  noise, rounding and 3 repeats; per-pair p-values with Benjamini-Hochberg
- molecule-clustered bootstrap CIs (12 molecules) for E1 fractions and E3 metrics
- leave-one-out base rate; stretch-only ranking; within-stretch AUROC
- maximum-likelihood inverse temperature beta for E2 (pre-registered, previously unreported)"""
import itertools, json
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(1)
S = ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"]; pairs = list(itertools.combinations(range(5), 2))
cl = lambda p: np.clip(p, .005, .995); lg = lambda p: np.log(cl(p) / (1 - cl(p)))
rows = json.load(open("experiments/qc_labels.json")); M = len(rows)
mol = np.array([r["name"] for r in rows]); mols = sorted(set(mol))
raw = json.load(open("experiments/e1e3_raw.json")); res = json.load(open("experiments/e1e3_results.json"))
g = defaultdict(list)
for x in raw:
    for q, p in x["p"].items(): g[(x["case"], str(x["cond"]), q)].append(p)
noise = float(np.sqrt(np.mean([np.var(v, ddof=1) for v in g.values() if len(v) == 3])))
m = {k: np.mean(v) for k, v in g.items()}
def P(i, cond, q): return m[(i, str(list(cond)) if cond else "None", q)]
out = {"noise_sd_unbiased": noise}

# matched null
obs_asym, pvals, case_of = [], [], []
def meas(p, n=3): return np.mean(np.round(np.clip(p + rng.normal(0, noise, (n,) + np.shape(p)), 0, 1), 2), 0)
for i in range(M):
    for a, b in pairs:
        A, B = S[a], S[b]
        pa1, pa0 = P(i, (B, True), A), P(i, (B, False), A); pb1, pb0 = P(i, (A, True), B), P(i, (A, False), B)
        o = abs((lg(pa1) - lg(pa0)) - (lg(pb1) - lg(pb0)))
        # coherent joint implied by p(b) (marginal) and p(a|b); its reverse conditionals p(b|a)
        pb = P(i, None, B); j11, j01 = cl(pa1)*pb, (1-cl(pa1))*pb; j10, j00 = cl(pa0)*(1-pb), (1-cl(pa0))*(1-pb)
        qb1, qb0 = j11/(j11+j10), j01/(j01+j00)
        sims = np.abs((lg(meas(np.full(4000, pa1))) - lg(meas(np.full(4000, pa0)))) - (lg(meas(np.full(4000, qb1))) - lg(meas(np.full(4000, qb0)))))
        obs_asym.append(o); pvals.append((np.sum(sims >= o) + 1) / (len(sims) + 1)); case_of.append(i)
pvals = np.array(pvals); obs_asym = np.array(obs_asym); case_of = np.array(case_of)
order = np.argsort(pvals); bh = np.zeros(len(pvals), bool); nP = len(pvals)
passed = [k for k in range(nP) if pvals[order[k]] <= 0.05 * (k + 1) / nP]
if passed: bh[order[:max(passed) + 1]] = True
def clus_ci(stat_fn, B=2000):
    vals = []
    for _ in range(B):
        pick = rng.choice(mols, len(mols)); idx = np.concatenate([np.where(mol == mm)[0] for mm in pick]); vals.append(stat_fn(idx))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
frac = lambda mask: (lambda idx: np.mean(np.concatenate([mask[case_of == i] for i in idx])))
out["e1_matched"] = {"frac_p05": float(np.mean(pvals < .05)), "frac_p05_ci": clus_ci(frac(pvals < .05)),
                     "frac_BH": float(bh.mean()), "frac_BH_ci": clus_ci(frac(bh)), "median_asym": float(np.median(obs_asym))}
k01 = [n for n, (a, b) in enumerate(pairs * M) if (a, b) == (0, 1)]
out["e1_matched"]["pvals"] = pvals.tolist()
out["e1_matched"]["mr_rhf_pair"] = {"median_asym": float(np.median(obs_asym[k01])), "frac_BH": float(bh[k01].mean())}
print("noise(unbiased)", round(noise, 3), "| matched null:", {k: (np.round(v, 2) if not isinstance(v, dict) else v) for k, v in out["e1_matched"].items()})

# E3 with LOO base rate, stretch-only ranking, clustered CIs, within-stretch
Pm = np.array(res["Pm"]); st = np.array([r["stretch"] for r in rows]); out["e3"] = {}
for k in ["mr", "t1", "unstable"]:
    y = np.array(res["labels"][k]).astype(int); p = Pm[:, S.index(k)]
    loo_base = np.array([np.mean(np.delete(y, i)) for i in range(M)])
    d = {"jev_auroc": roc_auc_score(y, p), "stretch_auroc": roc_auc_score(y, st),
         "jev_auroc_ci": clus_ci(lambda idx: roc_auc_score(y[idx], p[idx]) if 0 < y[idx].mean() < 1 else np.nan),
         "jev_minus_stretch_auroc_ci": clus_ci(lambda idx: roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], st[idx]) if 0 < y[idx].mean() < 1 else np.nan),
         "brier_jev": float(np.mean((p - y)**2)), "brier_loo_base": float(np.mean((loo_base - y)**2)),
         "within_stretch": {str(f): (float(y[st == f].mean()), roc_auc_score(y[st == f], p[st == f]) if 0 < y[st == f].mean() < 1 else None) for f in [1.0, 1.5, 2.0, 2.5]}}
    out["e3"][k] = d
    print(k, {kk: (np.round(v, 3) if not isinstance(v, dict) else v) for kk, v in d.items()})

# E2 beta by maximum likelihood: p(s) ~ exp(-beta E(s)) / Z, fitted per n on the mean distributions
e2 = json.load(open("experiments/e2_raw.json")); e2b = json.load(open("experiments/e2b_raw.json"))
def beta_fit(recs):
    def nll(b):
        t = 0.0
        for x in recs:
            E = np.array(list(x["E"].values())); p = np.array([x["p"].get(c, 0) for c in x["E"]]); p = p / p.sum()
            lw = -b * E; lz = np.log(np.sum(np.exp(lw - lw.max()))) + lw.max(); t -= np.sum(p * (lw - lz))
        return t
    return float(minimize_scalar(nll, bounds=(-20, 20), method="bounded").x)
out["e2_beta"] = {f"{name}_n{n}": beta_fit([x for x in recs if x["n"] == n]) for name, recs in (("stated", e2["ising"]), ("flipped", e2b)) for n in range(3, 8)}
print("beta ML:", {k: round(v, 2) for k, v in out["e2_beta"].items()})
json.dump(out, open("experiments/r1_results.json", "w"), indent=1)
