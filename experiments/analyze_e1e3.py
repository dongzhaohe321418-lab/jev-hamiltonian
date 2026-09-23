"""Analysis for E1 (conditional compatibility) and E3 (QC decisions)."""
import json, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
raw = json.load(open("experiments/e1e3_raw.json")); rows = json.load(open("experiments/qc_labels.json"))
S = ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"]; M = len(rows); R = 3
clip = lambda p: np.clip(p, 0.005, 0.995); logit = lambda p: np.log(clip(p)/(1-clip(p)))

# P[case, rep, cond_idx, var]; cond_idx 0 = marginal, then (k,True),(k,False) per k
conds = [None] + [(k, v) for k in S for v in (True, False)]
P = np.full((M, R, len(conds), len(S)), np.nan)
for x in raw:
    c = conds.index(tuple(x["cond"]) if x["cond"] else None)
    for k, p in x["p"].items(): P[x["case"], x["rep"], c, S.index(k)] = p
ci = lambda k, v: conds.index((k, v))

def stats(P):
    """Per (case, ordered pair) couplings and marginal-consistency residuals, averaging reps first."""
    Pm = np.nanmean(P, 1)
    J, resid = {}, []
    for a, b in itertools.permutations(range(len(S)), 2):
        ka, kb = S[a], S[b]
        J[(a, b)] = logit(Pm[:, ci(kb, True), a]) - logit(Pm[:, ci(kb, False), a])   # effect of b on a
        pb = Pm[:, 0, b]
        resid.append(Pm[:, 0, a] - (Pm[:, ci(kb, True), a]*pb + Pm[:, ci(kb, False), a]*(1-pb)))
    asym = np.array([J[(a, b)] - J[(b, a)] for a, b in itertools.combinations(range(len(S)), 2)])  # pairs x cases
    return J, asym, np.array(resid)

J, asym, resid = stats(P)
# noise floor: resample repeats with replacement per (case, cond) and recompute asymmetry
boot = []
for _ in range(2000):
    # each (case, condition, statement) series comes from separate API calls, so resample repeats independently per series
    idx = rng.integers(0, R, size=P.shape)
    Pb = np.take_along_axis(P, idx, 1)
    boot.append(stats(Pb)[1])
se = np.std(boot, 0)
z = np.abs(asym) / np.maximum(se, 1e-6)
print(f"E1 pairs x cases: {asym.size}; median |J_ab-J_ba| = {np.median(np.abs(asym)):.2f} nats (median |J| = {np.median(np.abs(np.array(list(J.values())))):.2f})")
print(f"   fraction asymmetry > 3 bootstrap SE: {np.mean(z>3):.2f}   (> 2 SE: {np.mean(z>2):.2f})")
print(f"   marginal-consistency |resid| median {np.median(np.abs(resid)):.3f}, 90th pct {np.percentile(np.abs(resid),90):.3f}")
# logically forced couplings: mr <-> rhf_ok should be strongly negative and symmetric
for a, b in [("mr", "rhf_ok"), ("t1", "ccsdt_ok"), ("mr", "unstable")]:
    A, B = S.index(a), S.index(b)
    print(f"   J({a}|{b}) = {np.mean(J[(A,B)]):+.2f}   J({b}|{a}) = {np.mean(J[(B,A)]):+.2f}")

# coherent null: exact pairwise Ising joint, same rounding + observed noise sd -> asymmetry distribution
noise_sd = np.nanmean(np.nanstd(P, 1))
null = []
for _ in range(2000):
    h = rng.normal(0, 1, 2); Jc = rng.normal(0, 1.5)
    st = np.array(list(itertools.product([0, 1], repeat=2)))
    w = np.exp(st@h + Jc*st[:, 0]*st[:, 1]); w /= w.sum()
    def cond(a, b, v):  # p(x_a=1 | x_b=v)
        m = st[:, b] == v; return w[m & (st[:, a] == 1)].sum()/w[m].sum()
    obs = lambda p: np.round(np.clip(p + rng.normal(0, noise_sd, R), 0, 1), 2).mean()
    Jab = logit(obs(cond(0, 1, 1))) - logit(obs(cond(0, 1, 0))); Jba = logit(obs(cond(1, 0, 1))) - logit(obs(cond(1, 0, 0)))
    null.append(Jab - Jba)
null = np.abs(null)
print(f"   noise sd per call {noise_sd:.3f}; coherent-null |asym| median {np.median(null):.2f}, 95th pct {np.percentile(null,95):.2f}; "
      f"Jev pairs above null 95th pct: {np.mean(np.abs(asym) > np.percentile(null,95)):.2f}")

# E3
Pm = np.nanmean(P, 1)[:, 0, :]
labels = {"mr": np.array([r["mr"] for r in rows]), "t1": np.array([r["t1_flag"] for r in rows]),
          "unstable": np.array([r["rhf_unstable"] for r in rows])}
X = np.array([[r["stretch"], r["gap_eh"]] for r in rows]); Xs = X[:, :1]
def loo(X, y):
    out = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        out[i] = LogisticRegression().fit(X[m], y[m]).predict_proba(X[i:i+1])[0, 1]
    return out
def ece(p, y, b=10):
    bins = np.minimum((p*b).astype(int), b-1)
    return sum(np.abs(p[bins == k].mean() - y[bins == k].mean())*np.mean(bins == k) for k in range(b) if np.any(bins == k))
e3 = {}
for k, y in labels.items():
    y = y.astype(int); preds = {"jev": Pm[:, S.index(k)], "base_rate": np.full(len(y), y.mean()),
                                "logreg_stretch": loo(Xs, y), "logreg_stretch_gap": loo(X, y)}
    e3[k] = {}
    for n, p in preds.items():
        auc = roc_auc_score(y, p) if n != "base_rate" else 0.5
        e3[k][n] = {"auroc": auc, "brier": float(np.mean((p-y)**2)), "ece": float(ece(p, y))}
    print(f"E3 {k:9s} (pos rate {y.mean():.2f}) " + "  ".join(f"{n}: AUC {v['auroc']:.2f} Brier {v['brier']:.3f} ECE {v['ece']:.3f}" for n, v in e3[k].items()))
pairsJ = [[J[(a, b)].tolist(), J[(b, a)].tolist()] for a, b in itertools.combinations(range(len(S)), 2)]
json.dump({"J_pairs": pairsJ, "asym": asym.tolist(), "se": se.tolist(), "prereg_frac_gt_3se": float(np.mean(z > 3)), "null_abs": null.tolist(), "resid": resid.tolist(),
           "J_mean": {f"{S[a]}|{S[b]}": float(np.mean(v)) for (a, b), v in J.items()}, "e3": e3,
           "Pm": Pm.tolist(), "labels": {k: v.tolist() for k, v in labels.items()}, "noise_sd": float(noise_sd)},
          open("experiments/e1e3_results.json", "w"), indent=1)
