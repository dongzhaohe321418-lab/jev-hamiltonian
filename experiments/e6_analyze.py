"""E6 analysis: workflow routing between CCSD(T) and an (assumed exact) multireference method. Writes e6_results.json.
A policy routes a case to the expensive method (final error 0 by assumption) or keeps CCSD(T) (final error
|E_CCSD(T) - E_FCI|). Reported: fraction routed, fraction of cases whose final |error| > 1.6 mEh, mean final |error|."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
rows_all = json.load(open("experiments/e6_labels.json"))
raw = json.load(open("experiments/e6_raw.json"))["answers"]
TH = 1.6e-3
key = lambda r: (r["name"], r["basis"], r["stretch"])
pj = {}
for x in raw: pj.setdefault(key(x), []).append(x["p"])
# exclusion: RHF or CCSD not converged (flag set in e6_labels), or FCI root not a clean singlet / not converged
drop = lambda r: r["exclude"] or r["fci_s2"] > 0.01 or not r["fci_converged"]
excluded = [r for r in rows_all if drop(r)]
rows = [r for r in rows_all if not drop(r)]
n = len(rows)
err = np.array([r["abs_err_eh"] for r in rows]); bad = err > TH
p = np.array([np.mean(pj[key(r)]) for r in rows]); p_sd = np.array([np.std(pj[key(r)]) for r in rows])
t1 = np.array([r["t1"] for r in rows]); X = np.array([[r["stretch"], r["gap_eh"]] for r in rows])
def loo(X, y, groups=None):
    """Leave-one-molecule-out (all bases and stretches of a molecule held out together; round-2 review, Codex C3)."""
    groups = np.array([r["name"] for r in rows]) if groups is None else groups
    out = np.zeros(len(y))
    for g in set(groups):
        m = groups != g
        out[~m] = LogisticRegression().fit(X[m], y[m]).predict_proba(X[~m])[:, 1] if len(set(y[m])) > 1 else y[m].mean()
    return out
p_lr = loo(X, (~bad).astype(int))          # P(CCSD(T) reliable)

def policy(route):
    route = np.asarray(route, bool); fin = np.where(route, 0.0, err)
    return {"frac_routed": float(route.mean()), "frac_err_gt_1p6": float(np.mean(fin > TH)),
            "mean_abs_err_mEh": float(fin.mean() * 1000), "n_routed": int(route.sum()), "n_fail": int(np.sum(fin > TH))}
def curve(score):  # route if P(ok) < tau, tau swept over all distinct values (+ endpoints)
    taus = np.unique(np.r_[0.0, score, 1.01])
    return [{"tau": float(t), **policy(score < t)} for t in taus]
def auc_area(c):  # area under (frac_routed, frac_fail) curve; lower is better
    xs = np.array([d["frac_routed"] for d in c]); ys = np.array([d["frac_err_gt_1p6"] for d in c]); o = np.argsort(xs, kind="stable")
    return float(np.trapezoid(ys[o], xs[o]))
cj, cl = curve(p), curve(p_lr)
# budget-matched comparison: at the routing fraction used by the T1 rule, residual failure of Jev / logistic
t1_pol = policy(t1 > 0.02)
def at_budget(score, frac):
    k = int(round(frac * n)); o = np.argsort(score, kind="stable")  # route the k lowest P(ok)
    r = np.zeros(n, bool); r[o[:k]] = True; return policy(r)
boot = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    if len(set(bad[i])) > 1: boot.append(roc_auc_score(bad[i], -p[i]))
dboot = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    if len(set(bad[i])) > 1: dboot.append(roc_auc_score(bad[i], -p[i]) - roc_auc_score(bad[i], t1[i]))
res = {"auroc_diff_jev_minus_t1_ci95": [float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))], "n_cases_total": len(rows_all), "n_used": n,
       "excluded": [{"case": f"{r['name']}/{r['basis']} x{r['stretch']}", "rhf_converged": r["rhf_converged"], "ccsd_converged": r["ccsd_converged"], "fci_s2": r["fci_s2"], "fci_converged": r["fci_converged"]} for r in excluded],
       "cases": [f"{r['name']}/{r['basis']} x{r['stretch']}" for r in rows],
       "fci_nonsinglet_roots": [f"{r['name']}/{r['basis']} x{r['stretch']} S2={r['fci_s2']:.3f}" for r in rows_all if r["fci_s2"] > 0.01],
       "n_ccsdt_fail": int(bad.sum()), "frac_ccsdt_fail": float(bad.mean()),
       "abs_err_mEh": {"median": float(np.median(err)*1000), "mean": float(err.mean()*1000), "max": float(err.max()*1000)},
       "by_basis": {b: {"n": int(sum(r["basis"] == b for r in rows)), "fail": int(sum(bad[i] for i, r in enumerate(rows) if r["basis"] == b))} for b in ("sto-3g", "6-31g")},
       "by_stretch": {str(f): {"n": int(np.sum(X[:, 0] == f)), "fail": int(bad[X[:, 0] == f].sum())} for f in (1.0, 1.5, 2.0, 2.5)},
       "discrimination_auroc_for_failure": {"jev(1-p)": float(roc_auc_score(bad, -p)), "jev_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                              "t1": float(roc_auc_score(bad, t1)), "logreg_stretch_gap_loo": float(roc_auc_score(bad, -p_lr)),
                                              "stretch": float(roc_auc_score(bad, X[:, 0]))},
       "jev_calibration": {"mean_p_ok": float(p.mean()), "true_ok_rate": float(1 - bad.mean()), "brier": float(np.mean((p - (~bad))**2)),
                            "brier_base_rate": float(np.mean(((1 - bad.mean()) - (~bad))**2)), "rep_sd_mean": float(p_sd.mean())},
       "policies": {"always_ccsdt": policy(np.zeros(n)), "always_multireference": policy(np.ones(n)),
                    "oracle": policy(bad), "jev_tau0.5": policy(p < 0.5), "logreg_tau0.5": policy(p_lr < 0.5),
                    "t1_gt_0.02": t1_pol, "stretch_gt_1.0": policy(X[:, 0] > 1.0),
                    "jev_at_t1_budget": at_budget(p, t1_pol["frac_routed"]), "logreg_at_t1_budget": at_budget(p_lr, t1_pol["frac_routed"])},
       "curve_area_lower_is_better": {"jev": auc_area(cj), "logreg": auc_area(cl), "random_expected": float(bad.mean() / 2)},
       "curves": {"jev": cj, "logreg": cl},
       "per_case": [{"case": f"{r['name']}/{r['basis']} x{r['stretch']}", "abs_err_mEh": float(e*1000), "p_jev": float(pp), "p_logreg": float(pl), "t1": float(tt)}
                    for r, e, pp, pl, tt in zip(rows, err, p, p_lr, t1)],
       "timing_s": {"ccsdt_median": float(np.median([r["t_ccsdt_s"] for r in rows_all])), "fci_median": float(np.median([r["t_fci_s"] for r in rows_all]))}}

# ---- sensitivity analyses ----
def summarize(sub, errs):
    e = np.asarray(errs); b = e > TH; pp = np.array([np.mean(pj[key(r)]) for r in sub]); tt = np.array([r["t1"] for r in sub])
    Xs = np.array([[r["stretch"], r["gap_eh"]] for r in sub]); pl = loo(Xs, (~b).astype(int), np.array([r["name"] for r in sub]))
    def pol(route):
        route = np.asarray(route, bool); fin = np.where(route, 0.0, e)
        return {"frac_routed": float(route.mean()), "frac_err_gt_1p6": float(np.mean(fin > TH))}
    def bud(score, frac):
        k = int(round(frac * len(sub))); o = np.argsort(score, kind="stable"); r_ = np.zeros(len(sub), bool); r_[o[:k]] = True; return pol(r_)
    def area(score):
        taus = np.unique(np.r_[0.0, score, 1.01]); c = [pol(score < t) for t in taus]
        xs = np.array([d["frac_routed"] for d in c]); ys = np.array([d["frac_err_gt_1p6"] for d in c]); o = np.argsort(xs, kind="stable")
        return float(np.trapezoid(ys[o], xs[o]))
    t1p = pol(tt > 0.02)
    return {"n": len(sub), "n_fail": int(b.sum()), "frac_fail": float(b.mean()),
            "auroc_failure": {"jev": float(roc_auc_score(b, -pp)), "t1": float(roc_auc_score(b, tt)), "logreg": float(roc_auc_score(b, -pl))},
            "t1_rule": t1p, "jev_at_t1_budget": bud(pp, t1p["frac_routed"]), "logreg_at_t1_budget": bud(pl, t1p["frac_routed"]),
            "jev_tau0.5": pol(pp < 0.5), "curve_area": {"jev": area(pp), "logreg": area(pl), "random": float(b.mean() / 2)}}
nvirt = lambda r: r["nao"] - r["nelectron"] // 2
nontriv = [r for r in rows if nvirt(r) > 1]
res["sensitivity"] = {
    "exclude_cases_with_at_most_1_virtual_orbital": {"dropped": [f"{r['name']}/{r['basis']} x{r['stretch']}" for r in rows if nvirt(r) <= 1],
                                                       **summarize(nontriv, [r["abs_err_eh"] for r in nontriv])}}
inc = [r for r in rows_all if r["rhf_converged"] and r["fci_converged"]]
res["sensitivity"]["include_ccsd_nonconverged_as_failures"] = {
    "added": [f"{r['name']}/{r['basis']} x{r['stretch']}" for r in inc if not r["ccsd_converged"]],
    **summarize(inc, [r["abs_err_eh"] if r["ccsd_converged"] else np.inf for r in inc])}
print(json.dumps(res["sensitivity"], indent=1))
json.dump(res, open("experiments/e6_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("curves", "per_case", "cases")}, indent=1))
