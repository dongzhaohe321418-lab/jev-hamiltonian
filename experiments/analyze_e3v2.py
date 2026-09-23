"""E3 on v2 labels with leave-one-molecule-out (LOMO) models and molecule-clustered bootstrap CIs.
Predictors: base rate, stretch factor (as stated in the prompt), logistic(stretch, gap), Jev raw, Jev Platt-recalibrated,
logistic(stretch, gap, logit Jev); Jev ablations: no stretch note, no name, geometry only.
Usage: analyze_e3v2.py [labels.json raw.json]  (defaults: the 48 original cases)."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(3)
cl = lambda p: np.clip(p, .005, .995); lg = lambda p: np.log(cl(p) / (1 - cl(p)))
def lomo(X, y, groups):
    out = np.zeros(len(y))
    for g in set(groups):
        te = groups == g; tr = ~te
        out[te] = (LogisticRegression().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] if len(set(y[tr])) > 1 else y[tr].mean())
    return out
def ece(p, y, b=10):
    k = np.minimum((p * b).astype(int), b - 1)
    return float(sum(abs(p[k == j].mean() - y[k == j].mean()) * np.mean(k == j) for j in range(b) if np.any(k == j)))
def metrics(p, y): return {"auroc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None, "brier": float(np.mean((p - y) ** 2)), "ece": ece(p, y)}
def clus(fn, groups, B=2000):
    gs = sorted(set(groups)); v = []
    for _ in range(B):
        idx = np.concatenate([np.where(groups == g)[0] for g in rng.choice(gs, len(gs))])
        try: v.append(fn(idx))
        except ValueError: pass
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def run(rows, P, ablations):
    groups = np.array([r["name"] for r in rows]); st = np.array([r["stretch"] for r in rows]); gap = np.array([r["gap_eh"] for r in rows])
    res = {}
    for k, labkey in [("mr", "mr"), ("t1", "t1_flag"), ("unstable", "rhf_unstable")]:
        ok = np.array([r[labkey] is not None for r in rows]); y = np.array([bool(r[labkey]) for r in rows])[ok].astype(int)
        g, s_, gp, pj = groups[ok], st[ok], gap[ok], P[k][ok]
        preds = {"base_rate": np.array([y[g != gg].mean() for gg in g]), "stretch": lomo(s_[:, None], y, g),
                 "logistic": lomo(np.c_[s_, gp], y, g), "jev": pj, "jev_platt": lomo(lg(pj)[:, None], y, g),
                 "logistic+jev": lomo(np.c_[s_, gp, lg(pj)], y, g), "stretch+jev": lomo(np.c_[s_, lg(pj)], y, g)}
        for a, Pa in ablations.items(): preds[a] = Pa[k][ok]
        r_ = {"n": int(ok.sum()), "pos_rate": float(y.mean())}
        for m, p in preds.items():
            r_[m] = metrics(p, y)
            r_[m]["auroc_ci"] = clus(lambda idx: roc_auc_score(y[idx], p[idx]) if 0 < y[idx].mean() < 1 else None, g) if m != "base_rate" else None
        for m in ["jev_platt", "logistic+jev", "jev"]:
            d = (preds[m] - y) ** 2 - (preds["logistic"] - y) ** 2
            r_[f"dbrier_{m}_vs_logistic"] = [float(d.mean())] + clus(lambda idx: d[idx].mean(), g)
        r_["stretch_raw"] = {"auroc": float(roc_auc_score(y, s_)), "auroc_ci": clus(lambda idx: roc_auc_score(y[idx], s_[idx]) if 0 < y[idx].mean() < 1 else None, g)}
        ll = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        dl = ll(preds["stretch+jev"]) - ll(preds["stretch"])
        r_["dlogloss_stretch+jev_vs_stretch"] = [float(dl.mean())] + clus(lambda idx: dl[idx].mean(), g)
        d = roc_auc_score(y, pj) - roc_auc_score(y, s_)
        r_["dauroc_jev_vs_stretch"] = [float(d)] + clus(lambda idx: roc_auc_score(y[idx], pj[idx]) - roc_auc_score(y[idx], s_[idx]) if 0 < y[idx].mean() < 1 else None, g)
        r_["within_stretch_auroc"] = {str(f): (float(roc_auc_score(y[s_ == f], pj[s_ == f])) if 0 < y[s_ == f].mean() < 1 else None) for f in sorted(set(s_))}
        res[k] = r_
        print(f"{k:9s} n={r_['n']} pos={r_['pos_rate']:.2f}  " + "  ".join(f"{m}: {v['auroc'] if v['auroc'] is None else round(v['auroc'],2)}/{v['brier']:.3f}" for m, v in r_.items() if isinstance(v, dict) and "brier" in v))
        print(f"           dAUROC jev-stretch {np.round(r_['dauroc_jev_vs_stretch'],2)}  dBrier platt-log {np.round(r_['dbrier_jev_platt_vs_logistic'],3)}  stack-log {np.round(r_['dbrier_logistic+jev_vs_logistic'],3)}  within-stretch {r_['within_stretch_auroc']}")
        print(f"           raw stretch AUROC {r_['stretch_raw']['auroc']:.2f} {np.round(r_['stretch_raw']['auroc_ci'],2)}; dlogloss stretch+jev vs stretch {np.round(r_['dlogloss_stretch+jev_vs_stretch'],3)}")
    return res

if __name__ == "__main__":
    rows = json.load(open("experiments/qc_labels_v2.json")); res0 = json.load(open("experiments/e1e3_results.json"))
    S = ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"]; Pm = np.array(res0["Pm"])
    P = {k: Pm[:, S.index(k)] for k in ["mr", "t1", "unstable"]}
    e9 = json.load(open("experiments/e9_raw.json")); e3b = json.load(open("experiments/e1b_e3b_raw.json"))["geo"]
    def mean_by_case(recs, k): return np.array([np.mean([x["p"][k] for x in recs if x["case"] == i]) for i in range(len(rows))])
    abl = {"jev_no_stretch_note": {k: mean_by_case([x for x in e9 if x["kind"] == "e9c"], k) for k in P},
           "jev_no_name": {k: mean_by_case([x for x in e9 if x["kind"] == "e9d"], k) for k in P},
           "jev_geometry_only": {k: np.array([np.mean([x[2][k] for x in e3b if x[0] == i]) for i in range(len(rows))]) for k in P}}
    json.dump(run(rows, P, abl), open("experiments/e3v2_results.json", "w"), indent=1)
