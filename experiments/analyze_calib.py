"""Items 3-4: leave-one-out Platt recalibration of Jev, and Jev as an extra feature for the descriptor model.
Paired bootstrap (2000 resamples of cases) for Brier differences."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
lg = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))

def loo(X, y, C=1.0):
    out = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        out[i] = LogisticRegression(C=C).fit(X[m], y[m]).predict_proba(X[i:i+1])[0, 1]
    return out
def ece(p, y, b=10):
    k = np.minimum((p*b).astype(int), b-1)
    return float(sum(abs(p[k == j].mean() - y[k == j].mean()) * np.mean(k == j) for j in range(b) if np.any(k == j)))
def brier_diff_ci(pa, pb, y):
    d = (pa - y)**2 - (pb - y)**2
    bs = [d[idx].mean() for idx in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def evaluate(p_jev, desc, y):
    y = y.astype(int); Z = lg(p_jev)[:, None]
    preds = {"jev_raw": p_jev, "jev_platt": loo(Z, y), "desc": loo(desc, y), "desc+jev": loo(np.hstack([desc, Z]), y)}
    res = {k: {"auroc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y)**2)), "ece": ece(p, y)} for k, p in preds.items()}
    res["d_brier_platt_minus_desc"] = brier_diff_ci(preds["jev_platt"], preds["desc"], y)
    res["d_brier_stack_minus_desc"] = brier_diff_ci(preds["desc+jev"], preds["desc"], y)
    res["d_brier_platt_minus_raw"] = brier_diff_ci(preds["jev_platt"], p_jev, y)
    return res

if __name__ == "__main__":
    r = json.load(open("experiments/e1e3_results.json")); rows = json.load(open("experiments/qc_labels.json"))
    Pm = np.array(r["Pm"]); S = ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"]
    desc = np.array([[x["stretch"], x["gap_eh"]] for x in rows])
    out = {}
    for k in ["mr", "t1", "unstable"]:
        out[k] = evaluate(Pm[:, S.index(k)], desc, np.array(r["labels"][k]))
        print(k, {m: (round(v["auroc"], 2), round(v["brier"], 3), round(v["ece"], 2)) for m, v in out[k].items() if isinstance(v, dict)})
        for d in ["d_brier_platt_minus_raw", "d_brier_platt_minus_desc", "d_brier_stack_minus_desc"]:
            print("   ", d, np.round(out[k][d], 3))
    json.dump(out, open("experiments/calib_results.json", "w"), indent=1)
