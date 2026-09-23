"""E5 analysis: closed-shell labels (protocol v2) vs Jev, and the spin-state task. Writes experiments/e5_results.json."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
lab = json.load(open("experiments/e5_labels_v2.json"))
v1 = json.load(open("experiments/e5_labels.json"))
raw = json.load(open("experiments/e5_raw.json"))
N = len(lab); R = 3
assert [(a["name"], a["stretch"]) for a in lab] == [(b["name"], b["stretch"]) for b in v1]

def mean_p(rows, keys):
    P = {k: np.full((N, R), np.nan) for k in keys}
    for x in rows:
        for k in keys: P[k][x["case"], x["rep"]] = x["p"][k]
    return {k: np.nanmean(v, 1) for k, v in P.items()}, {k: v for k, v in P.items()}
Pf, Pf_rep = mean_p(raw["full"], ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"])
Pg, _ = mean_p(raw["geo"], ["mr", "t1", "unstable"])

def ece(p, y, b=10):
    bins = np.minimum((p*b).astype(int), b-1)
    return float(sum(np.abs(p[bins == k].mean() - y[bins == k].mean())*np.mean(bins == k) for k in range(b) if np.any(bins == k)))
def loo(X, y):
    out = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        out[i] = LogisticRegression().fit(X[m], y[m]).predict_proba(X[i:i+1])[0, 1] if len(set(y[m])) > 1 else y[m].mean()
    return out
def boot_auc(y, p, n=2000):
    a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: a.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
def boot_diff(y, p, q, n=2000):
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: d.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], q[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

stretch = np.array([r["stretch"] for r in lab]); gap = np.array([r["gap_eh"] for r in lab])
rhf_ok = np.array([r["rhf_converged"] for r in lab]); split = np.array([r["cas_split_degenerate"] for r in lab])
t1c = np.array([r["t1"] is not None for r in lab])
cas_singlet = np.array([r["cas_ss"] <= 0.1 for r in lab])          # CASCI root is a singlet despite fix_spin_
uhf_ok = np.array([np.isfinite(r["uhf_drop_eh"]) for r in lab])    # at least one UHF start converged
LABELS = {  # key -> (jev statement key, label array, inclusion mask, description)
    "mr": ("mr", np.array([r["mr"] for r in lab]), rhf_ok & ~split & cas_singlet, "c_HF^2<0.9, full-valence CASCI; excl. RHF non-conv., CAS-split-degenerate, non-singlet CASCI root"),
    "mr_incl_split": ("mr", np.array([r["mr"] for r in lab]), rhf_ok & cas_singlet, "as mr but including CAS-split-degenerate cases"),
    "t1": ("t1", np.array([bool(r["t1_flag"]) for r in lab]), rhf_ok & t1c, "frozen-core CCSD T1>0.02; excl. unconverged CCSD"),
    "unstable": ("unstable", np.array([r["rhf_unstable"] for r in lab]), rhf_ok & uhf_ok, "UHF drop > 1 mEh (primary); excl. no converged UHF"),
    "unstable_0p1": ("unstable", np.array([r["rhf_unstable_0p1"] for r in lab]), rhf_ok & uhf_ok, "UHF drop > 0.1 mEh (E3 v1 threshold)"),
}
res = {"n_cases": N, "cases": [f"{r['name']} x{r['stretch']}" for r in lab],
       "excluded_rhf_nonconverged": [f"{r['name']} x{r['stretch']}" for r in lab if not r["rhf_converged"]],
       "cas_split_degenerate": [f"{r['name']} x{r['stretch']}" for r in lab if r["cas_split_degenerate"]],
       "ccsd_nonconverged": [f"{r['name']} x{r['stretch']}" for r in lab if r["t1"] is None],
       "cas_nonsinglet_root": [f"{r['name']} x{r['stretch']} S2={r['cas_ss']:.2f}" for r in lab if r["cas_ss"] > 0.1],
       "uhf_no_converged_start": [f"{r['name']} x{r['stretch']}" for r in lab if not np.isfinite(r["uhf_drop_eh"])],
       "labels": {}}
for key, (jk, y, m, desc) in LABELS.items():
    y = y[m].astype(int); X = np.c_[stretch, gap][m]
    preds = {"jev": Pf[jk][m], "jev_geometry_only": Pg[jk][m], "base_rate": np.full(m.sum(), y.mean()),
             "logreg_stretch": loo(X[:, :1], y), "logreg_stretch_gap": loo(X, y)}
    d = {"description": desc, "n": int(m.sum()), "n_pos": int(y.sum()), "pos_rate": float(y.mean()), "methods": {}}
    for n_, p in preds.items():
        auc = float(roc_auc_score(y, p)) if n_ != "base_rate" else 0.5
        d["methods"][n_] = {"auroc": auc, "auroc_ci95": boot_auc(y, p) if n_ != "base_rate" else None,
                            "brier": float(np.mean((p - y)**2)), "ece": ece(p, y), "mean_p": float(p.mean())}
    d["auroc_diff_jev_minus_logreg_stretch_gap_ci95"] = boot_diff(y, preds["jev"], preds["logreg_stretch_gap"])
    d["auroc_diff_jev_minus_geometry_only_ci95"] = boot_diff(y, preds["jev"], preds["jev_geometry_only"])
    # within-stretch discrimination (does Jev rank beyond the stretch factor?)
    ws = {}
    for f in (1.0, 1.5, 2.0, 2.5):
        mm = stretch[m] == f
        if len(set(y[mm])) > 1:
            ws[str(f)] = {"n": int(mm.sum()), "pos": int(y[mm].sum()), "auroc_jev": float(roc_auc_score(y[mm], preds["jev"][mm])),
                          "auroc_gap_only": float(roc_auc_score(y[mm], -X[mm, 1]))}
    d["within_stretch"] = ws
    # per-repeat noise
    d["jev_rep_sd_mean"] = float(np.nanmean(np.nanstd(Pf_rep[jk][m], 1)))
    res["labels"][key] = d
    print(f"{key:13s} n={d['n']} pos={d['pos_rate']:.2f} " + "  ".join(
        f"{k}: {v['auroc']:.2f}/{v['brier']:.3f}/{v['ece']:.3f}" for k, v in d["methods"].items()))
# v1 vs v2 label agreement (for the report)
res["v1_v2_agreement"] = {
    "mr": float(np.mean([a["mr"] == b["mr"] for a, b in zip(v1, lab)])),
    "t1": float(np.mean([a["t1_flag"] == bool(b["t1_flag"]) for a, b in zip(v1, lab) if b["t1"] is not None and not a["exclude_t1"]])),
    "unstable_0p1": float(np.mean([a["rhf_unstable"] == b["rhf_unstable_0p1"] for a, b in zip(v1, lab)]))}
res["jev_mean_p_by_stretch"] = {str(f): {k: float(Pf[k][stretch == f].mean()) for k in ("mr", "t1", "unstable")} for f in (1.0, 1.5, 2.0, 2.5)}
res["label_rate_by_stretch"] = {str(f): {k: float(np.mean(LABELS[k][1][(stretch == f) & LABELS[k][2]])) for k in ("mr", "t1", "unstable")} for f in (1.0, 1.5, 2.0, 2.5)}

# ---------------- spin-state task ----------------
sp = json.load(open("experiments/e5_spin_labels.json"))
M = ["1", "3", "5"]
spin_rows = []
for s in sp:
    st = s["states"]
    E = {m: st[m].get("e_cc") for m in M if m in st and st[m].get("e_cc") is not None}
    # a state's adiabatic energy cannot exceed its energy at the ground-state geometry; where the Nelder-Mead /
    # bounded search ended higher (stuck on a worse SCF branch), use the lower vertical value and record it
    viol = {}
    for m, ev in s["vertical_at_ground_geom"].items():
        if ev is not None and m in E and ev < E[m] - 1e-6:
            viol[m] = float((E[m] - ev) * 1000); E[m] = ev
    truth = min(E, key=E.get)
    others = sorted(E[m] for m in E if m != truth)
    Ehf = {m: st[m].get("e_hf") for m in E if st[m].get("e_hf") is not None}
    ps = [x["probabilities"] for x in raw["spin"] if x["name"] == s["name"]]
    p = np.array([np.mean([q.get(m, 0.0) for q in ps]) for m in M]); p = p / p.sum()
    vert = {m: v for m, v in s["vertical_at_ground_geom"].items() if v is not None}
    spin_rows.append({"name": s["name"], "charge": s["charge"], "truth": truth, "expt": str(s["expt_mult"]),
                      "gap_to_next_eh": float(sorted(E[m] for m in E if m != truth)[0] - E[truth]), "E_ccsdt": E,
                      "optimisation_worse_than_vertical_mEh": viol, "ground_state_ref_sane": st[truth].get("all_sane"),
                      "s2_ref": {m: st[m].get("s2_ref") for m in E}, "all_sane": {m: st[m].get("all_sane") for m in E},
                      "hf_order": min(Ehf, key=Ehf.get), "vertical_order": min(vert, key=vert.get),
                      "rohf_check": s.get("rohf_check"), "params": {m: st[m].get("params") for m in E},
                      "jev_p": dict(zip(M, p.tolist())), "jev_argmax": M[int(np.argmax(p))],
                      "jev_rep_choices": [x["choice"] for x in raw["spin"] if x["name"] == s["name"]]})
def spin_metrics(rows):
    y = np.array([[r["truth"] == m for m in M] for r in rows], float)
    P = np.array([[r["jev_p"][m] for m in M] for r in rows])
    freq = y.mean(0)
    out = {"n": len(rows), "class_counts": dict(zip(M, y.sum(0).astype(int).tolist())),
           "jev": {"accuracy": float(np.mean(P.argmax(1) == y.argmax(1))), "brier": float(np.mean(np.sum((P - y)**2, 1))),
                   "log_loss": float(-np.mean(np.log(np.clip(np.sum(P*y, 1), 0.005, 1))))},
           "always_singlet": {"accuracy": float(y[:, 0].mean()), "brier": float(np.mean(np.sum((np.array([1, 0, 0]) - y)**2, 1)))},
           "class_frequency": {"accuracy": float(np.mean(y.argmax(1) == freq.argmax())), "brier": float(np.mean(np.sum((freq - y)**2, 1)))},
           "hf_ordering": {"accuracy": float(np.mean([r["hf_order"] == r["truth"] for r in rows]))},
           "expt_assignment_agrees_with_ccsdt": float(np.mean([r["expt"] == r["truth"] for r in rows])),
           "jev_agrees_with_expt": float(np.mean([r["jev_argmax"] == r["expt"] for r in rows]))}
    return out
res["spin"] = {"all": spin_metrics(spin_rows),
               "clear_gap_gt_5mEh": spin_metrics([r for r in spin_rows if r["gap_to_next_eh"] > 5e-3]),
               "near_degenerate_lt_5mEh": [r["name"] for r in spin_rows if r["gap_to_next_eh"] <= 5e-3],
               "clear_gap_gt_10mEh": spin_metrics([r for r in spin_rows if r["gap_to_next_eh"] > 10e-3]),
               "near_degenerate_lt_10mEh": [r["name"] for r in spin_rows if r["gap_to_next_eh"] <= 10e-3],
               "ccsdt_vs_expt_disagreements": [r["name"] for r in spin_rows if r["truth"] != r["expt"]],
               "vs_expt_labels": spin_metrics([{**r, "truth": r["expt"]} for r in spin_rows]),
               "ground_state_without_sane_ccsdt": [r["name"] for r in spin_rows if not r["ground_state_ref_sane"]],
               "optimisation_fixups": {r["name"]: r["optimisation_worse_than_vertical_mEh"] for r in spin_rows if r["optimisation_worse_than_vertical_mEh"]},
               "vertical_vs_adiabatic_disagreements": [r["name"] for r in spin_rows if r["vertical_order"] != r["truth"]],
               "jev_errors": [r["name"] for r in spin_rows if r["jev_argmax"] != r["truth"]],
               "rows": spin_rows, "question": raw["spin_question"]}
print("spin", json.dumps({k: v for k, v in res["spin"].items() if k != "rows"}, indent=0)[:2500])
json.dump(res, open("experiments/e5_results.json", "w"), indent=1)
