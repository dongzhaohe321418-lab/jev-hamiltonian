"""E7 analysis: the same statistics as analyze_e1e3.py / e1b_e3b.py / figures.py (E2 summary), for open-weight models.
Run from repo root after e7_collect.py. Writes experiments/e7_results.json."""
import itertools, json, sys
import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
import e2_hamiltonian as e2

MODELS = {"Qwen2.5-1.5B-Instruct": "experiments/e7_raw_Qwen2.5-1.5B-Instruct.json",
          "Qwen2.5-7B-Instruct": "experiments/e7_raw_Qwen2.5-7B-Instruct.json"}
MODELS = {k: v for k, v in MODELS.items() if __import__("os").path.exists(v)}
rows = json.load(open("experiments/qc_labels.json"))
S = ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"]; M = len(rows)
conds = [None] + [(k, v) for k in S for v in (True, False)]
ci = lambda k, v: conds.index((k, v))
jevr = json.load(open("experiments/e1e3_results.json"))
NULL95 = float(np.percentile(jevr["null_abs"], 95))   # Jev-noise coherent null, 95th pct (0.357)


def stats(Pm, lo=0.005):
    """Identical to analyze_e1e3.stats with R=1 (Pm already rep-averaged); clip [lo, 1-lo] before logit."""
    logit = lambda p: np.log(np.clip(p, lo, 1 - lo) / (1 - np.clip(p, lo, 1 - lo)))
    J, resid = {}, []
    for a, b in itertools.permutations(range(len(S)), 2):
        ka, kb = S[a], S[b]
        J[(a, b)] = logit(Pm[:, ci(kb, True), a]) - logit(Pm[:, ci(kb, False), a])
        pb = Pm[:, 0, b]
        resid.append(Pm[:, 0, a] - (Pm[:, ci(kb, True), a] * pb + Pm[:, ci(kb, False), a] * (1 - pb)))
    asym = np.array([J[(a, b)] - J[(b, a)] for a, b in itertools.combinations(range(len(S)), 2)])
    return J, asym, np.array(resid)


def ece(p, y, b=10):   # identical to analyze_e1e3.ece
    bins = np.minimum((p * b).astype(int), b - 1)
    return sum(np.abs(p[bins == k].mean() - y[bins == k].mean()) * np.mean(bins == k) for k in range(b) if np.any(bins == k))


def pct(x, qs=(10, 25, 50, 75, 90, 95)): return {str(q): float(np.percentile(x, q)) for q in qs}


def ising_summary(recs, extra=True):
    """figures.py summ(): mean/sem of per-record Spearman rho(p, -E), mean top-1; plus field/coupling decomposition."""
    out = {}
    for n in range(3, 8):
        rs = [x for x in recs if x["n"] == n]
        rho = [st.spearmanr([-x["E"][k] for k in x["E"]], [x["p"].get(k, 0) for k in x["E"]]).statistic for x in rs]
        top = [max(x["p"], key=x["p"].get) == min(x["E"], key=x["E"].get) for x in rs]
        d = {"rho_mean": float(np.mean(rho)), "rho_sem": float(st.sem(rho)), "top1": float(np.mean(top)),
             "top1_se": float(np.std(top) / np.sqrt(len(top))), "chance": 2.0 ** -n, "n_records": len(rs)}
        if extra and "field" in rs[0]:
            d["rho_field_plus"] = float(np.mean([st.spearmanr([x["field"][k] for k in x["E"]], [x["p"][k] for k in x["E"]]).statistic for x in rs]))
            d["rho_coupling_plus"] = float(np.mean([st.spearmanr([x["coupling"][k] for k in x["E"]], [x["p"][k] for k in x["E"]]).statistic for x in rs]))
        if "logp" in rs[0]:
            first = [max(x["p"], key=x["p"].get) == next(iter(x["p"])) for x in rs]
            last = [max(x["p"], key=x["p"].get) == list(x["p"])[-1] for x in rs]
            d["argmax_is_first_option_allplus"] = float(np.mean(first)); d["argmax_is_last_option_allminus"] = float(np.mean(last))
            d["max_p_mean"] = float(np.mean([max(x["p"].values()) for x in rs]))
        out[n] = d
    return out


res = {"null95_jev_noise": NULL95, "models": {}, "e3_labels": "primary e3 = experiments/qc_labels_v2.json (mr, t1_flag, rhf_unstable; t1_flag None skipped -> n=47); e3_v1_labels = experiments/qc_labels.json"}
rows2 = json.load(open("experiments/qc_labels_v2.json"))
assert [(a["name"], a["stretch"]) for a in rows2] == [(a["name"], a["stretch"]) for a in rows]


def e3_metrics(P5, rws):
    """AUROC/Brier/ECE as in analyze_e1e3 (ECE 10 bins); P5 = marginal p(yes) per case x statement S."""
    out = {}
    for k, lk in [("mr", "mr"), ("t1", "t1_flag"), ("unstable", "rhf_unstable")]:
        ok = np.array([x[lk] is not None for x in rws]); y = np.array([bool(x[lk]) for x in rws])[ok].astype(int)
        p = np.asarray(P5)[ok, S.index(k)]
        out[k] = {"n": int(ok.sum()), "pos_rate": float(y.mean()), "auroc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
                  "ece": float(ece(p, y)), "p_mean": float(p.mean()), "p_min": float(p.min()), "p_max": float(p.max())}
    return out

# ---- Jev reference numbers recomputed from the committed raw/result files ----
neg = json.load(open("experiments/e1b_e3b_raw.json"))["neg"]
jev_neg = {}
for k in S:
    v = np.array([p[k] + p[k + "_neg"] for _, _, p in neg])
    jev_neg[k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "frac_absdev_gt_0.1": float(np.mean(np.abs(v - 1) > 0.1))}
insts = list(e2.ising_instances())
fieldE = {(n, k): (h, J) for n, k, h, J in insts}
def add_terms(recs):
    for x in recs:
        h, J = fieldE[(x["n"], x["k"])]
        x["field"] = {}; x["coupling"] = {}
        for lab in x["E"]:
            s = [1 if c == "+" else -1 for c in lab]
            x["field"][lab] = float(sum(h[i] * s[i] for i in range(len(s))))
            x["coupling"][lab] = float(sum(v * s[i] * s[j] for (i, j), v in J.items()))
        x["p"] = {lab: x["p"].get(lab, 0) for lab in x["E"]} | {kk: vv for kk, vv in x["p"].items()}
    return recs
e2raw = json.load(open("experiments/e2_raw.json"))["ising"]; e2b = json.load(open("experiments/e2b_raw.json"))
res["jev"] = {"asym_median": float(np.median(np.abs(jevr["asym"]))),
              "frac_asym_gt_null95": float(np.mean(np.abs(np.array(jevr["asym"])) > NULL95)),
              "asym_pct": pct(np.abs(np.array(jevr["asym"])).ravel()),
              "median_absJ": float(np.median(np.abs(np.array([v for p in jevr["J_pairs"] for v in p])))),
              "resid_median": float(np.median(np.abs(jevr["resid"]))), "resid_p90": float(np.percentile(np.abs(jevr["resid"]), 90)),
              "e3": e3_metrics(np.array(jevr["Pm"]), rows2), "e3_v1_labels": e3_metrics(np.array(jevr["Pm"]), rows), "neg": jev_neg,
              "ising_stated": ising_summary(add_terms(e2raw)), "ising_flipped": ising_summary(add_terms(e2b)),
              "note": "Jev: R=3 repeats averaged for E1/E3; E1b sums per call (as in e1b_e3b.py); Ising summaries per call (10 instances x 3 repeats), as in figures.py"}
_v2 = json.load(open("experiments/e3v2_results.json"))
for k in ("mr", "t1", "unstable"):   # cross-check Jev's v2 numbers against analyze_e3v2.py's committed output
    assert abs(res["jev"]["e3"][k]["auroc"] - _v2[k]["jev"]["auroc"]) < 1e-9 and abs(res["jev"]["e3"][k]["brier"] - _v2[k]["jev"]["brier"]) < 1e-9
res["baselines_e3_v2_lomo"] = {k: {b: {m: _v2[k][b][m] for m in ("auroc", "brier", "ece")} for b in ("base_rate", "stretch", "logistic")} for k in ("mr", "t1", "unstable")}
res["baselines_e3_v1_loo"] = {k: {b: jevr["e3"][k][b] for b in ("base_rate", "logreg_stretch", "logreg_stretch_gap")} for k in ("mr", "t1", "unstable")}

def analyze(raw):
    r = {}
    Pm = np.full((M, len(conds), len(S)), np.nan)
    for x in raw["e1"]:
        Pm[x["case"], conds.index(tuple(x["cond"]) if x["cond"] else None), S.index(x["q"])] = x["p"]
    assert np.isnan(Pm).sum() == M * 10  # only the conditioned statement itself is missing in each condition
    J, asym, resid = stats(Pm)
    Jall = np.array(list(J.values())); aa = np.abs(asym).ravel()
    r["e1"] = {"n_pair_cases": int(asym.size), "asym_median": float(np.median(aa)), "asym_mean": float(aa.mean()), "asym_pct": pct(aa),
               "frac_asym_gt_null95": float(np.mean(aa > NULL95)), "frac_asym_gt_0.1": float(np.mean(aa > 0.1)),
               "frac_asym_gt_1": float(np.mean(aa > 1.0)), "median_absJ": float(np.median(np.abs(Jall))),
               "resid_median": float(np.median(np.abs(resid))), "resid_p90": float(np.percentile(np.abs(resid), 90)),
               "J_mean": {f"{S[a]}|{S[b]}": float(np.mean(v)) for (a, b), v in J.items()},
               "frac_p_clipped": float(np.mean((Pm[~np.isnan(Pm)] < 0.005) | (Pm[~np.isnan(Pm)] > 0.995)))}
    J6, asym6, resid6 = stats(Pm, lo=1e-6)
    # Saturation diagnostic: pair-cases whose four conditionals all lie inside [0.005, 0.995] (unaffected by the clip).
    inside = lambda x: (x >= 0.005) & (x <= 0.995)
    un = np.array([inside(Pm[:, ci(S[b], True), a]) & inside(Pm[:, ci(S[b], False), a]) & inside(Pm[:, ci(S[a], True), b]) & inside(Pm[:, ci(S[a], False), b])
                   for a, b in itertools.combinations(range(len(S)), 2)])
    au = np.abs(asym)[un]
    r["e1"]["unclipped_subset"] = {"n": int(un.sum()), "asym_median": float(np.median(au)) if un.any() else None,
                                   "frac_asym_gt_null95": float(np.mean(au > NULL95)) if un.any() else None}
    r["e1"]["frac_pair_cases_with_a_clipped_conditional"] = float(1 - un.mean())
    r["e1"]["sensitivity_clip_1e-6"] = {"asym_median": float(np.median(np.abs(asym6))), "median_absJ": float(np.median(np.abs(np.array(list(J6.values())))))}
    r["e1"]["coherent_null_zero_noise"] = "a coherent model read without noise or rounding gives |J_ab-J_ba| = 0 exactly (up to the clip), so every nonzero asymmetry is a violation"
    mass = np.array([x["mass"] for x in raw["e1"] + raw["neg"]])
    tops = [x["top"] for x in raw["e1"] + raw["neg"]]
    r["readout"] = {"yn_mass_median": float(np.median(mass)), "yn_mass_min": float(mass.min()), "yn_mass_p5": float(np.percentile(mass, 5)),
                    "frac_argmax_in_yn_set": float(np.mean([t in set(raw["yn_token_strings"].values()) for t in tops])),
                    "argmax_tokens": {t: tops.count(t) for t in set(tops)},
                    "prompt_tokens_median": float(np.median([x["n_prompt_tokens"] for x in raw["e1"]]))}
    # E1b
    marg = {(x["case"], x["q"]): x["p"] for x in raw["e1"] if x["cond"] is None}
    r["neg"] = {}
    for k in S:
        v = np.array([marg[(x["case"], k)] + x["p"] for x in raw["neg"] if x["q"] == k])
        r["neg"][k] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "frac_absdev_gt_0.1": float(np.mean(np.abs(v - 1) > 0.1))}
    # E3
    r["e3"] = e3_metrics(Pm[:, 0, :], rows2)          # primary: v2 labels (qc_labels_v2.json), None skipped
    r["e3_v1_labels"] = e3_metrics(Pm[:, 0, :], rows)  # secondary: original qc_labels.json
    # E2
    r["ising_stated"] = ising_summary([x for x in raw["ising"] if x["form"] == "stated"])
    r["ising_flipped"] = ising_summary([x for x in raw["ising"] if x["form"] == "flipped"])
    r["runtime_s"] = raw["runtime_s"]; r["load_s"] = raw["load_s"]; r["snapshot_path"] = raw.get("snapshot_path")
    r["yn_token_ids"] = raw["yn_token_ids"]; r["yn_token_strings"] = raw["yn_token_strings"]
    r["e1"]["_Pm_marginals"] = Pm[:, 0, :].tolist()
    r["_asym"] = asym.tolist(); r["_Pall"] = Pm
    return r


for name, path in MODELS.items():
    res["models"][name] = analyze(json.load(open(path)))
# Precision check: identical 1.5B run with the whole network in float32 (E7_FP32=1) vs the bf16 body used above.
fp = "experiments/e7_raw_Qwen2.5-1.5B-Instruct_fp32.json"
if __import__("os").path.exists(fp) and "Qwen2.5-1.5B-Instruct" in res["models"]:
    r32 = analyze(json.load(open(fp))); r16 = res["models"]["Qwen2.5-1.5B-Instruct"]
    d = np.abs(r32["_Pall"] - r16["_Pall"]); d = d[~np.isnan(d)]
    da = np.abs(np.abs(np.array(r32["_asym"])) - np.abs(np.array(r16["_asym"]))).ravel()
    res["precision_check_1.5B_fp32_vs_bf16"] = {
        "p_absdiff_median": float(np.median(d)), "p_absdiff_max": float(d.max()), "abs_asym_absdiff_median": float(np.median(da)),
        "abs_asym_absdiff_p90": float(np.percentile(da, 90)),
        "fp32": {"asym_median": r32["e1"]["asym_median"], "frac_asym_gt_null95": r32["e1"]["frac_asym_gt_null95"],
                 "resid_median": r32["e1"]["resid_median"], "resid_p90": r32["e1"]["resid_p90"], "e3": r32["e3"], "neg": r32["neg"],
                 "rho_stated": {n: v["rho_mean"] for n, v in r32["ising_stated"].items()},
                 "rho_flipped": {n: v["rho_mean"] for n, v in r32["ising_flipped"].items()},
                 "top1_stated": {n: v["top1"] for n, v in r32["ising_stated"].items()},
                 "top1_flipped": {n: v["top1"] for n, v in r32["ising_flipped"].items()}}}
    print("precision check", {k: v for k, v in res["precision_check_1.5B_fp32_vs_bf16"].items() if k != "fp32"})
for r in res["models"].values(): r.pop("_Pall")

json.dump(res, open("experiments/e7_results.json", "w"), indent=1)
for name, r in [("Jev", None)] + list(res["models"].items()):
    if r is None:
        j = res["jev"]; print(f"Jev  asym med {j['asym_median']:.2f} >null95 {j['frac_asym_gt_null95']:.2f} resid {j['resid_median']:.3f}/{j['resid_p90']:.3f}")
        e3, neg_, A, B = j["e3"], j["neg"], j["ising_stated"], j["ising_flipped"]
    else:
        e = r["e1"]; print(f"{name} asym med {e['asym_median']:.2f} >null95 {e['frac_asym_gt_null95']:.2f} resid {e['resid_median']:.3f}/{e['resid_p90']:.3f} |J| {e['median_absJ']:.2f} clipped {e['frac_p_clipped']:.2f} mass {r['readout']['yn_mass_median']:.3f}")
        e3, neg_, A, B = r["e3"], r["neg"], r["ising_stated"], r["ising_flipped"]
    print("  E3 " + "  ".join(f"{k}: AUC {v['auroc']:.2f} Brier {v['brier']:.3f} ECE {v['ece']:.3f}" for k, v in e3.items()))
    print("  neg " + "  ".join(f"{k}: {v['mean']:.2f} [{v['min']:.2f},{v['max']:.2f}] {v['frac_absdev_gt_0.1']:.2f}" for k, v in neg_.items()))
    print("  rho stated " + " ".join(f"{A[n]['rho_mean']:+.2f}" for n in A) + " | flipped " + " ".join(f"{B[n]['rho_mean']:+.2f}" for n in B))
    print("  top1 stated " + " ".join(f"{A[n]['top1']:.2f}" for n in A) + " | flipped " + " ".join(f"{B[n]['top1']:.2f}" for n in B))
    print("  rho field+ stated " + " ".join(f"{A[n].get('rho_field_plus', float('nan')):+.2f}" for n in A) + " coupling+ " + " ".join(f"{A[n].get('rho_coupling_plus', float('nan')):+.2f}" for n in A))
