"""All E2 numbers offline from cache: Ising (as stated / sign-flipped), field-coupling decomposition, instance-level
uncertainty, H2 dominant state, particle-sector leakage, energy read-out vs uninformed baselines, qubit relabelling."""
import json
import numpy as np
import scipy.stats as st
e2 = json.load(open("experiments/e2_raw.json")); e2b = json.load(open("experiments/e2b_raw.json")); e2c = json.load(open("experiments/e2c_raw.json"))
out = {}
def ising_summary(recs):
    res = {}
    for n in range(3, 8):
        inst = {}
        for x in [r for r in recs if r["n"] == n]:
            inst.setdefault(x["k"], []).append(x)
        rho_i, top_i, fld_i, cpl_i = [], [], [], []
        for k, xs in inst.items():
            keys = list(xs[0]["E"]); E = np.array([xs[0]["E"][c] for c in keys])
            p = np.mean([[x["p"].get(c, 0) for c in keys] for x in xs], 0)
            S = np.array([[1 if ch == "+" else -1 for ch in c] for c in keys])
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            X = np.hstack([S, np.array([[s[i] * s[j] for i, j in pairs] for s in S])])
            coef = np.linalg.lstsq(X, E, rcond=None)[0]
            rho_i.append(st.spearmanr(p, -E).statistic); fld_i.append(st.spearmanr(p, S @ coef[:n]).statistic)
            cpl_i.append(st.spearmanr(p, -(X[:, n:] @ coef[n:])).statistic)
            top_i.append(np.mean([max(x["p"], key=x["p"].get) == keys[int(np.argmin(E))] for x in xs]))
        res[n] = {"rho": [float(np.mean(rho_i)), float(st.sem(rho_i))], "top1": [float(np.mean(top_i)), float(st.sem(top_i))],
                  "rho_plus_field": float(np.mean(fld_i)), "rho_minus_coupling": float(np.mean(cpl_i)), "chance": 2.0 ** -n}
    return res
out["stated"] = ising_summary(e2["ising"]); out["flipped"] = ising_summary(e2b)
for name in ("stated", "flipped"):
    print(name, {n: (round(v["rho"][0], 2), round(v["rho"][1], 2), round(v["top1"][0], 2), round(v["rho_plus_field"], 2), round(v["rho_minus_coupling"], 2)) for n, v in out[name].items()})
# H2
ex = {d["R"]: d for d in e2["h2_exact"]}; lv = np.linspace(-1.20, -0.75, 10); step = lv[1] - lv[0]
two = lambda b: b.count("1") == 2
tops = [max(x["p_dom"], key=x["p_dom"].get) for x in e2["h2"]]
leak = [sum(v for b, v in x["p_dom"].items() if not two(b)) for x in e2["h2"]]
Eread = {R: np.mean([lv[0] + x["score"] * step for x in e2["h2"] if x["R"] == R]) for R in ex}
Eex = np.array([ex[R]["E_exact"] for R in ex]); Er = np.array([Eread[R] for R in ex])
snap = lv[np.abs(Eex[:, None] - lv[None, :]).argmin(1)]
mode = [max(x["E_probs"], key=x["E_probs"].get) for x in e2["h2"]]
out["h2"] = {"top_1100": tops.count("1100"), "top_0011": tops.count("0011"), "n": len(tops),
             "p_1100_by_R": {R: float(np.mean([x["p_dom"].get("1100", 0) for x in e2["h2"] if x["R"] == R])) for R in ex},
             "w_1100_by_R": {R: ex[R]["weights"]["1100"] for R in ex}, "leak_mean": float(np.mean(leak)), "leak_range": [float(min(leak)), float(max(leak))],
             "mae_jev_meh": float(1000 * np.mean(np.abs(Er - Eex))), "mae_const_median_meh": float(1000 * np.mean(np.abs(np.median(Eex) - Eex))),
             "mae_const_mid_meh": float(1000 * np.mean(np.abs(-0.975 - Eex))), "mae_snap_meh": float(1000 * np.mean(np.abs(snap - Eex))),
             "readout_range_meh": float(1000 * (Er.max() - Er.min())), "exact_range_meh": float(1000 * (Eex.max() - Eex.min())),
             "mode_level_counts": {m: mode.count(m) for m in set(mode)}}
tc = [max(x["p"], key=x["p"].get) for x in e2c]
out["h2_relabelled"] = {"true_dominant": "0101", "top_counts": {t: tc.count(t) for t in set(tc)},
                        "p_true": float(np.mean([x["p"].get("0101", 0) for x in e2c])), "p_1100": float(np.mean([x["p"].get("1100", 0) for x in e2c]))}
print(json.dumps({k: v for k, v in out["h2"].items() if "by_R" not in k}, indent=0)); print(out["h2_relabelled"])
r1 = json.load(open("experiments/r1_results.json")); out["beta"] = r1["e2_beta"]
json.dump(out, open("experiments/e2_results.json", "w"), indent=1)
