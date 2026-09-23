"""E1c path independence, E1p/E3p/E2p paraphrase robustness, and item 8: projection of Jev's conditionals onto one
pairwise Ising joint per case (fit, held-out prediction of two-fact conditionals, repaired marginals for E3)."""
import itertools, json
import numpy as np
import scipy.stats as st
from scipy.optimize import least_squares
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
S = ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"]; K = len(S)
cl = lambda p: np.clip(p, .005, .995); lg = lambda p: np.log(cl(p) / (1 - cl(p)))
rows = json.load(open("experiments/qc_labels.json")); M = len(rows)
e1 = json.load(open("experiments/e1e3_raw.json")); e8 = json.load(open("experiments/e8_raw.json"))
res = json.load(open("experiments/e1e3_results.json")); noise_sd = json.load(open("experiments/r1_results.json"))["noise_sd_unbiased"]

# ---- gather mean probabilities: D[case][facts_tuple][var] ; facts_tuple = tuple(sorted((k, v)))
def collect(records, key_of):
    acc = {}
    for x in records:
        acc.setdefault((x["case"], key_of(x)), {}).setdefault("_", []).append(x["p"])
    D = [dict() for _ in range(M)]
    for (i, f), v in acc.items():
        ps = v["_"]; D[i][f] = {k: float(np.mean([p[k] for p in ps])) for k in ps[0]}
    return D
D = collect(e1, lambda x: tuple(sorted([tuple(x["cond"])])) if x["cond"] else ())
D2 = collect([x for x in e8 if x["kind"] == "e1c"], lambda x: tuple(sorted(tuple(c) for c in x["cond"])))
for i in range(M): D[i].update(D2[i])

# ---- E1c: path independence of log p(111)/p(000) over the 6 chain-rule orderings of each triple
def path_logratio(Di, order):
    tot = 0.0
    for v in (True, False):
        s, facts = 0.0, []
        for x in order:
            p = Di[tuple(sorted(facts))][x]; s += np.log(cl(p) if v else cl(1 - p)); facts.append((x, v))
        tot += s if v else -s
    return tot
spread = []
for i in range(M):
    for tri in itertools.combinations(S, 3):
        vals = [path_logratio(D[i], o) for o in itertools.permutations(tri)]
        spread.append(max(vals) - min(vals))
spread = np.array(spread)
# coherent null: random 3-spin Ising, each measured conditional gets per-call noise, rounding, averaged over reps
def null_spread():
    h = rng.normal(0, 1, 3); J = rng.normal(0, 1.5, 3)
    st3 = np.array(list(itertools.product([0, 1], repeat=3)))
    w = np.exp(st3 @ h + J[0]*st3[:, 0]*st3[:, 1] + J[1]*st3[:, 0]*st3[:, 2] + J[2]*st3[:, 1]*st3[:, 2]); w /= w.sum()
    def cond(x, facts):
        m = np.ones(8, bool)
        for y, v in facts: m &= st3[:, y] == v
        return w[m & (st3[:, x] == 1)].sum() / w[m].sum()
    reps = {0: 3, 1: 3, 2: 2}
    def obs(p, nf): return np.mean(np.round(np.clip(p + rng.normal(0, noise_sd, reps[nf]), 0, 1), 2))
    vals = []
    for o in itertools.permutations(range(3)):
        tot = 0.0
        for v in (1, 0):
            s, facts = 0.0, []
            for x in o:
                p = obs(cond(x, facts), len(facts)); s += np.log(cl(p) if v else cl(1 - p)); facts.append((x, v))
            tot += s if v else -s
        vals.append(tot)
    return max(vals) - min(vals)
null = np.array([null_spread() for _ in range(3000)])
q95 = np.percentile(null, 95)
print(f"E1c path spread (nats): median {np.median(spread):.2f}, 90th {np.percentile(spread,90):.2f}; null median {np.median(null):.2f}, 95th {q95:.2f}; frac above null95 {np.mean(spread>q95):.2f}")

# ---- item 8: pairwise Ising projection per case
st5 = np.array(list(itertools.product([0, 1], repeat=K)))
pairs = list(itertools.combinations(range(K), 2))
F = np.hstack([st5, np.array([[s[a]*s[b] for a, b in pairs] for s in st5])])
def model_cond(theta, x, facts):
    e = F @ theta; w = np.exp(e - e.max()); m = np.ones(len(st5), bool)
    for y, v in facts: m &= st5[:, S.index(y)] == int(v)
    return w[m & (st5[:, S.index(x)] == 1)].sum() / w[m].sum()
def obs_list(Di, max_facts):
    return [(x, f, p) for f, d in Di.items() if len(f) <= max_facts for x, p in d.items()]
def fit(obs):
    r = lambda th: np.array([lg(model_cond(th, x, f)) - lg(p) for x, f, p in obs])
    return least_squares(r, np.zeros(F.shape[1]), method="lm").x
fit_rms, heldout, heldout_null, repaired = [], [], [], np.zeros((M, K))
for i in range(M):
    full = obs_list(D[i], 2); th = fit(full)
    fit_rms.append(np.sqrt(np.mean([(lg(model_cond(th, x, f)) - lg(p))**2 for x, f, p in full])))
    repaired[i] = [model_cond(th, x, ()) for x in S]
    th1 = fit(obs_list(D[i], 1))                                  # fit on marginals + one-fact conditionals only
    two = [(x, f, p) for x, f, p in full if len(f) == 2]
    heldout.append(np.sqrt(np.mean([(lg(model_cond(th1, x, f)) - lg(p))**2 for x, f, p in two])))
    heldout_null.append(np.sqrt(np.mean([(lg(D[i][tuple(sorted([f[0]]))][x]) - lg(p))**2 for x, f, p in two])))  # baseline: ignore 2nd fact
# noise floor in logit units at the typical probability
Pall = np.array([p for Di in D for d in Di.values() for p in d.values()])
logit_noise = np.median(noise_sd / np.sqrt(2.5) / (cl(Pall) * (1 - cl(Pall))))
print(f"Ising projection: in-sample logit RMS median {np.median(fit_rms):.2f} (noise ~{logit_noise:.2f}); "
      f"held-out 2-fact RMS: pairwise model {np.median(heldout):.2f} vs ignore-second-fact {np.median(heldout_null):.2f}")
v2 = json.load(open("experiments/qc_labels_v2.json"))
okt1 = np.array([r["t1_flag"] is not None for r in v2])
lab = {"mr": np.array([r["mr"] for r in v2]).astype(int), "unstable": np.array([r["rhf_unstable"] for r in v2]).astype(int),
       "t1": np.array([bool(r["t1_flag"]) for r in v2]).astype(int)}
msk = {"mr": np.ones(M, bool), "unstable": np.ones(M, bool), "t1": okt1}
Pm = np.array(res["Pm"]); rep_e3 = {}
for k, y in lab.items():
    pr, pp, y = Pm[msk[k], S.index(k)], repaired[msk[k], S.index(k)], y[msk[k]]
    rep_e3[k] = {"raw": [roc_auc_score(y, pr), float(np.mean((pr-y)**2))], "repaired": [roc_auc_score(y, pp), float(np.mean((pp-y)**2))]}
    print(f"  E3 {k:9s} raw AUROC {rep_e3[k]['raw'][0]:.2f} Brier {rep_e3[k]['raw'][1]:.3f} | repaired AUROC {rep_e3[k]['repaired'][0]:.2f} Brier {rep_e3[k]['repaired'][1]:.3f}")

# ---- paraphrase robustness
P1 = collect([x for x in e8 if x["kind"] == "e1p"], lambda x: tuple(sorted([tuple(c) for c in x["cond"]])) if x["cond"] else ())
asy_p, J_orig, J_para = [], [], []
for i in range(M):
    for a, b in pairs:
        A, B = S[a], S[b]
        J = lambda Dd, x, y: lg(Dd[((y, True),)][x]) - lg(Dd[((y, False),)][x])
        asy_p.append(J(P1[i], A, B) - J(P1[i], B, A)); J_para.append(J(P1[i], A, B)); J_orig.append(J(D[i], A, B))
asy_p = np.abs(asy_p); nq = np.percentile(res["null_abs"], 95)
print(f"E1p paraphrase: median |asym| {np.median(asy_p):.2f}, frac above coherent-null95 {np.mean(asy_p>nq):.2f} (original {np.mean(np.abs(np.ravel(res['asym']))>nq):.2f}); "
      f"corr(J_orig, J_para) {st.pearsonr(J_orig, J_para).statistic:.2f}")
e3p = {}
for k, y in lab.items():
    au = []
    for j in range(4):
        p = np.array([np.mean([x["p"][k] for x in e8 if x["kind"] == "e3p" and x["case"] == i and x["cond"] == j]) for i in range(M)])
        au.append(roc_auc_score(y[msk[k]], p[msk[k]]))
    e3p[k] = au; print(f"E3p {k:9s} AUROC over 4 paraphrases: {np.round(au, 2)} (original {roc_auc_score(y[msk[k]], Pm[msk[k], S.index(k)]):.2f})")
e2p = {}
for w in (0, 1):
    for n in range(3, 8):
        xs = [x for x in e8 if x["kind"] == "e2p" and x["case"][0] == n and x["cond"] == w]
        rho = [st.spearmanr([-x["E"][c] for c in x["E"]], [x["p"].get(c, 0) for c in x["E"]]).statistic for x in xs]
        e2p[f"w{w}_n{n}"] = float(np.mean(rho))
print("E2p mean rho(p,-E) by wording/n:", {k: round(v, 2) for k, v in e2p.items()})
json.dump({"e1c_spread": spread.tolist(), "e1c_null": null.tolist(), "fit_rms": fit_rms, "heldout": heldout, "heldout_null": heldout_null,
           "repaired_e3": rep_e3, "e1p_asym": asy_p.tolist(), "e1p_Jcorr": float(st.pearsonr(J_orig, J_para).statistic),
           "e3p_auroc": e3p, "e2p_rho": e2p, "logit_noise": float(logit_noise)}, open("experiments/e8_results.json", "w"), indent=1)
