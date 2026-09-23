"""Exploratory. E1b: negation normalisation p(s)+p(not s). E3b: geometry-only input (no name, no stretch note)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from jev import ask, noul
from e1e3_collect import S, state_of
NEG = {"mr": "The ground state does NOT have strong multireference character: the Hartree-Fock determinant carries at least 90% of the exact wavefunction weight.",
       "rhf_ok": "Restricted Hartree-Fock is NOT a qualitatively correct single-determinant reference for this system.",
       "t1": "The CCSD T1 diagnostic does NOT exceed 0.02.",
       "unstable": "The RHF solution is stable: no spin-unrestricted (UHF) solution with lower energy exists.",
       "ccsdt_ok": "CCSD(T) will NOT be within 1 kcal/mol of the exact (full CI) energy in this basis."}
rows = json.load(open("experiments/qc_labels.json")); R = 3
def neg_job(t):
    i, r = t
    qs = {**{k: noul(v) for k, v in S.items()}, **{k + "_neg": noul(v) for k, v in NEG.items()}}
    return i, r, {k: v["noul"] for k, v in ask(state_of(rows[i]), qs, rep=r)["answers"].items()}
def geo_job(t):
    i, r = t
    st = {"geometry_xyz_angstrom": rows[i]["geometry"], "method_context": "electronic structure, cc-pVDZ basis, singlet reference"}
    return i, r, {k: v["noul"] for k, v in ask(st, {k: noul(S[k]) for k in ("mr", "t1", "unstable")}, rep=r)["answers"].items()}
jobs = [(i, r) for i in range(len(rows)) for r in range(R)]
with ThreadPoolExecutor(6) as ex:
    neg = list(ex.map(neg_job, jobs)); geo = list(ex.map(geo_job, jobs))
json.dump({"neg": neg, "geo": geo}, open("experiments/e1b_e3b_raw.json", "w"))
sums = {k: [p[k] + p[k + "_neg"] for _, _, p in neg] for k in S}
for k, v in sums.items(): print(f"E1b {k:9s} p+p_neg: mean {np.mean(v):.2f}  range [{min(v):.2f}, {max(v):.2f}]  |dev|>0.1: {np.mean(np.abs(np.array(v)-1)>0.1):.2f}")
lab = {"mr": "mr", "t1": "t1_flag", "unstable": "rhf_unstable"}
for k in lab:
    p = np.array([np.mean([x[2][k] for x in geo if x[0] == i]) for i in range(len(rows))]); y = np.array([r[lab[k]] for r in rows]).astype(int)
    print(f"E3b {k:9s} geometry-only AUROC {roc_auc_score(y, p):.2f}  Brier {np.mean((p-y)**2):.3f}")
