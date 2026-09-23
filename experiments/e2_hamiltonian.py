"""E2: can Jev represent a Hamiltonian given in context?
(a) random classical 2-local Ising models, n=3..7, choice over all 2^n configurations;
(b) Jordan-Wigner H2/STO-3G qubit Hamiltonian: dominant basis state and energy read-out."""
import itertools, json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, "harness")
from jev import ask, choice, score

R = 3
rng = np.random.default_rng(20260923)

def ising_instances():
    for n in range(3, 8):
        for k in range(10):
            h = rng.uniform(-1, 1, n).round(2)
            J = {(i, j): round(float(rng.uniform(-1, 1)), 2) for i in range(n) for j in range(i+1, n)}
            yield n, k, h, J

def energy(s, h, J):
    return float(sum(h[i]*s[i] for i in range(len(s))) + sum(v*s[i]*s[j] for (i, j), v in J.items()))

def lab(s): return "".join("+" if x > 0 else "-" for x in s)

def ising_job(inst, r):
    n, k, h, J = inst
    confs = list(itertools.product([1, -1], repeat=n))
    st = {"hamiltonian": "H(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j with spins s_i in {-1,+1}",
          "h": {f"h_{i}": float(h[i]) for i in range(n)},
          "J": {f"J_{i}{j}": v for (i, j), v in J.items()}}
    crit = {lab(s): "spins (" + ", ".join(f"s_{i}={s[i]:+d}" for i in range(n)) + ")" for s in confs}
    a = ask(st, {"gs": choice("Which spin configuration is the ground state, i.e. minimises H(s)?", crit)}, rep=r)["answers"]["gs"]
    return {"n": n, "k": k, "rep": r, "E": {lab(s): energy(s, h, J) for s in confs}, "p": a["probabilities"]}

def h2_jobs():
    from openfermion import MolecularData, jordan_wigner, get_sparse_operator
    from openfermionpyscf import run_pyscf
    out = []
    for R_ in [0.5, 0.74, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        m = run_pyscf(MolecularData([("H", (0, 0, 0)), ("H", (0, 0, R_))], "sto-3g", 1, 0), run_fci=True)
        qh = jordan_wigner(m.get_molecular_hamiltonian())
        H = get_sparse_operator(qh, n_qubits=4).toarray()
        w, v = np.linalg.eigh(H)
        # ground state within the 2-electron sector
        nocc = [bin(b).count("1") for b in range(16)]
        idx = [i for i in range(len(w)) if abs(sum(abs(v[b, i])**2 for b in range(16) if nocc[b] == 2) - 1) < 1e-6]
        g = idx[0]
        weights = {format(b, "04b"): float(abs(v[b, g])**2) for b in range(16)}
        terms = {" ".join(f"{p}{q}" for q, p in t) or "I": round(float(np.real(c)), 6) for t, c in qh.terms.items()}
        out.append({"R": R_, "E_exact": float(w[g]), "E_fci": float(m.fci_energy), "weights": weights, "terms": terms})
    return out

def h2_ask(d, r):
    st = {"system": "4-qubit Hamiltonian (Jordan-Wigner encoding, hartree units), qubit order q0 q1 q2 q3",
          "pauli_terms": d["terms"], "particle_number": 2}
    crit = {b: f"computational basis state |{b}>" for b in d["weights"]}
    levels = [f"{e:.2f} hartree" for e in np.linspace(-1.20, -0.75, 10)]
    a = ask(st, {"dom": choice("Which computational basis state has the largest weight in the 2-particle ground state?", crit),
                 "E": score("What is the ground-state energy of this Hamiltonian (2-particle sector)?", levels)}, rep=r)["answers"]
    lv = np.linspace(-1.20, -0.75, 10)
    # score is normalised to [0,1] over the levels (checked in E0 output); map back to hartree
    return {"R": d["R"], "rep": r, "p_dom": a["dom"]["probabilities"], "score": a["E"]["score"],
            "E_read": float(lv[0] + a["E"]["score"]*(lv[-1]-lv[0])), "E_probs": a["E"]["probabilities"]}

if __name__ == "__main__":
    insts = list(ising_instances())
    with ThreadPoolExecutor(6) as ex:
        ising = list(ex.map(lambda t: ising_job(*t), [(i, r) for i in insts for r in range(R)]))
    H2 = h2_jobs()
    h2 = [h2_ask(d, r) for d in H2 for r in range(R)]
    json.dump({"ising": ising, "h2_exact": H2, "h2": h2}, open("experiments/e2_raw.json", "w"), indent=1)
    print("ising", len(ising), "h2", len(h2))
