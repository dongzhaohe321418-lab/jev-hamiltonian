"""E2b (exploratory): same energy landscapes as E2, presented in the textbook convention
H(s) = -sum h'_i s_i - sum J'_ij s_i s_j with h' = -h, J' = -J. Identical energies, opposite coefficient signs."""
import itertools, json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from jev import ask, choice
import e2_hamiltonian as e2

insts = list(e2.ising_instances())   # same seed -> same instances as E2
def job(inst, r):
    n, k, h, J = inst
    confs = list(itertools.product([1, -1], repeat=n))
    st = {"hamiltonian": "H(s) = - sum_i h_i s_i - sum_{i<j} J_ij s_i s_j with spins s_i in {-1,+1}",
          "h": {f"h_{i}": float(-h[i]) for i in range(n)},
          "J": {f"J_{i}{j}": -v for (i, j), v in J.items()}}
    crit = {e2.lab(s): "spins (" + ", ".join(f"s_{i}={s[i]:+d}" for i in range(n)) + ")" for s in confs}
    a = ask(st, {"gs": choice("Which spin configuration is the ground state, i.e. minimises H(s)?", crit)}, rep=r)["answers"]["gs"]
    return {"n": n, "k": k, "rep": r, "E": {e2.lab(s): e2.energy(s, h, J) for s in confs}, "p": a["probabilities"]}
with ThreadPoolExecutor(6) as ex:
    out = list(ex.map(lambda t: job(*t), [(i, r) for i in insts for r in range(3)]))
json.dump(out, open("experiments/e2b_raw.json", "w"), indent=1)
import scipy.stats as st
for n in range(3, 8):
    rs = [x for x in out if x["n"] == n]
    top = np.mean([max(x["p"], key=x["p"].get) == min(x["E"], key=x["E"].get) for x in rs])
    rho = np.mean([st.spearmanr([-x["E"][k] for k in x["E"]], [x["p"].get(k, 0) for k in x["E"]]).statistic for x in rs])
    print(n, "top1", round(top, 2), "chance", round(1/2**n, 3), "rho", round(rho, 2))
