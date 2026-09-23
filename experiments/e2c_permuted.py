"""E2c (after review round 1, R2-M4): the H2/STO-3G Jordan-Wigner Hamiltonian with qubits relabelled so that the
Hartree-Fock state is |0101> instead of the textbook |1100>. Relabelling map: old qubit q -> new qubit PERM[q]."""
import json, sys
import numpy as np
sys.path[:0] = ["harness", "experiments"]
from jev import ask, choice
PERM = [3, 1, 2, 0]
d = json.load(open("experiments/e2_raw.json"))["h2_exact"]
def perm_term(t):
    if t == "I": return "I"
    ops = [(o[0], int(o[1:])) for o in t.split()]
    return " ".join(f"{p}{PERM[q]}" for p, q in sorted(ops, key=lambda x: PERM[x[1]]))
def perm_state(b):
    new = ["0"] * 4
    for q, bit in enumerate(b): new[PERM[q]] = bit
    return "".join(new)
out = []
for ex in d:
    terms = {perm_term(t): c for t, c in ex["terms"].items()}
    st = {"system": "4-qubit Hamiltonian (Jordan-Wigner encoding, hartree units), qubit order q0 q1 q2 q3",
          "pauli_terms": terms, "particle_number": 2}
    w = {perm_state(b): v for b, v in ex["weights"].items()}
    crit = {b: f"computational basis state |{b}>" for b in sorted(w)}
    for r in range(3):
        a = ask(st, {"dom": choice("Which computational basis state has the largest weight in the 2-particle ground state?", crit)}, rep=r)["answers"]["dom"]
        out.append({"R": ex["R"], "rep": r, "p": a["probabilities"], "weights": w})
json.dump(out, open("experiments/e2c_raw.json", "w"), indent=1)
top = [max(x["p"], key=x["p"].get) for x in out]; dom = [max(x["weights"], key=x["weights"].get) for x in out]
print("dominant state after relabelling:", set(dom), "| Jev top picks:", {t: top.count(t) for t in set(top)})
print("mean p on true dominant state:", np.mean([x["p"].get(dm, 0) for x, dm in zip(out, dom)]).round(2),
      "mean p on |1100>:", np.mean([x["p"].get("1100", 0) for x in out]).round(2))
