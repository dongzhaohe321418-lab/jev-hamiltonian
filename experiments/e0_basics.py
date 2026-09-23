"""E0: determinism, output quantisation, and cross-talk between parallel questions."""
import json, sys
sys.path.insert(0, "harness")
from jev import ask, noul, choice

state = {"molecule": "N2", "geometry": "bond length 2.2 angstrom (stretched, equilibrium 1.10)",
         "basis": "cc-pVDZ"}
Q = {"mr": noul("Does this system have strong multireference (static correlation) character?"),
     "rhf": noul("Is restricted Hartree-Fock a qualitatively correct reference here?"),
     "ccsdt": noul("Will CCSD(T) be within 1 kcal/mol of the exact energy?"),
     "spin": choice("Ground-state spin multiplicity?", {"1": "singlet", "3": "triplet", "5": "quintet"})}
out = {"repeats": [], "alone": {}, "together": None}
for r in range(5):
    out["repeats"].append(ask(state, Q, rep=r)["answers"])
out["together"] = out["repeats"][0]
for k, q in Q.items():
    out["alone"][k] = ask(state, {k: q})["answers"][k]
json.dump(out, open("experiments/e0_out.json", "w"), indent=1)
for k in Q:
    reps = [a[k].get("noul", a[k].get("probabilities")) for a in out["repeats"]]
    print(k, "repeats:", reps, "| alone:", out["alone"][k].get("noul", out["alone"][k].get("probabilities")))
