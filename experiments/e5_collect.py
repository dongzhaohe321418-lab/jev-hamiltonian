"""E5 Jev collection.
(a) the E3 marginal query, unchanged: state_of(row) from e1e3_collect and all five statements S (as in the
    E3 marginal call), R=3 repeats, on the 124 E5 cases;
(b) geometry-only variant, exactly as E3b (experiments/e1b_e3b.py geo_job): geometry + method context only,
    the three label statements;
(c) spin-state Choice over multiplicities for the 32 species of e5_spin_labels.py, R=3.
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from jev import ask, noul, choice
from e1e3_collect import S, state_of
from e5_spin_labels import SPECIES

R = 3
Z = {"H": 1, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17}
SPIN_Q = choice("Which spin multiplicity does the ground (lowest-energy) electronic state of this species have?",
                {"1": "singlet (S=0)", "3": "triplet (S=1)", "5": "quintet (S=2)"})

def spin_state(name):
    q, kind, atoms, _, _ = SPECIES[name]
    atoms_full = atoms if kind != "xy2" else [atoms[0], atoms[1], atoms[1]]
    n = sum(Z[a] for a in atoms_full) - q
    desc = "atom" if kind == "atom" else ("molecular anion" if q < 0 else "molecular cation" if q > 0 else "molecule")
    return {"system": f"{name} ({desc})", "charge": q, "electron_count": n,
            "method_context": "electronic structure; compare the lowest singlet, triplet and quintet states, each at its own optimized geometry"}

if __name__ == "__main__":
    rows = json.load(open("experiments/e5_labels.json"))
    def full(t):
        i, r = t
        a = ask(state_of(rows[i]), {k: noul(v) for k, v in S.items()}, rep=r)["answers"]
        return {"case": i, "rep": r, "p": {k: v["noul"] for k, v in a.items()}}
    def geo(t):
        i, r = t
        st = {"geometry_xyz_angstrom": rows[i]["geometry"], "method_context": "electronic structure, cc-pVDZ basis, singlet reference"}
        a = ask(st, {k: noul(S[k]) for k in ("mr", "t1", "unstable")}, rep=r)["answers"]
        return {"case": i, "rep": r, "p": {k: v["noul"] for k, v in a.items()}}
    def spin(t):
        name, r = t
        a = ask(spin_state(name), {"mult": SPIN_Q}, rep=r)["answers"]["mult"]
        return {"name": name, "rep": r, "choice": a["choice"], "probabilities": a["probabilities"]}
    jobs = [(i, r) for i in range(len(rows)) for r in range(R)]
    with ThreadPoolExecutor(6) as ex:
        F = list(ex.map(full, jobs)); print("full", len(F), flush=True)
        G = list(ex.map(geo, jobs)); print("geo", len(G), flush=True)
        SP = list(ex.map(spin, [(n, r) for n in SPECIES for r in range(R)])); print("spin", len(SP), flush=True)
    json.dump({"full": F, "geo": G, "spin": SP, "spin_question": SPIN_Q,
               "spin_states": {n: spin_state(n) for n in SPECIES}}, open("experiments/e5_raw.json", "w"))
