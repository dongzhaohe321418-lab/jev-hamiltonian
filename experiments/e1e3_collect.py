"""Collect E1 (conditional compatibility) and E3 (QC decisions) probabilities from Jev."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "harness")
from jev import ask, noul

R = 3
S = {"mr": "The ground state has strong multireference (static correlation) character: the Hartree-Fock determinant carries less than 90% of the exact wavefunction weight.",
     "rhf_ok": "Restricted Hartree-Fock is a qualitatively correct single-determinant reference for this system.",
     "t1": "The CCSD T1 diagnostic exceeds 0.02.",
     "unstable": "The RHF solution is unstable: a spin-unrestricted (UHF) solution with lower energy exists.",
     "ccsdt_ok": "CCSD(T) will be within 1 kcal/mol of the exact (full CI) energy in this basis."}
FACT = {True: "Established fact (computed): {} This statement is TRUE.",
        False: "Established fact (computed): {} This statement is FALSE."}

def state_of(row, cond=None):
    st = {"system": f"{row['name']} molecule", "method_context": "electronic structure, cc-pVDZ basis, singlet reference",
          "geometry_xyz_angstrom": row["geometry"],
          "geometry_note": f"all bonds stretched to {row['stretch']:.1f} x equilibrium length"}
    if cond:
        k, v = cond; st["known"] = FACT[v].format(S[k])
    return st

def jobs(rows):
    for i, row in enumerate(rows):
        for r in range(R):
            yield (i, None, r, state_of(row), {k: noul(t) for k, t in S.items()})
            for k in S:
                for v in (True, False):
                    qs = {q: noul(t) for q, t in S.items() if q != k}
                    yield (i, (k, v), r, state_of(row, (k, v)), qs)

rows = json.load(open("experiments/qc_labels.json"))
J = list(jobs(rows))
def run(j):
    i, cond, r, st, qs = j
    a = ask(st, qs, rep=r)["answers"]
    return {"case": i, "cond": cond, "rep": r, "p": {k: v["noul"] for k, v in a.items()}}
with ThreadPoolExecutor(6) as ex:
    out = []
    for n, res in enumerate(ex.map(run, J)):
        out.append(res)
        if n % 200 == 0: print(n, "/", len(J), flush=True)
json.dump(out, open("experiments/e1e3_raw.json", "w"))
print("done", len(out))
