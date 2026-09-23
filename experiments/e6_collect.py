"""E6 Jev collection: p('CCSD(T) within 1 kcal/mol of FCI in this basis') for every E6 case, R=3.
State format = e1e3_collect.state_of with the basis name replaced by the actual basis (STO-3G / 6-31G);
statement = e1e3_collect.S['ccsdt_ok'] verbatim, asked alone. Prompts depend only on name/basis/geometry/stretch."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from jev import ask, noul
from e1e3_collect import S, state_of
from e6_labels import SETS, STRETCH

R = 3
BASIS_NAME = {"sto-3g": "STO-3G", "6-31g": "6-31G"}

def e6_state(name, basis, f, geom):
    st = state_of({"name": name, "stretch": f, "geometry": geom})
    st["method_context"] = st["method_context"].replace("cc-pVDZ", BASIS_NAME[basis])
    return st

if __name__ == "__main__":
    cases = [(n, b, f, g(f)) for n, b, g in SETS for f in STRETCH]
    def run(t):
        i, r = t
        n, b, f, geo = cases[i]
        a = ask(e6_state(n, b, f, geo), {"ccsdt_ok": noul(S["ccsdt_ok"])}, rep=r)["answers"]
        return {"name": n, "basis": b, "stretch": f, "rep": r, "p": a["ccsdt_ok"]["noul"]}
    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(run, [(i, r) for i in range(len(cases)) for r in range(R)]))
    json.dump({"answers": out, "example_state": e6_state(*cases[0])}, open("experiments/e6_raw.json", "w"), indent=0)
    print("done", len(out))
