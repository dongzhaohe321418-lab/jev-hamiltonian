"""Robustness extensions (added 2026-09-23 after first review round; exploratory).
E1c: two-fact conditionals p(x | y=v, z=w) for cycle / path-independence tests.
E1p: full E1 conditional protocol with an independently worded paraphrase of all five statements.
E3p: marginals for the three E3 labels under four paraphrases.
E2p: Ising ground-state question under two alternative wordings (as-stated sign convention)."""
import itertools, json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from jev import ask, noul, choice
from e1e3_collect import S, FACT, state_of
import e2_hamiltonian as e2

rows = json.load(open("experiments/qc_labels.json"))
S_P = {"mr": "Static (nondynamical) correlation is strong here: in the exact wavefunction the Hartree-Fock configuration has a weight below 0.9.",
       "rhf_ok": "A single closed-shell RHF determinant gives a qualitatively right description of this molecule's ground state.",
       "t1": "The T1 diagnostic from a CCSD calculation on this system is larger than 0.02.",
       "unstable": "There is a broken-symmetry unrestricted (UHF) solution lying below the RHF energy, so RHF is unstable.",
       "ccsdt_ok": "The error of CCSD(T) relative to full CI in this basis is under 1 kcal/mol."}
E3P = {"mr": [S_P["mr"], "Would a multireference method be needed to describe this system's ground state qualitatively?",
              "Is the Hartree-Fock determinant weight in the exact ground state below 90%?",
              "This molecule, at this geometry, is a strongly correlated (multireference) case."],
       "t1": [S_P["t1"], "Is the CCSD T1 diagnostic above the 0.02 threshold for this system?",
              "Coupled-cluster singles amplitudes are large here: T1 diagnostic > 0.02.",
              "A CCSD calculation would flag this system with T1 greater than 0.02."],
       "unstable": [S_P["unstable"], "Does a stability analysis of the RHF solution find a lower-energy UHF solution?",
                    "Is the restricted Hartree-Fock wavefunction unstable toward spin symmetry breaking?",
                    "Breaking spin symmetry (going from RHF to UHF) lowers the SCF energy for this system."]}

def two_fact_state(row, facts, texts):
    st = state_of(row)
    st["known"] = " ".join(FACT[v].format(texts[k]) for k, v in facts)
    return st

def jobs():
    keys = list(S)
    for i, row in enumerate(rows):
        for r in range(2):                                  # E1c
            for y, z in itertools.combinations(keys, 2):
                for vy, vz in itertools.product([True, False], repeat=2):
                    qs = {k: noul(S[k]) for k in keys if k not in (y, z)}
                    yield ("e1c", i, r, ((y, vy), (z, vz)), two_fact_state(row, ((y, vy), (z, vz)), S), qs)
        for r in range(2):                                  # E1p
            yield ("e1p", i, r, None, state_of(row), {k: noul(t) for k, t in S_P.items()})
            for k in keys:
                for v in (True, False):
                    st = state_of(row); st["known"] = FACT[v].format(S_P[k])
                    yield ("e1p", i, r, ((k, v),), st, {q: noul(t) for q, t in S_P.items() if q != k})
        for r in range(2):                                  # E3p
            for j in range(4):
                yield ("e3p", i, r, j, state_of(row), {k: noul(v[j]) for k, v in E3P.items()})

WORDINGS = ["Which spin configuration gives the lowest value of H(s)?",
            "Find the minimum-energy assignment of the spins for this Hamiltonian."]
def e2p_jobs():
    for inst in e2.ising_instances():
        n, k, h, J = inst
        confs = list(itertools.product([1, -1], repeat=n))
        st = {"hamiltonian": "H(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j with spins s_i in {-1,+1}",
              "h": {f"h_{i}": float(h[i]) for i in range(n)}, "J": {f"J_{i}{j}": v for (i, j), v in J.items()}}
        crit = {e2.lab(s): "spins (" + ", ".join(f"s_{i}={s[i]:+d}" for i in range(n)) + ")" for s in confs}
        E = {e2.lab(s): e2.energy(s, h, J) for s in confs}
        for w, q in enumerate(WORDINGS):
            yield ("e2p", (n, k), 0, w, st, {"gs": choice(q, crit)}, E)

def run(j):
    kind, i, r, cond, st, qs = j[:6]
    a = ask(st, qs, rep=r)["answers"]
    out = {"kind": kind, "case": i, "rep": r, "cond": cond}
    if kind == "e2p": out.update(p=a["gs"]["probabilities"], E=j[6])
    else: out["p"] = {k: v["noul"] for k, v in a.items()}
    return out

if __name__ == "__main__":
    J = list(jobs()) + list(e2p_jobs())
    print("jobs", len(J), flush=True)
    with ThreadPoolExecutor(6) as ex:
        out = []
        for n, res in enumerate(ex.map(run, J)):
            out.append(res)
            if n % 500 == 0: print(n, flush=True)
    json.dump(out, open("experiments/e8_raw.json", "w"))
    print("done", len(out))
