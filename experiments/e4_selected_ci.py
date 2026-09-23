"""E4: Jev as an importance prior for selected CI in a CAS(6e,5o) space (100 determinants, <= 255 options).
Rankings: Jev choice probabilities, oracle |c|^2, first-order perturbative |<D|H|0>|^2/(H_DD-E0), lowest
diagonal energy, random. Metric: variational energy error of the k-determinant subspace vs full CASCI."""
import json, sys, itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from jev import ask, choice
from qc_labels import MOLS

NCAS, NELE = 5, 6
SYSTEMS = [("H2O", 1.0), ("H2O", 2.0), ("N2", 1.0), ("N2", 1.5), ("F2", 1.5), ("CH4", 2.0),
           ("HF", 2.0), ("C2", 1.0), ("CO", 1.5), ("NH3", 2.0)]
R = 3

def build(name, f):
    mol = gto.M(atom=MOLS[name](f), basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, NCAS, NELE)
    h1, ecore = mc.get_h1eff(); eri = ao2mo.restore(1, mc.get_h2eff(), NCAS)
    na = nb = NELE // 2
    strs = fci.cistring.make_strings(range(NCAS), na)
    dets = list(itertools.product(range(len(strs)), range(len(strs))))
    N = len(dets); H = np.zeros((N, N))
    for j, (a, b) in enumerate(dets):
        v = np.zeros((len(strs), len(strs))); v[a, b] = 1
        H[:, j] = fci.direct_spin1.contract_2e(fci.direct_spin1.absorb_h1e(h1, eri, NCAS, (na, nb), .5), v, NCAS, (na, nb)).ravel()
    H = (H + H.T) / 2 + ecore*np.eye(N)
    S2 = np.zeros((N, N))
    for j, (a, b) in enumerate(dets):
        v = np.zeros((len(strs), len(strs))); v[a, b] = 1
        S2[:, j] = fci.spin_op.contract_ss(v, NCAS, (na, nb)).ravel()
    S2 = (S2 + S2.T) / 2
    # spin penalty keeps every (sub)space ground state a singlet; energies are reported as <H>
    Hp = H + 1.0*S2
    w, v = np.linalg.eigh(Hp)
    w = np.array([v[:, 0] @ H @ v[:, 0]])
    def occ(a, b):
        out = []
        for o in range(NCAS):
            ia, ib = (int(strs[a]) >> o) & 1, (int(strs[b]) >> o) & 1
            out.append({(1, 1): "2", (1, 0): "a", (0, 1): "b", (0, 0): "0"}[(ia, ib)])
        return "".join(out)
    labels = [occ(a, b) for a, b in dets]
    ref = labels.index("222" + "00")
    act = mc.ncore + np.arange(NCAS)
    return {"name": name, "stretch": f, "H": H, "Hp": Hp, "exc": None, "E_cas": float(w[0]), "c": v[:, 0], "labels": labels, "ref": ref,
            "orb_e": mf.mo_energy[act].round(4).tolist(), "mol": mol}

def e_sub(s, idx):
    w, v = np.linalg.eigh(s["Hp"][np.ix_(idx, idx)])
    x = v[:, 0]; return float(x @ s["H"][np.ix_(idx, idx)] @ x)

def jev_rank(sys_, r):
    lab, ref = sys_["labels"], sys_["ref"]
    st = {"system": f"{sys_['name']} molecule, bonds at {sys_['stretch']:.1f} x equilibrium, cc-pVDZ, RHF orbitals",
          "active_space": f"CAS({NELE}e,{NCAS}o); active orbital energies (hartree, ascending): {sys_['orb_e']}",
          "determinant_notation": "one character per active orbital in ascending energy: 2 doubly occupied, a alpha, b beta, 0 empty",
          "reference_determinant": lab[ref]}
    crit = {l: f"determinant {l}" for i, l in enumerate(lab) if i != ref}
    a = ask(st, {"imp": choice("Excluding the reference determinant, which determinant has the largest weight in the exact CASCI ground state?", crit)}, rep=r)
    return a["answers"]["imp"]["probabilities"]

def curves(s, jev_p, rng):
    H, c, ref, N = s["H"], s["c"], s["ref"], len(s["labels"])
    others = [i for i in range(N) if i != ref]
    E0 = H[ref, ref]
    lab = s["labels"]; oe = s["orb_e"]
    def exc(i):  # excitation level w.r.t. reference, then orbital-energy cost of the excitation
        occ = lambda ch: {"2": 2, "a": 1, "b": 1, "0": 0}[ch]
        d = [occ(x) - occ(y) for x, y in zip(lab[i], lab[ref])]
        return (sum(abs(t) for t in d) // 2, sum(t*e for t, e in zip(d, oe)))
    rank = {"excitation": sorted(others, key=exc),
            "oracle": sorted(others, key=lambda i: -c[i]**2),
            "pt1": sorted(others, key=lambda i: -(H[i, ref]**2/max(H[i, i]-E0, 1e-3))),
            "diag": sorted(others, key=lambda i: H[i, i]),
            "jev": sorted(others, key=lambda i: -jev_p.get(s["labels"][i], 0) + 1e-9*rng.random()),}
    cv = {k: [1000*(e_sub(s, [ref]+r[:k]) - s["E_cas"]) for k in range(0, N)] for k, r in rank.items()}
    # random baseline: geometric mean over 30 permutations
    rnd = [[1000*(e_sub(s, [ref]+list(pm[:k])) - s["E_cas"]) for k in range(0, N)]
           for pm in (rng.permutation(others) for _ in range(30))]
    # Jev: ~60% of options come back as exactly 0 -> many ties; average over 30 random tie-breaks
    jr = [[1000*(e_sub(s, [ref]+sorted(others, key=lambda i: (-jev_p.get(s["labels"][i], 0), tb[i]))[:k]) - s["E_cas"])
           for k in range(0, N)] for tb in (rng.random(N) for _ in range(30))]
    cv["jev"] = np.exp(np.mean(np.log(np.maximum(jr, 1e-6)), 0)).tolist()
    cv["random"] = np.exp(np.mean(np.log(np.maximum(rnd, 1e-6)), 0)).tolist()
    return cv

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    systems = [build(n, f) for n, f in SYSTEMS]
    with ThreadPoolExecutor(6) as ex:
        jp = list(ex.map(lambda t: jev_rank(*t), [(s, r) for s in systems for r in range(R)]))
    out = []
    for i, s in enumerate(systems):
        ps = jp[i*R:(i+1)*R]
        mean_p = {l: np.mean([p.get(l, 0) for p in ps]) for l in s["labels"]}
        cv = curves(s, mean_p, rng)
        out.append({"name": s["name"], "stretch": s["stretch"], "c0sq": float(s["c"][s["ref"]]**2),
                    "E_cas": s["E_cas"], "curves": cv, "jev_probs": ps,
                    "spearman_jev_vs_c2": float(__import__("scipy.stats").stats.spearmanr(
                        [mean_p[l] for i2, l in enumerate(s["labels"]) if i2 != s["ref"]],
                        [s["c"][i2]**2 for i2 in range(len(s["labels"])) if i2 != s["ref"]]).statistic)})
        k_at = {k: next((j for j, e in enumerate(v) if e < 1.6), None) for k, v in cv.items()}
        print(f"{s['name']:4s} x{s['stretch']}  c0^2={out[-1]['c0sq']:.2f}  rho(jev,|c|^2)={out[-1]['spearman_jev_vs_c2']:+.2f}  k to 1.6 mEh: {k_at}")
    json.dump(out, open("experiments/e4_raw.json", "w"), indent=1)
