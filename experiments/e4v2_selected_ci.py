"""E4 v2 (after review round 1, R2-M2/M3). Changes from e4_selected_ci.py:
- internally stable RHF orbitals; active space per system chosen so no degenerate shell is split (<= 255 determinants);
- spin-complete selection: determinants are added as whole spatial-occupation families (all M_s=0 spin couplings of
  the same orbital occupation), and the subspace energy is the lowest root with <S^2> < 0.1 (no penalty);
- every ranking breaks ties at random, 30 draws, geometric-mean curve;
- comparators: oracle |c|^2, first-order |H_i0|^2/(H_ii-H_00)^2, Epstein-Nesbet second-order contribution
  |H_i0|^2/(H_ii-H_00) (the CIPSI criterion), iterative CIPSI, excitation rank, random;
- Jev elicited two ways: (top1) the original Choice "largest weight" distribution; (noul) one Noul per determinant
  "weight above 1%", all asked in one call;
- question-aligned metrics: whether Jev's top family is the oracle's top family, and the oracle top's rank under Jev;
- cross-system similarity of Jev's rankings (same notation across systems)."""
import itertools, json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.stats as st
from pyscf import gto, mcscf, fci, ao2mo
sys.path[:0] = ["harness", "experiments"]
from jev import ask, choice, noul
from qc_labels import MOLS
from qc_labels_v2 import stable_rhf

SYSTEMS = [("H2O", 1.0), ("H2O", 2.0), ("N2", 1.0), ("N2", 1.5), ("F2", 1.5), ("CH4", 2.0),
           ("HF", 2.0), ("C2", 1.0), ("CO", 1.5), ("NH3", 2.0)]
WINDOWS = [(6, 5), (4, 5), (6, 6), (4, 4), (2, 5), (8, 5), (4, 6), (2, 4)]   # (electrons, orbitals), tried in order
R, NDRAW = 3, 30
rng = np.random.default_rng(7)

def pick_window(mf, nelec):
    e = mf.mo_energy; nocc = nelec // 2
    for ne, no in WINDOWS:
        ncore = nocc - ne // 2; lo, hi = ncore, ncore + no
        if ncore < 0 or hi > len(e): continue
        ndet = len(fci.cistring.make_strings(range(no), ne // 2)) ** 2
        if ndet > 255: continue
        if (lo > 0 and abs(e[lo] - e[lo - 1]) < 1e-4) or (hi < len(e) and abs(e[hi] - e[hi - 1]) < 1e-4): continue
        return ne, no
    raise RuntimeError("no symmetric window")

def build(name, f):
    mol = gto.M(atom=MOLS[name](f), basis="cc-pvdz", verbose=0)
    mf = stable_rhf(mol)
    ne, no = pick_window(mf, mol.nelectron)
    mc = mcscf.CASCI(mf, no, ne)
    h1, ecore = mc.get_h1eff(); eri = ao2mo.restore(1, mc.get_h2eff(), no)
    na = nb = ne // 2
    strs = fci.cistring.make_strings(range(no), na); ns = len(strs)
    dets = list(itertools.product(range(ns), range(ns))); N = len(dets)
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, no, (na, nb), .5)
    H = np.zeros((N, N)); S2 = np.zeros((N, N))
    for j, (a, b) in enumerate(dets):
        v = np.zeros((ns, ns)); v[a, b] = 1
        H[:, j] = fci.direct_spin1.contract_2e(h2, v, no, (na, nb)).ravel()
        S2[:, j] = fci.spin_op.contract_ss(v, no, (na, nb)).ravel()
    H = (H + H.T) / 2 + ecore * np.eye(N); S2 = (S2 + S2.T) / 2
    def occ(a, b):
        return "".join({(1, 1): "2", (1, 0): "a", (0, 1): "b", (0, 0): "0"}[((int(strs[a]) >> o) & 1, (int(strs[b]) >> o) & 1)] for o in range(no))
    labels = [occ(a, b) for a, b in dets]
    fam = [l.replace("a", "1").replace("b", "1") for l in labels]
    ref = labels.index("2" * na + "0" * (no - na))
    E_exact, c = singlet_ground(H, S2, list(range(N)))
    act = mc.ncore + np.arange(no)
    return {"name": name, "stretch": f, "H": H, "S2": S2, "E": E_exact, "c": c, "labels": labels, "fam": fam, "ref": ref,
            "cas": f"({ne}e,{no}o)", "orb_e": mf.mo_energy[act].round(4).tolist(), "ne": ne, "no": no}

def singlet_ground(H, S2, idx):
    w, v = np.linalg.eigh(H[np.ix_(idx, idx)])
    ss = np.einsum("ik,ij,jk->k", v, S2[np.ix_(idx, idx)], v)
    k = next((k for k in range(len(w)) if ss[k] < 0.1), None)
    if k is None: return np.inf, None
    return float(w[k]), v[:, k]

def curve(s, fam_order):
    """Add families in order; energy error (mEh) after each family, indexed by number of determinants."""
    idx = [i for i, f in enumerate(s["fam"]) if f == s["fam"][s["ref"]]]
    errs = np.full(len(s["labels"]) + 1, np.nan); errs[len(idx)] = 1000 * (singlet_ground(s["H"], s["S2"], idx)[0] - s["E"])
    for f in fam_order:
        if f == s["fam"][s["ref"]]: continue
        idx += [i for i, g in enumerate(s["fam"]) if g == f]
        errs[len(idx)] = 1000 * (singlet_ground(s["H"], s["S2"], idx)[0] - s["E"])
    # step function over k = number of determinants: carry the last completed subspace forward
    first = errs[len([i for i, f in enumerate(s["fam"]) if f == s["fam"][s["ref"]]])]
    last = first
    for k in range(len(errs)):
        if np.isnan(errs[k]): errs[k] = last
        else: last = errs[k]
    return errs

def fam_rank(s, score, draws=NDRAW):
    """Families ordered by max member score (higher = earlier), random tie-breaks; returns list of orders."""
    fams = sorted(set(s["fam"])); best = {f: max(score[i] for i, g in enumerate(s["fam"]) if g == f) for f in fams}
    return [sorted(fams, key=lambda f: (-best[f], tb[f])) for tb in ({f: rng.random() for f in fams} for _ in range(draws))]

def cipsi_order(s):
    fams = sorted(set(s["fam"])); cur = [f for f in [s["fam"][s["ref"]]]]; order = []
    while len(cur) < len(fams):
        idx = [i for i, g in enumerate(s["fam"]) if g in cur]
        E, v = singlet_ground(s["H"], s["S2"], idx)
        best, bf = -1, None
        for f in fams:
            if f in cur: continue
            mem = [i for i, g in enumerate(s["fam"]) if g == f]
            sc = max((s["H"][i, idx] @ v) ** 2 / max(s["H"][i, i] - E, 1e-6) for i in mem)
            if sc > best: best, bf = sc, f
        cur.append(bf); order.append(bf)
    return order

def geo(curves): return np.exp(np.mean(np.log(np.maximum(curves, 1e-4)), 0))

def jev_state(s):
    return {"system": f"{s['name']} molecule, bonds at {s['stretch']:.1f} x equilibrium, cc-pVDZ, RHF orbitals",
            "active_space": f"CAS({s['ne']}e,{s['no']}o); active orbital energies (hartree, ascending): {s['orb_e']}",
            "determinant_notation": "one character per active orbital in ascending energy: 2 doubly occupied, a alpha, b beta, 0 empty",
            "reference_determinant": s["labels"][s["ref"]]}

def jev_ask(s, r):
    lab, ref = s["labels"], s["ref"]; st_ = jev_state(s)
    top1 = ask(st_, {"imp": choice("Excluding the reference determinant, which determinant has the largest weight in the exact CASCI ground state?",
                                    {l: f"determinant {l}" for i, l in enumerate(lab) if i != ref})}, rep=r)["answers"]["imp"]["probabilities"]
    qs = {l: noul(f"Determinant {l} has a weight (squared coefficient) above 0.01 in the exact CASCI ground state.") for i, l in enumerate(lab) if i != ref}
    nl = {k: v["noul"] for k, v in ask(st_, qs, rep=r)["answers"].items()}
    return top1, nl

if __name__ == "__main__":
    systems = [build(n, f) for n, f in SYSTEMS]
    with ThreadPoolExecutor(4) as ex:
        J = list(ex.map(lambda t: jev_ask(*t), [(s, r) for s in systems for r in range(R)]))
    out = []
    for si, s in enumerate(systems):
        lab, H, c, ref, N = s["labels"], s["H"], s["c"], s["ref"], len(s["labels"])
        top1 = {l: np.mean([J[si * R + r][0].get(l, 0) for r in range(R)]) for l in lab}
        nl = {l: np.mean([J[si * R + r][1].get(l, 0) for r in range(R)]) for l in lab}
        d = np.array([H[i, i] - H[ref, ref] for i in range(N)]); h0 = H[:, ref]
        scores = {"oracle": c ** 2, "first_order": np.where(np.arange(N) == ref, 0, h0 ** 2 / np.maximum(d, 1e-6) ** 2),
                  "en_pt2": np.where(np.arange(N) == ref, 0, h0 ** 2 / np.maximum(d, 1e-6)),
                  "excitation": np.array([-sum(abs({"2": 2, "a": 1, "b": 1, "0": 0}[x] - {"2": 2, "a": 1, "b": 1, "0": 0}[y]) for x, y in zip(l, lab[ref])) / 2
                                          - 1e-3 * sum(({"2": 2, "a": 1, "b": 1, "0": 0}[x] - {"2": 2, "a": 1, "b": 1, "0": 0}[y]) * e for x, y, e in zip(l, lab[ref], s["orb_e"])) for l in lab]),
                  "jev_top1": np.array([top1[l] for l in lab]), "jev_noul": np.array([nl[l] for l in lab]), "random": np.zeros(N)}
        cv = {k: geo([curve(s, o) for o in fam_rank(s, v)]).tolist() for k, v in scores.items()}
        cv["cipsi"] = curve(s, cipsi_order(s)).tolist()
        oth = [i for i in range(N) if i != ref]
        ofam = s["fam"][max(oth, key=lambda i: c[i] ** 2)]
        jtop = s["fam"][max(oth, key=lambda i: top1[lab[i]])]
        fam_best = {f: max(top1[lab[i]] for i in oth if s["fam"][i] == f) for f in set(s["fam"][i] for i in oth)}
        rank_of_oracle = 1 + sum(v > fam_best[ofam] for v in fam_best.values())
        out.append({"name": s["name"], "stretch": s["stretch"], "cas": s["cas"], "ndet": N, "c_ref2": float(c[ref] ** 2), "E": s["E"],
                    "curves": cv, "top1_match": jtop == ofam, "oracle_rank_under_jev": int(rank_of_oracle), "n_families": len(fam_best),
                    "jev_top_det": max(oth, key=lambda i: top1[lab[i]]), "jev_top_label": lab[max(oth, key=lambda i: top1[lab[i]])],
                    "jev_top_h0": float(h0[max(oth, key=lambda i: top1[lab[i]])]),
                    "rho_top1_c2": float(st.spearmanr([top1[lab[i]] for i in oth], [c[i] ** 2 for i in oth]).statistic),
                    "rho_noul_c2": float(st.spearmanr([nl[lab[i]] for i in oth], [c[i] ** 2 for i in oth]).statistic),
                    "labels": lab, "top1": [top1[l] for l in lab], "noul": [nl[l] for l in lab], "jev_raw": J[si * R:(si + 1) * R]})
        o = out[-1]
        print(f"{s['name']:4s} x{s['stretch']} {s['cas']} N={N} c_ref2={o['c_ref2']:.2f} top1_match={o['top1_match']} oracle_rank={o['oracle_rank_under_jev']}/{o['n_families']} "
              f"jev_top={o['jev_top_label']} (H_i0={o['jev_top_h0']:.1e}) rho_top1={o['rho_top1_c2']:+.2f} rho_noul={o['rho_noul_c2']:+.2f}", flush=True)
    json.dump(out, open("experiments/e4v2_raw.json", "w"))
