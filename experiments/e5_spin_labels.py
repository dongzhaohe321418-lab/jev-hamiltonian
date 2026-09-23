"""E5 spin-state task: ground-truth lowest multiplicity (singlet / triplet / quintet) for 32 small species.

Ground truth: frozen-core CCSD(T)/cc-pVDZ, ADIABATIC comparison: each multiplicity is optimised separately
within a fixed point-group parameterisation (diatomic: r; C2v XY2: r, angle; Cs HCX: r_CH, r_CX, angle) by
Nelder-Mead / bounded scalar minimisation on the CCSD(T) energy itself (no analytic gradients).
Because single-reference CCSD(T) depends on which SCF solution it starts from (a first version that used one
stability-followed SCF solution gave C2 singlet 60 mEh too high and a spurious (T)-collapse for BN singlet),
every energy evaluation builds several SCF solutions and keeps the lowest *sane* CCSD(T) (see calc()).
  * singlet: closed-shell RHF (several guesses, with/without internal-stability following) -> RCCSD(T). Open-shell singlets (e.g. O2 1Delta_g,
    NH 1Delta) are therefore described by one closed-shell component only, which biases the singlet UP.
  * triplet / quintet: UHF (internal stability followed) -> UCCSD(T); <S^2> of the UHF reference recorded.
    At the optimised geometry we also recompute with an ROHF reference (ROHF-UCCSD(T)) as a check.
  * quintet is optimised only if its energy at the lowest singlet/triplet geometry is within 0.10 Eh
    (2.7 eV) of that state; otherwise its vertical energy there is recorded and it is marked 'not competitive'.
  * we also record the VERTICAL ordering of all multiplicities at the ground-state geometry, and the UHF-level
    (HF-only) adiabatic ordering, used as a cheap-computation baseline.
The experimental/accepted ground-state multiplicity (standard spectroscopic assignments: Huber & Herzberg,
NIST CCCBDB, and the carbene/silylene literature) is recorded in EXPT as a sanity check; it is NOT the label.
"""
import json, sys, warnings
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize, minimize_scalar
warnings.filterwarnings("ignore")

# name: (charge, kind, atoms, initial params, expt multiplicity)
SPECIES = {
    "C": (0, "atom", ["C"], [], 3), "O": (0, "atom", ["O"], [], 3), "Si": (0, "atom", ["Si"], [], 3),
    "Be": (0, "atom", ["Be"], [], 1),
    "NH": (0, "diat", ["N", "H"], [1.036], 3), "O2": (0, "diat", ["O", "O"], [1.208], 3),
    "NF": (0, "diat", ["N", "F"], [1.317], 3), "BH": (0, "diat", ["B", "H"], [1.232], 1),
    "AlH": (0, "diat", ["Al", "H"], [1.648], 1), "C2": (0, "diat", ["C", "C"], [1.243], 1),
    "CN-": (-1, "diat", ["C", "N"], [1.177], 1), "S2": (0, "diat", ["S", "S"], [1.889], 3),
    "SO": (0, "diat", ["S", "O"], [1.481], 3), "PH": (0, "diat", ["P", "H"], [1.422], 3),
    "PF": (0, "diat", ["P", "F"], [1.590], 3), "NCl": (0, "diat", ["N", "Cl"], [1.611], 3),
    "B2": (0, "diat", ["B", "B"], [1.590], 3), "SiC": (0, "diat", ["Si", "C"], [1.722], 3),
    "BN": (0, "diat", ["B", "N"], [1.281], 3), "NO-": (-1, "diat", ["N", "O"], [1.258], 3),
    "BeO": (0, "diat", ["Be", "O"], [1.331], 1), "CO": (0, "diat", ["C", "O"], [1.128], 1),
    "CH2": (0, "xy2", ["C", "H"], [1.09, 115.0], 3), "SiH2": (0, "xy2", ["Si", "H"], [1.51, 100.0], 1),
    "CF2": (0, "xy2", ["C", "F"], [1.30, 105.0], 1), "CCl2": (0, "xy2", ["C", "Cl"], [1.72, 110.0], 1),
    "NH2+": (1, "xy2", ["N", "H"], [1.04, 130.0], 3), "PH2+": (1, "xy2", ["P", "H"], [1.42, 100.0], 1),
    "NH2-": (-1, "xy2", ["N", "H"], [1.03, 104.0], 1), "SiF2": (0, "xy2", ["Si", "F"], [1.59, 101.0], 1),
    "CHF": (0, "hcx", ["C", "H", "F"], [1.12, 1.31, 103.0], 1), "CHCl": (0, "hcx", ["C", "H", "Cl"], [1.10, 1.69, 105.0], 1),
}
MULTS = (1, 3, 5)

def geom(kind, atoms, p):
    if kind == "atom":
        return f"{atoms[0]} 0 0 0"
    if kind == "diat":
        return f"{atoms[0]} 0 0 0; {atoms[1]} 0 0 {p[0]:.6f}"
    if kind == "xy2":
        r, t = p[0], np.deg2rad(p[1]) / 2
        return f"{atoms[0]} 0 0 0; {atoms[1]} {r*np.sin(t):.6f} 0 {r*np.cos(t):.6f}; {atoms[1]} {-r*np.sin(t):.6f} 0 {r*np.cos(t):.6f}"
    if kind == "hcx":
        rh, rx, a = p[0], p[1], np.deg2rad(p[2])
        return f"{atoms[0]} 0 0 0; {atoms[2]} 0 0 {rx:.6f}; {atoms[1]} {rh*np.sin(a):.6f} 0 {rh*np.cos(a):.6f}"

def sanitize(kind, p):
    p = list(p)
    if kind in ("xy2", "hcx"):
        a = p[-1] % 360
        p[-1] = 360 - a if a > 180 else a
    return p

def _scf_solutions(mol, mult, ref):
    """Several SCF solutions: guesses minao/atom/1e, each with and without following internal instabilities;
    for open shells also HOMO->LUMO occupation swaps of the lowest solution. Deduplicated by energy (1e-5 Eh)."""
    from pyscf import scf
    make = (lambda: scf.RHF(mol)) if mult == 1 else ((lambda: scf.UHF(mol)) if ref == "uhf" else (lambda: scf.ROHF(mol)))
    sols = []
    def add(mf):
        if mf.converged and all(abs(mf.e_tot - o.e_tot) > 1e-5 for o in sols):
            sols.append(mf)
    for g in ("minao", "atom", "1e"):
        mf = make(); mf.init_guess = g; mf.max_cycle = 150; mf.kernel()
        if not mf.converged:
            mf = make().newton(); mf.init_guess = g; mf.kernel()
        add(mf)
        if ref == "uhf" or mult == 1:
            m2 = mf
            for _ in range(3):
                mo, _, stable, _ = m2.stability(return_status=True)
                if stable: break
                m3 = make(); m3.max_cycle = 150; m3.kernel(dm0=m2.make_rdm1(mo, m2.mo_occ)); m2 = m3
            add(m2)
    if mult > 1 and ref == "uhf" and sols:
        base = min(sols, key=lambda m: m.e_tot)
        for spin in (0, 1):
            occ = base.mo_occ[spin]; ho = int(np.sum(occ > 0)) - 1
            for k in (0, 1):
                for l in (0, 1):
                    if ho - k < 0 or ho + 1 + l >= len(occ): continue
                    o2 = [base.mo_occ[0].copy(), base.mo_occ[1].copy()]
                    o2[spin][ho - k] = 0; o2[spin][ho + 1 + l] = 1
                    mf = make(); mf.max_cycle = 150; mf.kernel(dm0=base.make_rdm1(base.mo_coeff, o2)); add(mf)
    return sorted(sols, key=lambda m: m.e_tot)

def calc(name, mult, p, ref="uhf", n_cc=3):
    """Frozen-core CCSD(T) energy of the lowest state of multiplicity `mult` at geometry p.
    CCSD(T) is run on up to n_cc lowest distinct SCF solutions; a solution is 'sane' if CCSD converged,
    T1 diagnostic < 0.08 and |E(T)| < 15% of the CCSD correlation energy. The reported energy is the lowest
    sane CCSD(T) energy (lowest overall, flagged, if none is sane)."""
    from pyscf import gto, cc
    from pyscf.data import elements
    q, kind, atoms, _, _ = SPECIES[name]
    mol = gto.M(atom=geom(kind, atoms, p), basis="cc-pvdz", charge=q, spin=mult - 1, verbose=0)
    if mult - 1 > mol.nelectron or (mol.nelectron - (mult - 1)) % 2:
        return None
    ncore = elements.chemcore(mol); nb = (mol.nelectron - (mult - 1)) // 2
    if nb < ncore:  # frozen core would freeze an empty beta orbital: not computable in this scheme
        return {"e_cc": None, "invalid": "n_beta < n_frozen_core"}
    sols = _scf_solutions(mol, mult, ref)
    if not sols:
        return {"e_cc": None, "invalid": "no converged SCF"}
    cands = []
    for mf in sols[:n_cc]:
        ss = float(mf.spin_square()[0]) if mult > 1 else 0.0
        try:
            c = cc.CCSD(mf, frozen=ncore) if ncore else cc.CCSD(mf)
            c.max_cycle = 150; c.kernel(); et = c.ccsd_t()
            t1n = np.sqrt(sum(np.linalg.norm(x)**2 for x in c.t1)) if isinstance(c.t1, tuple) else np.linalg.norm(c.t1) * np.sqrt(2)
            t1 = t1n / np.sqrt(2 * (mol.nelectron - 2 * ncore))  # = ||t1_closed||/sqrt(N_corr) for RHF; spin-orbital analogue for UHF
            sane = bool(c.converged and t1 < 0.08 and abs(et) < 0.15 * abs(c.e_corr))
            cands.append({"e_hf": float(mf.e_tot), "e_ccsd": float(c.e_tot), "e_t": float(et), "e_cc": float(c.e_tot + et),
                          "cc_converged": bool(c.converged), "t1diag": float(t1), "s2_ref": ss, "sane": sane})
        except Exception as e:
            cands.append({"e_hf": float(mf.e_tot), "e_cc": None, "sane": False, "error": str(e)[:120], "s2_ref": ss})
    ok = [c for c in cands if c["sane"] and c["e_cc"] is not None] or [c for c in cands if c["e_cc"] is not None]
    if not ok:
        return {"e_cc": None, "invalid": "CCSD failed", "candidates": cands}
    best = min(ok, key=lambda c: c["e_cc"])
    return {**best, "ncore": int(ncore), "n_scf_solutions": len(sols), "all_sane": bool(best["sane"]),
            "cc_spread_eh": float(max(c["e_cc"] for c in ok) - min(c["e_cc"] for c in ok)), "candidates": cands}

def optimise(name, mult, p0):
    q, kind, atoms, _, _ = SPECIES[name]
    nev = [0]; last = {}
    def f(p):
        nev[0] += 1
        p = sanitize(kind, np.atleast_1d(p))
        r = calc(name, mult, p, n_cc=2)
        e = r["e_cc"] if (r and r.get("e_cc") is not None) else 0.0
        last[tuple(np.round(p, 6))] = r
        return e
    if kind == "atom":
        r = calc(name, mult, []); return [], r, 1
    if kind == "diat":
        res = minimize_scalar(f, bounds=(p0[0] - 0.35, p0[0] + 0.5), method="bounded", options={"xatol": 2e-3})
        p = [float(res.x)]
    else:
        step = {"xy2": [0.05, 12.0], "hcx": [0.05, 0.05, 12.0]}[kind]
        sim = [np.array(p0)] + [np.array(p0) + np.eye(len(p0))[i] * step[i] for i in range(len(p0))]
        res = minimize(f, p0, method="Nelder-Mead", options={"initial_simplex": sim, "xatol": 2e-3, "fatol": 2e-6, "maxfev": 200})
        p = sanitize(kind, res.x)
    r = calc(name, mult, p)
    return [float(x) for x in p], r, nev[0]

def run_species(name):
    from pyscf import lib
    lib.num_threads(2)
    q, kind, atoms, p0, expt = SPECIES[name]
    out = {"name": name, "charge": q, "kind": kind, "expt_mult": expt, "states": {}}
    for m in (1, 3):
        p, r, nev = optimise(name, m, p0)
        out["states"][str(m)] = {"params": p, "n_evals": nev, **(r or {})}
    lo = min((1, 3), key=lambda m: out["states"][str(m)]["e_cc"] if out["states"][str(m)].get("e_cc") is not None else 1e9)
    pg = out["states"][str(lo)]["params"]
    v5 = calc(name, 5, pg)
    if v5 is not None and v5.get("e_cc") is not None and v5["e_cc"] - out["states"][str(lo)]["e_cc"] < 0.10:
        p, r, nev = optimise(name, 5, pg)
        out["states"]["5"] = {"params": p, "n_evals": nev, "optimised": True, **(r or {})}
    elif v5 is not None:
        out["states"]["5"] = {"params": pg, "n_evals": 1, "optimised": False, "not_competitive": True, **v5}
    # vertical energies of every multiplicity at the lowest (1,3) geometry; ROHF-based check at optimised geometries
    out["vertical_at_ground_geom"] = {str(m): (calc(name, m, pg) or {}).get("e_cc") for m in MULTS}
    out["rohf_check"] = {str(m): (calc(name, m, out["states"][str(m)]["params"], ref="rohf") or {}).get("e_cc")
                         for m in (3, 5) if str(m) in out["states"]}
    print(name, {m: (s.get("e_cc"), round(s.get("s2_ref", 0), 3)) for m, s in out["states"].items()}, flush=True)
    return out

if __name__ == "__main__":
    names = sys.argv[1:] or list(SPECIES)
    with Pool(6) as pool:
        res = pool.map(run_species, names, chunksize=1)
    json.dump(res, open("experiments/e5_spin_labels.json", "w"), indent=1)
