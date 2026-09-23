"""E5 ground-truth labels: 31 less-textbook closed-shell molecules x 4 stretches, cc-pVDZ.

The label procedure is the one in qc_labels.case (same RHF, same UHF broken-symmetry guess + stability
step, same CASCI(6e,6o) on RHF orbitals, same CCSD T1 definition, same thresholds), with extra
convergence bookkeeping added:
  * rhf_converged: RHF converged with the default solver; if not, a second-order (Newton) retry is made
    and rhf_retry=True. Cases whose RHF still does not converge are flagged (exclude_mr/exclude_unstable).
  * uhf_converged, casci_converged, ccsd_converged recorded; T1 label excluded if CCSD did not converge.

Geometries: experimental equilibrium structures (r_e / r_0 and angles as tabulated in NIST CCCBDB,
experimental section, and Huber & Herzberg for diatomics), rounded to 3 decimals in Angstrom and 0.1 deg.
Where only approximate values were recalled, this is marked "approx." below. Stretch f scales every
Cartesian coordinate about the origin by f, so every interatomic distance is multiplied by f and all
angles are unchanged ("all bonds stretched to f x equilibrium", as in E3).
"""
import json, sys, time, warnings
import numpy as np
from pyscf import gto, scf, cc, mcscf, lib
warnings.filterwarnings("ignore")

def fmt(atoms):
    return "; ".join(f"{a} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in atoms)

def scaled(atoms):
    return lambda f: fmt([(a, (f * np.asarray(p)).tolist()) for a, p in atoms])

def diat(a, b, r):
    return scaled([(a, (0, 0, 0)), (b, (0, 0, r))])

def linear(names, bonds):  # consecutive bond lengths along z
    z = np.concatenate([[0], np.cumsum(bonds)])
    return scaled([(n, (0, 0, zz)) for n, zz in zip(names, z)])

def bent_xy2(x, y, r, ang):  # central x at origin
    t = np.deg2rad(ang) / 2
    return scaled([(x, (0, 0, 0)), (y, (r*np.sin(t), 0, r*np.cos(t))), (y, (-r*np.sin(t), 0, r*np.cos(t)))])

def bent_abc(a, b, c, rab, rbc, ang):  # central b at origin, angle a-b-c
    t = np.deg2rad(ang)
    return scaled([(b, (0, 0, 0)), (c, (0, 0, rbc)), (a, (rab*np.sin(t), 0, rab*np.cos(t)))])

def c3v(x, y, r, ang):  # XY3 pyramid with Y-X-Y angle ang (same construction as qc_labels.ammonia)
    th = np.deg2rad(ang); a = np.arcsin(np.sqrt((2/3)*(1-np.cos(th))))
    return scaled([(x, (0, 0, 0))] + [(y, (r*np.sin(a)*np.cos(p), r*np.sin(a)*np.sin(p), -r*np.cos(a)))
                                     for p in (0, 2*np.pi/3, 4*np.pi/3)])

def td(x, y, r):
    d = r / np.sqrt(3)
    return scaled([(x, (0, 0, 0))] + [(y, (s1*d, s2*d, s3*d)) for s1, s2, s3 in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]])

def formaldehyde(rco=1.205, rch=1.111, hch=116.1):
    t = np.deg2rad(hch)/2
    return scaled([("C", (0, 0, 0)), ("O", (0, 0, -rco)), ("H", (rch*np.sin(t), 0, rch*np.cos(t))), ("H", (-rch*np.sin(t), 0, rch*np.cos(t)))])

def ethylene(rcc=1.339, rch=1.086, hch=117.6):
    t = np.deg2rad(hch)/2; c = rcc/2
    return scaled([("C", (0, 0, c)), ("C", (0, 0, -c)),
                   ("H", (rch*np.sin(t), 0, c + rch*np.cos(t))), ("H", (-rch*np.sin(t), 0, c + rch*np.cos(t))),
                   ("H", (rch*np.sin(t), 0, -c - rch*np.cos(t))), ("H", (-rch*np.sin(t), 0, -c - rch*np.cos(t)))])

def diazene(rnn=1.252, rnh=1.028, nnh=106.9):  # trans-HN=NH, C2h
    t = np.deg2rad(nnh); c = rnn/2
    return scaled([("N", (0, 0, c)), ("N", (0, 0, -c)),
                   ("H", (rnh*np.sin(t), 0, c - rnh*np.cos(t))), ("H", (-rnh*np.sin(t), 0, -c + rnh*np.cos(t)))])

def peroxide(roo=1.475, roh=0.950, ooh=94.8, dih=111.5):  # H2O2, C2; dihedral approx. (gas-phase 111.5 deg)
    t, p = np.deg2rad(ooh), np.deg2rad(dih)
    return scaled([("O", (0, 0, 0)), ("O", (0, 0, roo)), ("H", (roh*np.sin(t), 0, roh*np.cos(t))),
                   ("H", (roh*np.sin(t)*np.cos(p), roh*np.sin(t)*np.sin(p), roo - roh*np.cos(t)))])

MOLS = {
    # diatomics, r_e in Angstrom (Huber-Herzberg / CCCBDB experimental)
    "LiF": diat("Li", "F", 1.564), "NaH": diat("Na", "H", 1.887), "MgO": diat("Mg", "O", 1.749),
    "SiO": diat("Si", "O", 1.510), "P2": diat("P", "P", 1.893), "Cl2": diat("Cl", "Cl", 1.988),
    "Be2": diat("Be", "Be", 2.454),  # Merritt et al. 2009 r_e = 2.4536 A (weakly bound)
    "BF": diat("B", "F", 1.263), "AlF": diat("Al", "F", 1.654), "HCl": diat("H", "Cl", 1.275),
    "NaF": diat("Na", "F", 1.926), "CS": diat("C", "S", 1.535), "AlH": diat("Al", "H", 1.648),
    "SiS": diat("Si", "S", 1.929),
    # polyatomics
    "HCN": linear(["H", "C", "N"], [1.064, 1.156]),
    "HNO": bent_abc("H", "N", "O", 1.063, 1.212, 108.6),
    "HOCl": bent_abc("H", "O", "Cl", 0.964, 1.689, 102.9),
    "FNO": bent_abc("F", "N", "O", 1.512, 1.136, 110.1),
    "CH2O": formaldehyde(),
    "C2H4": ethylene(),
    "C2H2": linear(["H", "C", "C", "H"], [1.063, 1.203, 1.063]),
    "N2H2": diazene(),  # trans-diazene
    "O3": bent_xy2("O", "O", 1.278, 116.8),
    "SiH4": td("Si", "H", 1.480),
    "PH3": c3v("P", "H", 1.420, 93.3),
    "H2S": bent_xy2("S", "H", 1.336, 92.1),
    "CO2": linear(["O", "C", "O"], [1.160, 1.160]),
    "N2O": linear(["N", "N", "O"], [1.128, 1.184]),
    "SO2": bent_xy2("S", "O", 1.431, 119.3),
    "H2O2": peroxide(),
    "HOF": bent_abc("H", "O", "F", 0.966, 1.442, 97.2),  # approx. (CCCBDB exp. OH 0.966, OF 1.442, 97.2 deg)
}
STRETCH = [1.0, 1.5, 2.0, 2.5]

def case(name, f, geom):
    mol = gto.M(atom=geom, basis="cc-pvdz", verbose=0)
    t0 = time.time()
    mf = scf.RHF(mol); mf.kernel()
    rhf_conv0 = bool(mf.converged); retry = False
    if not mf.converged:
        retry = True
        mf = scf.RHF(mol).newton(); mf.kernel(); mf = mf  # second-order retry
    # RHF -> UHF instability (identical procedure to qc_labels.case)
    umf = scf.UHF(mol); dm = umf.get_init_guess(); dm[0][0, 0] += 0.3; dm[1][0, 0] -= 0.3
    umf.kernel(dm0=dm); mo = umf.stability()[0]
    umf.kernel(dm0=umf.make_rdm1(mo, umf.mo_occ))
    uhf_drop = mf.e_tot - umf.e_tot
    ne = min(6, mol.nelectron); no = 6
    mc = mcscf.CASCI(mf, no, ne); mc.kernel()
    c0sq = float(np.max(np.abs(mc.ci))**2)
    try:
        mcc = cc.CCSD(mf); mcc.max_cycle = 200; mcc.kernel()
        t1 = float(np.linalg.norm(mcc.t1) / np.sqrt(mol.nelectron)); cc_ok = bool(mcc.converged)
    except Exception as e:
        t1, cc_ok = None, False
    homo = mf.mo_energy[mol.nelectron//2 - 1]; lumo = mf.mo_energy[mol.nelectron//2]
    rhf_ok = bool(mf.converged)
    return {"name": name, "stretch": f, "geometry": geom, "nao": int(mol.nao), "nelectron": int(mol.nelectron),
            "e_rhf": float(mf.e_tot), "gap_eh": float(lumo - homo), "uhf_drop_eh": float(uhf_drop),
            "rhf_unstable": bool(uhf_drop > 1e-4), "casci_c0sq": c0sq, "mr": bool(c0sq < 0.90),
            "t1": t1, "t1_flag": (t1 is not None and t1 > 0.02), "ccsd_converged": cc_ok,
            "rhf_converged_default": rhf_conv0, "rhf_newton_retry": retry, "rhf_converged": rhf_ok,
            "uhf_converged": bool(umf.converged), "casci_converged": bool(getattr(mc, "converged", True)),
            "exclude_mr": not rhf_ok, "exclude_unstable": (not rhf_ok) or (not umf.converged),
            "exclude_t1": (not rhf_ok) or (not cc_ok) or t1 is None, "seconds": time.time() - t0}

if __name__ == "__main__":
    lib.num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    out = "experiments/e5_labels.json"
    try:
        rows = json.load(open(out))
    except FileNotFoundError:
        rows = []
    done = {(r["name"], r["stretch"]) for r in rows}
    for n, g in MOLS.items():
        for f in STRETCH:
            if (n, f) in done:
                continue
            r = case(n, f, g(f)); rows.append(r)
            print(f"{n:5s} x{f:.1f} nao={r['nao']} gap={r['gap_eh']:.3f} c0^2={r['casci_c0sq']:.3f} "
                  f"t1={r['t1'] if r['t1'] is None else round(r['t1'],3)} uhfdrop={r['uhf_drop_eh']*1000:.1f}mEh "
                  f"rhf={r['rhf_converged_default']}/{r['rhf_converged']} uhf={r['uhf_converged']} cc={r['ccsd_converged']} {r['seconds']:.0f}s", flush=True)
            json.dump(rows, open(out, "w"), indent=1)
