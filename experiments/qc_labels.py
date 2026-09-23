"""Ground-truth labels for E1/E3 with PySCF (cc-pVDZ, closed-shell references)."""
import json, warnings
import numpy as np
from pyscf import gto, scf, cc, mcscf, lib
warnings.filterwarnings("ignore"); lib.num_threads(8)

# name -> (geometry template taking stretch factor f, equilibrium description)
def diat(a, b, r):
    return lambda f: f"{a} 0 0 0; {b} 0 0 {r*f}"
def water(f):
    r, t = 0.958 * f, np.deg2rad(104.5) / 2
    return f"O 0 0 0; H {r*np.sin(t)} 0 {r*np.cos(t)}; H {-r*np.sin(t)} 0 {r*np.cos(t)}"
def ammonia(f):
    r = 1.012 * f; th = np.deg2rad(106.7)
    z = -r*np.sqrt(1 - (2/3)*(1-np.cos(th)) * 2 / 2 * 1.0) if False else None
    # C3v: H atoms at polar angle a from the C3 axis, with a chosen so that angle HNH = th
    a = np.arcsin(np.sqrt((2/3)*(1-np.cos(th))))
    hs = [f"H {r*np.sin(a)*np.cos(p)} {r*np.sin(a)*np.sin(p)} {-r*np.cos(a)}" for p in (0, 2*np.pi/3, 4*np.pi/3)]
    return "N 0 0 0; " + "; ".join(hs)
def methane(f):
    d = 1.087 * f / np.sqrt(3)
    return "C 0 0 0; " + "; ".join(f"H {x*d} {y*d} {z*d}" for x, y, z in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)])
def h4(f):  # square H4, side 1.0*f angstrom... f=1 is rectangle near degeneracy
    return f"H 0 0 0; H 1.0 0 0; H 0 {f} 0; H 1.0 {f} 0"

MOLS = {"H2": diat("H","H",0.741), "LiH": diat("Li","H",1.595), "HF": diat("H","F",0.917),
        "N2": diat("N","N",1.098), "F2": diat("F","F",1.412), "CO": diat("C","O",1.128),
        "BH": diat("B","H",1.232), "BeO": diat("Be","O",1.331), "C2": diat("C","C",1.243),
        "H2O": water, "NH3": ammonia, "CH4": methane}
STRETCH = [1.0, 1.5, 2.0, 2.5]

def case(name, f, geom):
    mol = gto.M(atom=geom, basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    # RHF -> UHF instability: does a broken-symmetry UHF go lower?
    umf = scf.UHF(mol); dm = umf.get_init_guess(); dm[0][0, 0] += 0.3; dm[1][0, 0] -= 0.3
    umf.kernel(dm0=dm); mo = umf.stability()[0]
    umf.kernel(dm0=umf.make_rdm1(mo, umf.mo_occ))
    uhf_drop = mf.e_tot - umf.e_tot
    # CASCI(6e,6o) (or all electrons if fewer) on RHF orbitals: HF determinant weight
    ne = min(6, mol.nelectron); no = 6
    mc = mcscf.CASCI(mf, no, ne).run()
    c0sq = float(np.max(np.abs(mc.ci))**2)
    # CCSD T1 diagnostic
    try:
        mcc = cc.CCSD(mf).run(max_cycle=200)
        t1 = float(np.linalg.norm(mcc.t1) / np.sqrt(mol.nelectron)); cc_ok = bool(mcc.converged)
    except Exception:
        t1, cc_ok = None, False
    homo = mf.mo_energy[mol.nelectron//2 - 1]; lumo = mf.mo_energy[mol.nelectron//2]
    return {"name": name, "stretch": f, "geometry": geom, "e_rhf": mf.e_tot,
            "gap_eh": float(lumo - homo), "uhf_drop_eh": float(uhf_drop),
            "rhf_unstable": bool(uhf_drop > 1e-4), "casci_c0sq": c0sq, "mr": bool(c0sq < 0.90),
            "t1": t1, "t1_flag": (t1 is not None and t1 > 0.02), "ccsd_converged": cc_ok}

rows = []
for n, g in MOLS.items():
    for f in STRETCH:
        r = case(n, f, g(f)); rows.append(r)
        print(f"{n:4s} x{f:.1f}  gap={r['gap_eh']:.3f} c0^2={r['casci_c0sq']:.3f} t1={r['t1'] if r['t1'] is None else round(r['t1'],3)} uhfdrop={r['uhf_drop_eh']*1000:.1f}mEh cc_conv={r['ccsd_converged']}", flush=True)
json.dump(rows, open("experiments/qc_labels.json", "w"), indent=1)
