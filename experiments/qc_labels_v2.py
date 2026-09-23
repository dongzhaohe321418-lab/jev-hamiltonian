"""Ground-truth labels, version 2 (after review round 1, R2-M1). Differences from qc_labels.py:
(i)  RHF is followed to an internally stable minimum; non-converged SCF is flagged.
(ii) Multireference character from a symmetry-complete full-valence CASCI on RHF orbitals with a singlet constraint;
     the label uses the weight of the RHF determinant itself (c_HF^2 < 0.9); the largest weight is also stored.
(iii) CCSD with frozen core; T1 = ||t1|| / sqrt(N_correlated); non-converged CCSD gives no T1 label.
(iv) UHF: several starts (default guess, HOMO/LUMO mixing, RHF orbitals) each followed by iterated internal+external
     stability analysis; the lowest solution is kept. Primary threshold 1 mEh; 0.1 and 10 mEh stored for sensitivity.
Use `label(geometry, basis)` from other scripts."""
import json, warnings
import numpy as np
from pyscf import gto, scf, cc, mcscf, lib
warnings.filterwarnings("ignore")

VAL = {"H": (1, 1), "He": (2, 1), "Li": (1, 5), "Be": (2, 5), "B": (3, 4), "C": (4, 4), "N": (5, 4), "O": (6, 4), "F": (7, 4),
       "Ne": (8, 4), "Na": (1, 5), "Mg": (2, 5), "Al": (3, 4), "Si": (4, 4), "P": (5, 4), "S": (6, 4), "Cl": (7, 4)}
# (valence electrons, valence orbitals); Li/Be/Na/Mg use ns + np (5 with the empty p shell counted as 3 + s + 1 spare -> use 4)
VAL.update({"Li": (1, 4), "Be": (2, 4), "Na": (1, 4), "Mg": (2, 4)})

def stable_rhf(mol, maxit=10):
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    for _ in range(maxit):
        mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
        if stable: break
        mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
    return mf

def lowest_uhf(mol, rhf):
    best = None
    na, nb = mol.nelec
    starts = []
    umf = scf.UHF(mol); starts.append(umf.get_init_guess())
    for ang in (np.pi / 4, np.pi / 8):            # HOMO/LUMO mixing guesses from RHF orbitals
        C = rhf.mo_coeff; h, l = na - 1, na
        ca, cb = C.copy(), C.copy()
        ca[:, h], ca[:, l] = np.cos(ang) * C[:, h] + np.sin(ang) * C[:, l], -np.sin(ang) * C[:, h] + np.cos(ang) * C[:, l]
        cb[:, h], cb[:, l] = np.cos(ang) * C[:, h] - np.sin(ang) * C[:, l], np.sin(ang) * C[:, h] + np.cos(ang) * C[:, l]
        occ = np.zeros(C.shape[1]); occ[:na] = 1
        starts.append((ca[:, occ > 0] @ ca[:, occ > 0].T, cb[:, occ > 0] @ cb[:, occ > 0].T))
    for dm in starts:
        u = scf.UHF(mol); u.conv_tol = 1e-10; u.max_cycle = 200; u.kernel(dm0=np.array(dm))
        for _ in range(10):
            mo, _, stable, _ = u.stability(internal=True, external=False, return_status=True)
            if stable: break
            u.kernel(dm0=u.make_rdm1(mo, u.mo_occ))
        if u.converged and (best is None or u.e_tot < best.e_tot): best = u
    return best

def full_valence(mol):
    ne = sum(VAL[mol.atom_symbol(i)][0] for i in range(mol.natm)) - mol.charge
    no = sum(VAL[mol.atom_symbol(i)][1] for i in range(mol.natm))
    return ne, no

def label(geometry, basis="cc-pvdz", charge=0):
    lib.num_threads(4)
    mol = gto.M(atom=geometry, basis=basis, charge=charge, verbose=0)
    mf = stable_rhf(mol)
    out = {"rhf_converged": bool(mf.converged), "e_rhf": float(mf.e_tot)}
    nocc = mol.nelectron // 2
    out["gap_eh"] = float(mf.mo_energy[nocc] - mf.mo_energy[nocc - 1])
    # (iv) UHF
    u = lowest_uhf(mol, mf)
    drop = float(mf.e_tot - u.e_tot) if u is not None else float("nan")
    out.update(uhf_drop_eh=drop, rhf_unstable=bool(drop > 1e-3), rhf_unstable_0p1=bool(drop > 1e-4), rhf_unstable_10=bool(drop > 1e-2))
    # (ii) full-valence singlet CASCI
    ne, no = full_valence(mol)
    ncore = (mol.nelectron - ne) // 2
    e = mf.mo_energy; lo, hi = ncore, ncore + no
    split = (lo > 0 and abs(e[lo] - e[lo - 1]) < 1e-4) or (hi < len(e) and abs(e[hi] - e[hi - 1]) < 1e-4)
    mc = mcscf.CASCI(mf, no, ne); mc.fix_spin_(ss=0); mc.kernel()
    ci = np.asarray(mc.ci)
    out.update(cas=f"({ne}e,{no}o)", cas_split_degenerate=bool(split), c_hf2=float(ci[0, 0] ** 2), c_max2=float(np.max(ci ** 2)),
               mr=bool(ci[0, 0] ** 2 < 0.90), cas_ss=float(mc.fcisolver.spin_square(mc.ci, no, ne)[0]))
    # (iii) frozen-core CCSD
    nfrz = sum(0 if mol.atom_symbol(i) in ("H", "He", "Li", "Be") else (1 if mol.atom_charge(i) <= 10 else 5) for i in range(mol.natm))
    try:
        mcc = cc.CCSD(mf, frozen=nfrz or None); mcc.max_cycle = 300; mcc.kernel()
        ok = bool(mcc.converged); t1 = float(np.linalg.norm(mcc.t1) / np.sqrt(mol.nelectron - 2 * nfrz))
    except Exception:
        ok, t1 = False, None
    out.update(ccsd_converged=ok, t1=t1 if ok else None, t1_flag=(t1 > 0.02) if ok else None, frozen_core=nfrz)
    return out

if __name__ == "__main__":
    rows = json.load(open("experiments/qc_labels.json"))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(3) as ex:
        new = list(ex.map(label, [r["geometry"] for r in rows]))
    out = []
    for r, n in zip(rows, new):
        out.append({"name": r["name"], "stretch": r["stretch"], "geometry": r["geometry"], **n})
        print(f"{r['name']:4s} x{r['stretch']}  rhfconv={n['rhf_converged']} gap={n['gap_eh']:.3f} {n['cas']} split={n['cas_split_degenerate']} "
              f"cHF2={n['c_hf2']:.3f}(v1 {r['casci_c0sq']:.3f}) S2={n['cas_ss']:.2f} t1={n['t1']} conv={n['ccsd_converged']} uhfdrop={1000*n['uhf_drop_eh']:.1f}mEh(v1 {1000*r['uhf_drop_eh']:.1f})", flush=True)
    json.dump(out, open("experiments/qc_labels_v2.json", "w"), indent=1)
