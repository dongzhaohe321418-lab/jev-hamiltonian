"""E6 ground truth: CCSD(T) error against full CI in small bases where FCI is feasible.

Cases: the 12 E3 molecules (geometries from qc_labels.MOLS) and 12 E5 molecules (e5_labels.MOLS) in STO-3G,
plus H2, LiH, BH, HF, H2O in 6-31G; stretches 1.0/1.5/2.0/2.5 (same scaling convention as E3/E5).
All electrons correlated in both CCSD(T) and FCI (no frozen core), so the FCI is exact in the stated basis.
FCI: see robust_fci() -- two Davidson runs (with / without spin penalty, several roots), lowest singlet root;
cases where the runs disagree, did not converge, or E_FCI > E_RHF are flagged (fci_converged False) and excluded. Label ccsdt_ok = CCSD converged and |E_CCSD(T) - E_FCI| < 1.6 mEh.
RHF reference (protocol v2, coordinator instruction): qc_labels_v2.stable_rhf -- RHF followed to an internally
stable minimum; cases with a non-converged RHF are flagged and excluded. T1 (all-electron CCSD here, since the
same CCSD feeds CCSD(T)) and the HOMO-LUMO gap come from this RHF. Wall-clock times recorded.
"""
import json, sys, time, warnings
import numpy as np
from pyscf import gto, scf, cc, fci, lib
warnings.filterwarnings("ignore")
sys.path.insert(0, "experiments")
from qc_labels import MOLS as M3
from e5_labels import MOLS as M5
from qc_labels_v2 import stable_rhf

STRETCH = [1.0, 1.5, 2.0, 2.5]
SETS = ([(n, "sto-3g", M3[n]) for n in M3] +
        [(n, "sto-3g", M5[n]) for n in ["HCN", "C2H2", "CH2O", "N2H2", "O3", "LiF", "MgO", "Be2", "SiO", "P2", "BF", "HCl"]] +
        [(n, "6-31g", M3[n]) for n in ["H2", "LiH", "BH", "HF", "H2O"]])

def robust_fci(mol, mf):
    """Singlet ground-state FCI, two independent Davidson runs (a first version with one fix_spin_ run and the
    default settings returned unconverged / non-singlet roots for several stretched cases, e.g. N2H2/STO-3G x2.0
    was 1.4 Eh too high):  A = spin penalty (shift 0.2) with 3 roots;  B = no penalty with 4 roots.
    E_FCI = lowest root with <S^2> < 0.01 over both runs; the case is flagged unless run A converged to a singlet,
    run B found no singlet or the same one (1e-5 Eh), and E_FCI <= E_RHF."""
    from pyscf import ao2mo
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff; eri = ao2mo.kernel(mol, mf.mo_coeff)
    t0 = time.time(); runs = {}
    for tag, pen, nr in (("A", 0.2, 3), ("B", None, 4)):
        sol = fci.direct_spin1.FCI(mol)
        if pen is not None: sol = fci.addons.fix_spin_(sol, ss=0, shift=pen)
        sol.nroots = nr; sol.max_cycle = 600; sol.max_space = 30; sol.conv_tol = 1e-10
        es, vs = sol.kernel(h1, eri, mol.nao, mol.nelectron, ecore=mol.energy_nuc())
        ss = [float(fci.spin_op.spin_square0(v, mol.nao, mol.nelectron)[0]) for v in vs]
        sing = [e for e, x in zip(es, ss) if x < 0.01]
        runs[tag] = {"e_singlet": float(min(sing)) if sing else None, "roots": [float(e) for e in es], "s2": ss,
                     "converged": bool(np.all(sol.converged))}
    cands = [r["e_singlet"] for r in runs.values() if r["e_singlet"] is not None]
    e = min(cands) if cands else float("nan")
    a, b = runs["A"]["e_singlet"], runs["B"]["e_singlet"]
    agree = (a is not None and b is not None and abs(a - b) < 1e-5)
    # ok: run A converged to a singlet, run B found either no singlet among its 4 roots or the same singlet energy,
    # and the result is variationally sane (E_FCI <= E_RHF)
    ok = bool(a is not None and runs["A"]["converged"] and (b is None or agree) and e <= mf.e_tot + 1e-8)
    return e, 0.0 if cands else float("nan"), {"runs": runs, "agree": agree, "ok": ok}, time.time() - t0

def case(name, basis, f, geom):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = stable_rhf(mol); conv0 = bool(mf.converged)
    t0 = time.time()
    mcc = cc.CCSD(mf); mcc.max_cycle = 200; mcc.kernel()
    et = mcc.ccsd_t(); t_cc = time.time() - t0
    t1 = float(np.linalg.norm(mcc.t1) / np.sqrt(mol.nelectron))
    e_fci, ss, fci_info, t_fci = robust_fci(mol, mf)
    e_cc = float(mcc.e_tot + et)
    homo = mf.mo_energy[mol.nelectron//2 - 1]; lumo = mf.mo_energy[mol.nelectron//2]
    err = e_cc - float(e_fci)
    return {"name": name, "basis": basis, "stretch": f, "geometry": geom, "nao": int(mol.nao), "nelectron": int(mol.nelectron),
            "e_rhf": float(mf.e_tot), "e_ccsd": float(mcc.e_tot), "e_ccsdt": e_cc, "e_fci": float(e_fci),
            "err_eh": err, "abs_err_eh": abs(err), "fci_s2": ss, "fci_converged": fci_info["ok"], "fci_info": fci_info,
            "ccsd_converged": bool(mcc.converged), "rhf_converged": bool(mf.converged),
            "t1": t1, "t1_flag": t1 > 0.02, "gap_eh": float(lumo - homo),
            "ccsdt_ok": bool(mcc.converged and abs(err) < 1.6e-3), "exclude": not (mf.converged and mcc.converged), "t_ccsdt_s": t_cc, "t_fci_s": t_fci}

def _run(args):
    lib.num_threads(2)
    n, b, f = args
    g = dict((k, gg) for k, _, gg in SETS if k == n)[n]
    return case(n, b, f, g(f))

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    out = "experiments/e6_labels.json"
    jobs = [(n, b, f) for n, b, _ in SETS for f in STRETCH]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as ex:
        rows = []
        for r in ex.map(_run, jobs):
            rows.append(r)
            print(f"{r['name']:5s} {r['basis']:6s} x{r['stretch']:.1f} nao={r['nao']} err={r['err_eh']*1000:+.3f} mEh ok={r['ccsdt_ok']} t1={r['t1']:.3f} "
                  f"cc={r['ccsd_converged']} rhf={r['rhf_converged']} fci_ok={r['fci_converged']} "
                  f"t_cc={r['t_ccsdt_s']:.1f}s t_fci={r['t_fci_s']:.1f}s", flush=True)
    json.dump(rows, open(out, "w"), indent=1)
