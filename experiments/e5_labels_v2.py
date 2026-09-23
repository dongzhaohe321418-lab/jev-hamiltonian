"""E5 closed-shell labels, protocol v2 (coordinator instruction after review of the original labels):
labels come from experiments/qc_labels_v2.label(geometry, 'cc-pvdz') -- internally stable RHF, lowest of
several stability-followed UHF starts (primary threshold 1 mEh; 0.1 and 10 mEh stored), full-valence singlet
CASCI on RHF orbitals (mr = c_HF^2 < 0.9; cas_split_degenerate reported separately), frozen-core CCSD T1
(None if unconverged). The E5 geometries/cases are those of e5_labels.py (e5_labels.json holds the v1-protocol
labels, kept only for comparison; the Jev prompts depend on name/geometry/stretch only).
All elements used (H, Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl) are in qc_labels_v2.VAL."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "experiments")
from qc_labels_v2 import label

if __name__ == "__main__":
    v1 = json.load(open("experiments/e5_labels.json"))
    with ProcessPoolExecutor(3) as ex:
        new = list(ex.map(label, [r["geometry"] for r in v1]))
    out = []
    for r, n in zip(v1, new):
        out.append({"name": r["name"], "stretch": r["stretch"], "geometry": r["geometry"], **n})
        print(f"{r['name']:5s} x{r['stretch']} rhfconv={n['rhf_converged']} gap={n['gap_eh']:.3f} {n['cas']} split={n['cas_split_degenerate']} "
              f"cHF2={n['c_hf2']:.3f}(v1 {r['casci_c0sq']:.3f}) S2={n['cas_ss']:.2f} t1={n['t1']} conv={n['ccsd_converged']} "
              f"uhfdrop={1000*n['uhf_drop_eh']:.1f}mEh(v1 {1000*r['uhf_drop_eh']:.1f})", flush=True)
    json.dump(out, open("experiments/e5_labels_v2.json", "w"), indent=1)
