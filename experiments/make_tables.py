"""Write LaTeX tables straight from analysis outputs, so no number is transcribed by hand."""
import json
e3 = json.load(open("experiments/e3v2_results.json"))
rows = [("base_rate", "Base rate"), ("stretch", "Stretch factor (stated in prompt)"), ("logistic", "Logistic (stretch, HOMO--LUMO gap)"),
        ("jev", r"\jev{}"), ("jev_platt", r"\jev{}, Platt-recalibrated"), ("logistic+jev", r"Logistic + \jev{}"),
        ("jev_no_name", r"\jev{}, name removed"), ("jev_no_stretch_note", r"\jev{}, stretch note removed"), ("jev_geometry_only", r"\jev{}, geometry only")]
f = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r"& \multicolumn{2}{c}{Multireference} & \multicolumn{2}{c}{$T_1>0.02$} & \multicolumn{2}{c}{RHF unstable} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"Predictor & AUROC & Brier & AUROC & Brier & AUROC & Brier \\", r"\midrule"]
for k, name in rows:
    cells = []
    for lab in ["mr", "t1", "unstable"]:
        v = e3[lab][k]; cells += ["--" if k == "base_rate" else f(v["auroc"]), f(v["brier"], 3)]
    L.append(name + " & " + " & ".join(cells) + r" \\")
    if k in ("logistic", "logistic+jev"): L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_e3.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
