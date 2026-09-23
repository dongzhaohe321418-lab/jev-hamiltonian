"""All data figures for the paper. Style: SciencePlots ['science','no-latex','bright']; vector PDF + 300 dpi PNG."""
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import scipy.stats as st
from sklearn.metrics import roc_curve

plt.style.use(["science", "no-latex", "bright"])
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "legend.frameon": False})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
OUT = "paper/figs/"
def panel(ax, s): ax.text(-0.2, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9)
def save(fig, name):
    fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=300); plt.close(fig)

# ---------- Fig 2: E1 landscape consistency ----------
r = json.load(open("experiments/e1e3_results.json"))
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
ax = axs[0]
Jab = np.concatenate([p[0] for p in r["J_pairs"]]); Jba = np.concatenate([p[1] for p in r["J_pairs"]])
lim = 3.8
q = np.percentile(r["null_abs"], 95); xx = np.array([-lim, lim])
ax.fill_between(xx, xx - q, xx + q, color="0.88", lw=0, label="coherent null (95%)")
ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=0.6)
ax.scatter(Jab, Jba, s=4, color=C[0], alpha=0.6, lw=0)
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"$J_{ab}$ (nats)", ylabel=r"$J_{ba}$ (nats)")
ax.legend(loc="upper left", handlelength=1); panel(ax, "a")
ax = axs[1]
bins = np.linspace(0, 3, 31)
ax.hist(np.abs(r["null_abs"]), bins=bins, density=True, color="0.7", label="coherent null")
ax.hist(np.abs(r["asym"]).ravel(), bins=bins, density=True, histtype="step", color=C[1], lw=1.2, label="Jev")
ax.set(xlabel=r"$|J_{ab}-J_{ba}|$ (nats)", ylabel="density", xlim=(0, 3)); ax.legend(); panel(ax, "b")
ax = axs[2]
res = np.abs(np.array(r["resid"])).ravel()
xs = np.sort(res); ax.plot(xs, np.arange(1, len(xs)+1)/len(xs), color=C[2])
ax.axvline(r["noise_sd"], color="0.5", ls="--", lw=0.7); ax.text(r["noise_sd"]*0.85, 0.55, "per-call\nnoise s.d.", fontsize=6, color="0.35", ha="right")
ax.set(xscale="log", xlim=(1e-3, 0.6), ylim=(0, 1), xlabel="|total-probability residual|", ylabel="cumulative fraction"); panel(ax, "c")
fig.tight_layout(w_pad=1.2); save(fig, "fig2_consistency")

# ---------- Fig 3: E2 in-context Hamiltonians ----------
e2 = json.load(open("experiments/e2_raw.json")); e2b = json.load(open("experiments/e2b_raw.json"))
def summ(rows):
    out = {}
    for n in range(3, 8):
        rs = [x for x in rows if x["n"] == n]
        rho = [st.spearmanr([-x["E"][k] for k in x["E"]], [x["p"].get(k, 0) for k in x["E"]]).statistic for x in rs]
        top = [max(x["p"], key=x["p"].get) == min(x["E"], key=x["E"].get) for x in rs]
        out[n] = (np.mean(rho), st.sem(rho), np.mean(top), np.std(top)/np.sqrt(len(top)))
    return out
A, B = summ(e2["ising"]), summ(e2b); ns = np.arange(3, 8)
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
ax = axs[0]
ax.axhline(0, color="0.5", lw=0.6)
ax.errorbar(ns - 0.08, [A[n][0] for n in ns], [A[n][1] for n in ns], fmt="o-", ms=3, color=C[1], label=r"$H=+\sum h s+\sum J ss$")
ax.errorbar(ns + 0.08, [B[n][0] for n in ns], [B[n][1] for n in ns], fmt="s-", ms=3, color=C[0], label=r"$H=-\sum h' s-\sum J' ss$")
ax.set(xlabel="spins $n$", ylabel=r"Spearman $\rho(p,\,-E)$", xticks=ns, ylim=(-0.75, 0.95)); ax.legend(loc="upper center", fontsize=5.5, ncol=1); panel(ax, "a")
ax = axs[1]
ax.plot(ns, [2.0**-n for n in ns], "k:", lw=0.8, label="chance")
ax.errorbar(ns - 0.08, [A[n][2] for n in ns], [A[n][3] for n in ns], fmt="o-", ms=3, color=C[1], label="as stated")
ax.errorbar(ns + 0.08, [B[n][2] for n in ns], [B[n][3] for n in ns], fmt="s-", ms=3, color=C[0], label="sign-flipped form")
ax.set(xlabel="spins $n$", ylabel="top-1 ground-state accuracy", xticks=ns, ylim=(-0.02, 0.6)); ax.legend(fontsize=6, loc="upper right"); panel(ax, "b")
ax = axs[2]
Rs = [d["R"] for d in e2["h2_exact"]]; Ex = [d["E_exact"] for d in e2["h2_exact"]]
ax.plot(Rs, Ex, "k-", lw=1, label="exact (FCI)")
jr = {R_: [x["E_read"] for x in e2["h2"] if x["R"] == R_] for R_ in Rs}
lv = np.linspace(-1.20, -0.75, 10); step = lv[1] - lv[0]
Ej = [lv[0] + np.mean([x["score"] for x in e2["h2"] if x["R"] == R_])*step for R_ in Rs]
ax.plot(Rs, Ej, "o--", ms=3, color=C[1], label="Jev read-out")
ax.set(xlabel=r"H$_2$ bond length (Å)", ylabel="energy (hartree)"); ax.legend(fontsize=6); panel(ax, "c")
fig.tight_layout(w_pad=1.2); save(fig, "fig3_hamiltonian")

# ---------- Fig 4: E3 + E4 chemistry ----------
e4 = json.load(open("experiments/e4_raw.json")); e4r = json.load(open("experiments/e4_results.json"))
Pm = np.array(r["Pm"]); S = ["mr", "rhf_ok", "t1", "unstable", "ccsdt_ok"]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.2))
ax = axs[0]
names = {"mr": "multireference", "t1": r"$T_1>0.02$", "unstable": "RHF unstable"}
for i, (k, lab) in enumerate(names.items()):
    y = np.array(r["labels"][k]).astype(int); p = Pm[:, S.index(k)]
    edges = np.linspace(0, 1, 6); idx = np.minimum((p*5).astype(int), 4)
    xs = [p[idx == b].mean() for b in range(5) if np.any(idx == b)]; ys = [y[idx == b].mean() for b in range(5) if np.any(idx == b)]
    ax.plot(xs, ys, "o-", ms=3, color=C[i], label=f"{lab} (AUC {r['e3'][k]['jev']['auroc']:.2f})")
ax.plot([0, 1], [0, 1], color="0.5", lw=0.6)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Jev probability", ylabel="observed frequency"); ax.legend(fontsize=5.5, loc="lower right"); panel(ax, "a")
ax = axs[1]
s = next(x for x in e4 if x["name"] == "N2" and x["stretch"] == 1.0)
lab = {"oracle": r"oracle $|c|^2$", "pt1": "first-order PT", "jev": "Jev", "excitation": "excitation rank", "random": "random"}
col = {"oracle": "k", "pt1": C[2], "jev": C[1], "excitation": C[0], "random": "0.6"}
for m in lab:
    ax.plot(np.arange(len(s["curves"][m])), np.maximum(s["curves"][m], 1e-4), color=col[m], lw=1, label=lab[m])
ax.axhline(1.6, color="0.4", ls=":", lw=0.7)
ax.set(yscale="log", xlabel="determinants added $k$", ylabel="error vs CASCI (mE$_h$)", ylim=(1e-2, 5e3), xlim=(0, 99))
ax.legend(fontsize=5.5, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2); panel(ax, "b")
ax = axs[2]
order = ["oracle", "pt1", "jev", "excitation", "diag", "random"]
labs = ["oracle", "PT1", "Jev", "exc.", "diag.", "rand."]
vals = [np.array(e4r["area"][m]) for m in order]
ax.boxplot(vals, widths=0.5, showfliers=False, medianprops=dict(color="k"))
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), i+1) + np.linspace(-0.12, 0.12, len(v)), v, s=5, color=C[1] if order[i] == "jev" else "0.4", zorder=3)
ax.set(xticks=range(1, 7), xticklabels=labs, ylabel=r"mean $\log_{10}$ error (mE$_h$)"); panel(ax, "c")
fig.tight_layout(w_pad=1.2); save(fig, "fig4_chemistry")
print("ok")
