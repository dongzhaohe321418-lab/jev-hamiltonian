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

# ---------- Fig 2: E1 coherence ----------
r = json.load(open("experiments/e1e3_results.json")); r1 = json.load(open("experiments/r1_results.json"))
e9r = json.load(open("experiments/e9_results.json")); e8r = json.load(open("experiments/e8_results.json"))
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
ax = axs[0]
Jab = np.concatenate([p[0] for p in r["J_pairs"]]); Jba = np.concatenate([p[1] for p in r["J_pairs"]])
pc = np.array(e9r["pos_control_J"]); lim = 4.2
ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=0.6)
ax.scatter(Jab, Jba, s=4, color=C[0], alpha=0.55, lw=0, label="molecules (48 cases)")
ax.scatter(pc[:, 0], pc[:, 1], s=7, marker="^", color=C[1], alpha=0.8, lw=0, label="stated population")
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"$J_{ab}$ (nats)", ylabel=r"$J_{ba}$ (nats)")
ax.legend(loc="upper left", handlelength=1, fontsize=6); panel(ax, "a")
ax = axs[1]
pv = np.array(r1["e1_matched"]["pvals"])
ax.hist(pv, bins=np.linspace(0, 1, 21), density=True, color=C[0], alpha=0.8)
ax.axhline(1, color="k", lw=0.7, ls="--"); ax.text(0.55, 1.25, "coherent model", fontsize=6)
ax.set(xlabel="p-value under matched coherent null", ylabel="density", xlim=(0, 1)); panel(ax, "b")
ax = axs[2]
h, hn = np.array(e8r["heldout"]), np.array(e8r["heldout_null"])
ax.scatter(hn, h, s=6, color=C[2], lw=0)
m_ = max(hn.max(), h.max()) * 1.05; ax.plot([0, m_], [0, m_], color="0.4", lw=0.6)
ax.axhline(np.median(e8r["fit_rms"]), color=C[1], lw=0.7, ls=":"); ax.text(0.03, np.median(e8r["fit_rms"]) + 0.03, "in-sample fit", fontsize=6, color=C[1])
ax.set(xlim=(0, m_), ylim=(0, m_), xlabel="RMS error, ignore second fact", ylabel="RMS error, pairwise Ising"); panel(ax, "c")
fig.tight_layout(w_pad=1.2); save(fig, "fig2_consistency")

# ---------- Fig 3: E2 in-context Hamiltonians (instance-level uncertainty) ----------
e2r = json.load(open("experiments/e2_results.json")); e2 = json.load(open("experiments/e2_raw.json"))
ns = np.arange(3, 8); A, B = e2r["stated"], e2r["flipped"]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
ax = axs[0]
ax.axhline(0, color="0.5", lw=0.6)
ax.errorbar(ns - 0.08, [A[str(n)]["rho"][0] for n in ns], [A[str(n)]["rho"][1] for n in ns], fmt="o-", ms=3, color=C[1], label=r"$H=+\sum h s+\sum J ss$")
ax.errorbar(ns + 0.08, [B[str(n)]["rho"][0] for n in ns], [B[str(n)]["rho"][1] for n in ns], fmt="s-", ms=3, color=C[0], label=r"$H=-\sum h' s-\sum J' ss$")
ax.set(xlabel="spins $n$", ylabel=r"Spearman $\rho(p,\,-E)$", xticks=ns, ylim=(-0.75, 0.95)); ax.legend(loc="upper center", fontsize=5.5); panel(ax, "a")
ax = axs[1]
ax.plot(ns, [2.0 ** -n for n in ns], "k:", lw=0.8, label="chance")
ax.errorbar(ns - 0.08, [A[str(n)]["top1"][0] for n in ns], [A[str(n)]["top1"][1] for n in ns], fmt="o-", ms=3, color=C[1], label="as stated")
ax.errorbar(ns + 0.08, [B[str(n)]["top1"][0] for n in ns], [B[str(n)]["top1"][1] for n in ns], fmt="s-", ms=3, color=C[0], label="sign-flipped form")
ax.set(xlabel="spins $n$", ylabel="top-1 ground-state accuracy", xticks=ns, ylim=(-0.02, 0.6)); ax.legend(fontsize=6, loc="upper right"); panel(ax, "b")
ax = axs[2]
e2c = json.load(open("experiments/e2c_raw.json"))
Rs = [d["R"] for d in e2["h2_exact"]]
w = [d["weights"]["1100"] for d in e2["h2_exact"]]
p_orig = [np.mean([x["p_dom"].get("1100", 0) for x in e2["h2"] if x["R"] == R]) for R in Rs]
p_rel_true = [np.mean([x["p"].get("0101", 0) for x in e2c if x["R"] == R]) for R in Rs]
p_rel_old = [np.mean([x["p"].get("1100", 0) for x in e2c if x["R"] == R]) for R in Rs]
ax.plot(Rs, w, "k-", lw=1, label="exact weight of HF state")
ax.plot(Rs, p_orig, "o--", ms=3, color=C[1], label=r"Jev, HF $=|1100\rangle$")
ax.plot(Rs, p_rel_true, "s--", ms=3, color=C[0], label=r"Jev, relabelled: HF $=|0101\rangle$")
ax.plot(Rs, p_rel_old, "^:", ms=3, color=C[2], label=r"Jev, relabelled: $p(|1100\rangle)$")
ax.set(xlabel=r"H$_2$ bond length (Å)", ylabel="probability / weight", ylim=(0, 1.05)); ax.legend(fontsize=5.2, loc="lower left", bbox_to_anchor=(0.0, 0.3)); panel(ax, "c")
fig.tight_layout(w_pad=1.2); save(fig, "fig3_hamiltonian")

# ---------- Fig 4: E3 (v2 labels, LOMO) + E4 v2 ----------
e3 = json.load(open("experiments/e3v2_results.json")); e4 = json.load(open("experiments/e4v2_raw.json")); e4r = json.load(open("experiments/e4v2_results.json"))
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 1]})
ax = axs[0]
preds = [("stretch", "stretch factor"), ("logistic", "logistic (stretch, gap)"), ("jev", "Jev"), ("jev_no_name", "Jev, no name"),
         ("jev_no_stretch_note", "Jev, no stretch note"), ("jev_geometry_only", "Jev, geometry only")]
labs = {"mr": "multiref.", "t1": r"$T_1$", "unstable": "RHF unstable"}
for j, (k, kl) in enumerate(labs.items()):
    for i, (m, ml) in enumerate(preds):
        v = e3[k][m]; y = len(preds) - i + 0.22 * (1 - j)
        ax.errorbar(v["auroc"], y, xerr=[[v["auroc"] - v["auroc_ci"][0]], [v["auroc_ci"][1] - v["auroc"]]], fmt="o", ms=2.5, color=C[j], lw=0.8,
                    label=kl if i == 0 else None)
ax.axvline(0.5, color="0.6", lw=0.6, ls=":")
ax.set(yticks=range(len(preds), 0, -1), yticklabels=[p[1] for p in preds], xlabel="AUROC (95% CI, molecule bootstrap)", xlim=(0.3, 1.02))
ax.tick_params(axis="y", which="both", length=0); ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8); panel(ax, "a")
ax = axs[1]
s = next(x for x in e4 if x["name"] == "N2" and x["stretch"] == 1.0)
lab = {"oracle": r"oracle $|c|^2$", "cipsi": "iterative CIPSI", "en_pt2": "EN-PT2 (one shot)", "jev_top1": "Jev", "excitation": "excitation rank", "random": "random"}
col = {"oracle": "k", "cipsi": C[3], "en_pt2": C[2], "jev_top1": C[1], "excitation": C[0], "random": "0.6"}
for m in lab:
    ax.plot(np.arange(len(s["curves"][m])), np.maximum(s["curves"][m], 1e-3), color=col[m], lw=1, label=lab[m], drawstyle="steps-post")
ax.axhline(1.6, color="0.4", ls=":", lw=0.7)
ax.set(yscale="log", xlabel="determinants in subspace $k$", ylabel="error vs CASCI (mE$_h$)", ylim=(1e-3, 3e5), xlim=(1, 100))
ax.legend(fontsize=5.0, loc="upper right", ncol=2, columnspacing=0.5, handlelength=1.0); panel(ax, "b")
ax = axs[2]
order = ["oracle", "cipsi", "en_pt2", "first_order", "jev_top1", "excitation", "jev_noul", "random"]
xl = ["oracle", "CIPSI", "EN-PT2", "1st order", "Jev", "excit.", "Jev Noul", "random"]
vals = [np.array(e4r["area"][m]) for m in order]
ax.boxplot(vals, widths=0.5, showfliers=False, medianprops=dict(color="k"))
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), i + 1) + np.linspace(-0.12, 0.12, len(v)), v, s=4, color=C[1] if order[i].startswith("jev") else "0.4", zorder=3)
ax.set(xticks=range(1, len(order) + 1), ylabel=r"mean $\log_{10}$ error (mE$_h$)"); ax.set_xticklabels(xl, fontsize=5.5, rotation=45, ha="right"); panel(ax, "c")
fig.tight_layout(w_pad=1.0); save(fig, "fig4_chemistry")
print("ok")
