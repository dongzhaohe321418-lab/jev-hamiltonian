"""Fig 6: Jev vs open-weight Qwen2.5 instruction models on the same probes (E7). Reads experiments/e7_results.json.
Style as figures.py: SciencePlots ['science','no-latex','bright'], serif 8 pt, 6.6 x 2.1 in; PDF + 300 dpi PNG."""
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex", "bright"])
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "legend.frameon": False})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
def panel(ax, s): ax.text(-0.22, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9)

res = json.load(open("experiments/e7_results.json")); jevr = json.load(open("experiments/e1e3_results.json"))
e2r = json.load(open("experiments/e2_results.json"))   # Jev Ising: mean and s.e.m. over instances, as in Fig. 3
inst = lambda d: {n: {"rho_mean": v["rho"][0], "rho_sem": v["rho"][1]} for n, v in d.items()}
series = [("Jev", C[1], np.abs(np.array(jevr["asym"])).ravel(), res["jev"]["e3"], inst(e2r["stated"]), inst(e2r["flipped"]))]
short = {"Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B"}
for i, (name, r) in enumerate(res["models"].items()):
    series.append((short[name], [C[0], C[2]][i], np.abs(np.array(r["_asym"])).ravel(), r["e3"], r["ising_stated"], r["ising_flipped"]))

fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.1))
ax = axs[0]
for lab, c, a, *_ in series:
    xs = np.sort(a); ax.plot(np.maximum(xs, 1e-3), np.arange(1, len(xs) + 1) / len(xs), color=c, lw=1.1, label=lab)
ax.set(xscale="log", xlim=(1e-2, 20), ylim=(0, 1), xlabel=r"$|J_{ab}-J_{ba}|$ (nats)", ylabel="cumulative fraction")
ax.legend(loc="upper left", handlelength=1.2); panel(ax, "a")

ax = axs[1]
mk = {"mr": ("o", "multireference"), "t1": ("s", r"$T_1>0.02$"), "unstable": ("^", "RHF unstable")}
for lab, c, _, e3, *_ in series:
    for k, (m, _) in mk.items():
        ax.scatter(e3[k]["auroc"], e3[k]["brier"], marker=m, s=14, color=c, lw=0.4, edgecolor="k" if lab == "Jev" else c)
for k, (m, nm) in mk.items(): ax.scatter([], [], marker=m, s=12, color="0.6", label=nm)
ax.axvline(0.5, color="0.6", lw=0.6, ls=":")
ax.set(xlabel="AUROC", ylabel="Brier score", xlim=(0.3, 1.0)); ax.legend(loc="center left", bbox_to_anchor=(0.3, 0.5), handletextpad=0.2); panel(ax, "b")

ax = axs[2]
ns = np.arange(3, 8); ax.axhline(0, color="0.5", lw=0.6)
for j, (lab, c, _, _, A, B) in enumerate(series):
    off = (j - 1) * 0.07
    ax.errorbar(ns + off, [A[str(n)]["rho_mean"] for n in ns], [A[str(n)]["rho_sem"] for n in ns], fmt="o-", ms=2.5, lw=0.9, color=c)
    ax.errorbar(ns + off, [B[str(n)]["rho_mean"] for n in ns], [B[str(n)]["rho_sem"] for n in ns], fmt="s--", ms=2.5, lw=0.9, color=c, mfc="white")
ax.plot([], [], "ko-", ms=2.5, lw=0.9, label=r"as stated, $+\sum hs+\sum Jss$")
ax.plot([], [], "ks--", ms=2.5, lw=0.9, mfc="white", label=r"sign-flipped, $-\sum h's-\sum J'ss$")
ax.set(xlabel="spins $n$", ylabel=r"Spearman $\rho(p,\,-E)$", xticks=ns, ylim=(-0.8, 1.0)); ax.legend(loc="upper center", fontsize=5.5); panel(ax, "c")
fig.tight_layout(w_pad=1.0)
fig.savefig("paper/figs/fig6_models.pdf"); fig.savefig("paper/figs/fig6_models.png", dpi=300)
print("ok")
