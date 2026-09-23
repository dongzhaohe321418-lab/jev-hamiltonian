"""Figure 5 (E5 + E6): paper/figs/fig5_scaleup.{pdf,png}. Style as experiments/figures.py."""
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex", "bright"])
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "legend.frameon": False})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
def panel(ax, s): ax.text(-0.22, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9)

e5 = json.load(open("experiments/e5_results.json")); e6 = json.load(open("experiments/e6_results.json")); e5l = json.load(open("experiments/e5_lomo_results.json"))
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.2))

# a: E5 AUROC with bootstrap CIs
ax = axs[0]
labs = [("mr", "multiref."), ("t1", r"$T_1>0.02$"), ("unstable", "RHF unstable")]
meths = [("stretch", "stretch factor", "0.45", "D"), ("jev", "Jev", C[0], "o"), ("jev_geometry_only", "Jev, geometry only", C[1], "s"), ("logistic", "logistic (stretch, gap)", C[2], "^")]
for j, (m, name, col, mk) in enumerate(meths):
    xs = np.arange(len(labs)) + (j - 1.5) * 0.18
    g = lambda k: e5l[k]["stretch_raw"] if m == "stretch" else e5l[k][m]
    a = [g(k)["auroc"] for k, _ in labs]
    lo = [g(k)["auroc_ci"][0] for k, _ in labs]; hi = [g(k)["auroc_ci"][1] for k, _ in labs]
    ax.errorbar(xs, a, [np.subtract(a, lo), np.subtract(hi, a)], fmt=mk, ms=3.5, color=col, lw=0.8, capsize=1.5, label=name)
ax.axhline(0.5, color="0.5", ls=":", lw=0.7)
ax.set(xticks=range(len(labs)), ylim=(0.18, 1.0), ylabel="AUROC (95% CI)")
ax.set_xticklabels([l for _, l in labs])
ax.tick_params(axis="x", which="minor", bottom=False, top=False)
ax.legend(loc="lower left", fontsize=6, handlelength=1, borderpad=0.2, labelspacing=0.25); panel(ax, "a")

# b: spin-state task: Jev p(triplet) by CCSD(T) ground multiplicity
ax = axs[1]
rows = e5["spin"]["rows"]; rng = np.random.default_rng(1)
for yi, (t, name, col) in enumerate([("1", "singlet", C[0]), ("3", "triplet", C[3])]):
    rr = [r for r in rows if r["truth"] == t]
    x = np.array([r["jev_p"]["3"] for r in rr]); y = yi + rng.uniform(-0.18, 0.18, len(rr))
    near = np.array([r["gap_to_next_eh"] <= 0.010 for r in rr])
    ax.scatter(x[~near], y[~near], s=9, color=col, lw=0)
    ax.scatter(x[near], y[near], s=11, facecolor="none", edgecolor=col, lw=0.7)
ax.axvline(0.5, color="0.5", ls=":", lw=0.7)
ax.set(yticks=[0, 1], ylim=(-0.5, 1.5), xlim=(0, 1), xlabel="Jev $p$(triplet)")
ax.set_yticklabels(["singlet\nground", "triplet\nground"])
ax.tick_params(axis="y", which="minor", left=False, right=False)
acc = e5["spin"]["all"]["jev"]["accuracy"]
ax.text(0.03, 1.38, f"accuracy {acc:.2f} (n={e5['spin']['all']['n']})\nalways singlet {e5['spin']['all']['always_singlet']['accuracy']:.2f}",
        fontsize=6, va="top")
ax.scatter([], [], s=11, facecolor="none", edgecolor="0.3", lw=0.7, label="gap < 10 mE$_h$")
ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderpad=0.2); panel(ax, "b")

# c: E6 routing curves
ax = axs[2]
for k, name, col in [("jev", "Jev threshold", C[0]), ("logreg", "logistic threshold", C[2])]:
    c = sorted(e6["curves"][k], key=lambda d: (d["frac_routed"], -d["frac_err_gt_1p6"]))
    ax.plot([d["frac_routed"] for d in c], [d["frac_err_gt_1p6"] for d in c], "-", color=col, lw=1, label=name, drawstyle="default")
base = e6["policies"]["always_ccsdt"]["frac_err_gt_1p6"]
ax.plot([0, 1], [base, 0], color="0.6", ls=":", lw=0.8, label="random routing")
P = e6["policies"]
ax.plot(P["t1_gt_0.02"]["frac_routed"], P["t1_gt_0.02"]["frac_err_gt_1p6"], "D", ms=3.5, color=C[4], label=r"$T_1>0.02$")
ax.plot(P["stretch_gt_1.0"]["frac_routed"], P["stretch_gt_1.0"]["frac_err_gt_1p6"], "P", ms=4, color="0.3", label="stretch $>1$")
ax.plot(P["jev_tau0.5"]["frac_routed"], P["jev_tau0.5"]["frac_err_gt_1p6"], "o", ms=3.5, color=C[0], mfc="white", label=r"Jev, $p<0.5$")
ax.plot(P["oracle"]["frac_routed"], 0, "*", ms=5, color="k", label="oracle")
ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, base + 0.07), xlabel="fraction routed to MR method",
       ylabel=r"fraction with error $>$1.6 mE$_h$")
ax.legend(loc="upper right", fontsize=5.5, handlelength=1.2, borderpad=0.2, labelspacing=0.2); panel(ax, "c")

fig.tight_layout(w_pad=1.0)
fig.savefig("paper/figs/fig5_scaleup.pdf"); fig.savefig("paper/figs/fig5_scaleup.png", dpi=300)
