# Pre-registered plan (written 2026-09-23, before E1–E4 data were collected)

Energy of an answer: E(y|x) = -log p(y|x), p clipped to [0.005, 0.995] (API rounds to 0.01; zeros occur).
E0 (done before this plan): outputs are stochastic (sd ~0.02-0.03 across identical calls); parallel
questions answered alone vs together agree within that noise. Consequence: every probe below uses R=3 repeats
and reports means with bootstrap 95% CIs.

## E1 Does a consistent landscape exist? (conditional compatibility)
Binary statements s_1..s_5 about a molecule (multireference; RHF qualitatively correct; open-shell ground state;
CCSD(T) within 1 kcal/mol; low-lying excited state within 1 eV). For each of M molecules we measure p(s_i) and
p(s_i | s_j = v), v in {0,1}, by stating "Established fact: s_j is TRUE/FALSE" in the state.
A joint Gibbs distribution over pairs exists only if
 (a) coupling symmetry: J_ij := logit p(i|j=1) - logit p(i|j=0) equals J_ji (odds ratio is symmetric);
 (b) marginal consistency: p(i) = p(i|j=1) p(j) + p(i|j=0) (1 - p(j)).
Positive control: statements with a known logical/arithmetical coupling, where a coherent model must give J_ij = J_ji
(the same test applied to an exact Ising model's conditionals, and to a set of number statements).
Report: distribution of |J_ij - J_ji|, relative asymmetry, marginal-consistency residuals versus the noise floor
(estimated from repeats). Primary criterion: fraction of pairs whose asymmetry exceeds 3x the noise floor.

## E2 Can Jev represent a physical Hamiltonian given in context?
Random 2-local classical Ising Hamiltonians, n = 3..7 spins (J, h ~ U[-1,1]), all 2^n configurations as choice
options. Plus the 4-qubit Jordan-Wigner H2/STO-3G Hamiltonian at 8 bond lengths (question: which basis state has the
largest ground-state weight), and energy read-out via a 10-level score. Metrics: top-1 ground-state accuracy vs
uniform (1/2^n); Spearman rho between log p(s) and -E(s); maximum-likelihood inverse temperature beta; energy
read-out error in mHa (vs chemical accuracy 1.6 mHa).

## E3 Quantum-chemistry decisions with computed ground truth
~60 molecule/geometry cases (equilibrium and stretched). Labels from PySCF: multireference flag (CASCI HF weight
c0^2 < 0.90 in a fixed active space), T1 diagnostic > 0.02 (CCSD), RHF->UHF instability. Jev gives noul probabilities.
Metrics: AUROC, Brier, ECE (10 bins). Baselines: base rate; logistic regression on cheap HF descriptors
(HOMO-LUMO gap, bond stretch ratio), leave-one-out.

## E4 Jev as an importance prior for selected CI
Small active spaces (<= 255 determinants). Jev sees orbital energies/occupations and the determinant list and gives
a choice distribution over determinants. Selected CI using the top-k determinants ranked by: Jev, exact |c|^2
(oracle), Epstein-Nesbet first-order estimate (cheap baseline), random. Metric: energy error (mHa) vs k.

Anything not pre-registered here is reported as exploratory.
