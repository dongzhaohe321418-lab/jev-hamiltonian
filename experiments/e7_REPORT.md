# E7 (exploratory): the same probes on open-weight instruction models

This experiment was not pre-registered. It was added after E1–E3 to ask whether Jev's incoherence (E1, E1b) and its reading of coefficient signs (E2/E2b) are specific to Jev or show up in general instruction models too.
Models: **Qwen2.5-1.5B-Instruct** and **Qwen2.5-7B-Instruct**. The 7B model ran in bf16 on MPS, so the 3B fallback was not needed.
The Jev API was not called. Jev's numbers are recomputed from `e1e3_results.json`, `e1b_e3b_raw.json`, `e2_raw.json`, `e2b_raw.json` and `e3v2_results.json`.

Files: `e7_collect.py` (collection), `e7_analyze.py` (produces `e7_results.json`), `e7_figure.py` (produces `paper/figs/fig6_models.{pdf,png}`).
Raw data: `e7_raw_Qwen2.5-1.5B-Instruct.json`, `e7_raw_Qwen2.5-7B-Instruct.json`, and `e7_raw_Qwen2.5-1.5B-Instruct_fp32.json` (a precision check, not used in the comparison).

## Models, software, hardware

| | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct |
|---|---|---|
| HF revision (commit) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| weight blobs (HF LFS sha256) | model.safetensors `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` | 00001 `a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7`; 00002 `f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185`; 00003 `8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5`; 00004 `1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd` |
| weights location | `~/Documents/jev-hamiltonian-models/hub/` (HF_HOME, outside the repo) | same |
| model load (s) | 1.96 | 11.09 |
| runtime E1 marginals + conditionals / E1b negations / Ising (s) | 299.8 / 28.5 / 172.9 | 1146.0 / 108.2 / 1223.0 |

Apple M4 Pro, 24 GB unified memory, MPS backend. Python 3.13.5, torch 2.14.0, transformers 5.17.0, accelerate 1.15.0, huggingface_hub 1.32.0, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, SciencePlots 2.2.2.
Both snapshots were downloaded with `allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']`, so README/LICENSE were skipped. That is why the raw files record `snapshot_err: IncompleteSnapshotError`. The weights and tokenizer are complete.

Numerics: the transformer body runs in **bf16**. The final projection (last hidden state × `lm_head`) runs in **float32** (see "Code history" for why). Everything is deterministic: there is no sampling, R = 1 per prompt, and there are no repeats.

## Probability readout

**Yes/No (the analogue of Jev's Noul).** Each prompt asks one statement. We take the next-token logits at the first assistant position and restrict the softmax to four token ids. The ids are the same for both models, since they share a tokenizer:

| string | token id | token |
|---|---|---|
| `Yes` | 9454 | `Yes` |
| ` Yes` | 7414 | `ĠYes` |
| `No` | 2753 | `No` |
| ` No` | 2308 | `ĠNo` |

p(yes) = (e^ℓ[9454] + e^ℓ[7414]) / (e^ℓ[9454] + e^ℓ[7414] + e^ℓ[2753] + e^ℓ[2308]).
Lowercase variants were not included. Check of the readout: the full-vocabulary probability mass on these four tokens has median 0.99993 and minimum 0.99979 for 1.5B, and median 0.9999997 and minimum 0.99994 for 7B. The argmax token over the whole vocabulary was one of the four in 100% of the 2,400 Yes/No prompts per model: 1.5B gave Yes 1316 / No 1084, 7B gave Yes 573 / No 1827.

**Choice (Ising).** The options are the same `+`/`-` label strings Jev saw (e.g. `++-+`), in the same order: `itertools.product([1,-1])`, which starts with all-plus and ends with all-minus. Each label is scored by the summed log-probability of its tokens as the assistant's answer. The scores are then softmax-normalised over the 2^n options. Labels take 1–5 tokens. No end-of-turn token is scored, but all labels have the same character length and none is a prefix of another. The script asserts that tokenising prompt+label keeps the prompt tokens unchanged.
Implementation: the prompt is prefilled in 512-token chunks into one KV cache. Then all label continuations go through **one** forward pass with a tree attention mask. Each label token sees the whole prompt plus the earlier tokens of its own label, at positions L, L+1, … . This gives exactly each label's conditional log-probability. Validation in fp32 (1.5B, 3 instances, n = 3, 4, 6) against the per-label forward-and-crop method: max |Δ log p| ≤ 8.7e-5. In bf16 the same comparison differs by up to 0.26 nats in log p, but by at most 0.0065 in normalised p. That is bf16 rounding, not a difference between the methods.

## Prompt templates (verbatim; applied with the model's own chat template, `add_generation_prompt=True`)

Yes/No system message:
```
You are an expert in quantum chemistry and electronic-structure theory. Answer with a single word.
```
Yes/No user message (`{state}` = `json.dumps(state_of(row, cond), indent=1)` from `e1e3_collect.py`; `{statement}` = the E1 statement `S[k]`, or the negation `NEG[k]` from `e1b_e3b.py`):
```
Input state (JSON):
{state}

Statement: {statement}

Is this statement true for the system described in the input state? Answer with one word: Yes or No.
```
Choice system message:
```
You are an expert in statistical physics. Answer with the option label only.
```
Choice user message (`{state}` = the same dict as `e2_hamiltonian.ising_job`, or the sign-flipped `e2b_convention.job`; `{question}` = `Which spin configuration is the ground state, i.e. minimises H(s)?`; `{options}` = one line per option, `<label>: spins (s_0=+1, s_1=-1, ...)`, i.e. Jev's criteria):
```
Input state (JSON):
{state}

Question: {question}

Options:
{options}

Answer with the label of exactly one option (the string of + and - signs), nothing else.
```
Full rendered example: H2 at 1.0× stretch, condition "mr is TRUE", statement t1. It is stored as `example_yn_prompt` in each raw file:
```
<|im_start|>system
You are an expert in quantum chemistry and electronic-structure theory. Answer with a single word.<|im_end|>
<|im_start|>user
Input state (JSON):
{
 "system": "H2 molecule",
 "method_context": "electronic structure, cc-pVDZ basis, singlet reference",
 "geometry_xyz_angstrom": "H 0 0 0; H 0 0 0.741",
 "geometry_note": "all bonds stretched to 1.0 x equilibrium length",
 "known": "Established fact (computed): The ground state has strong multireference (static correlation) character: the Hartree-Fock determinant carries less than 90% of the exact wavefunction weight. This statement is TRUE."
}

Statement: The CCSD T1 diagnostic exceeds 0.02.

Is this statement true for the system described in the input state? Answer with one word: Yes or No.<|im_end|>
<|im_start|>assistant
```

## Differences from the Jev protocol (read before comparing)

- Jev answered all questions about one state in a single parallel call and was repeated 3 times. The Qwen models get **one statement per prompt**, once each. For E1b this means p(s) and p(not s) come from separate prompts, whereas Jev was asked them together.
- Jev returns probabilities rounded to 0.01 and has per-call noise with s.d. 0.013. Qwen probabilities are exact softmax values. The coherent null for Qwen therefore has zero noise and no rounding: a coherent model would give |J_ab − J_ba| = 0 exactly, up to the clip, so every nonzero asymmetry is a violation. The bootstrap-SE criterion (3 SE) is undefined at R = 1. As instructed, we report the fraction above **Jev's** noise-null 95th percentile, 0.357 nats (recomputed from `e1e3_results.json["null_abs"]`), and the full distribution.
- The logit clip [0.005, 0.995] from `analyze_e1e3.py` is kept for comparability. It binds rarely for 1.5B but for 48% of 7B probabilities (see below), so the 7B E1 numbers need care.
- E2: the same instances as `ising_instances()` (seed 20260923, 10 instances per n, n = 3–7), in both forms. Qwen has 1 record per instance and Jev has 3 repeats. Summaries use the same method as `figures.py`: mean and s.e.m. of per-record Spearman ρ(p, −E), and mean top-1.

## Results

### E1: coupling symmetry and total probability (48 cases × 10 pairs = 480 pair-cases)

| E1 statistic | Jev (jev-1.13.0) | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| median \|J_ab−J_ba\| (nats) | 0.26 | 0.69 | 1.16 |
| fraction \|J_ab−J_ba\| > 0.357 (Jev-noise null 95th pct) | 0.352 | 0.798 | 0.735 |
| median \|J\| (nats) | 1.10 | 2.14 | 0.58 |
| total-probability residual \|·\|, median | 0.041 | 0.112 | 0.320 |
| total-probability residual \|·\|, 90th pct | 0.138 | 0.358 | 0.908 |
| \|asym\| percentiles 10/25/50/75/90/95 | 0.04/0.12/0.26/0.47/0.71/0.83 | 0.23/0.42/0.69/1.00/1.48/2.13 | 0.00/0.30/1.16/3.74/7.81/8.66 |
| fraction \|asym\| > 0.1 / > 1 nat | – | 0.952 / 0.252 | 0.829 / 0.533 |
| fraction of p outside [0.005, 0.995] | – | 0.0005 | 0.478 |
| pair-cases with ≥ 1 clipped conditional | – | 0.002 | 0.758 |
| unclipped pair-cases only: n; median \|asym\|; fraction > 0.357 | – | 479; 0.68; 0.797 | 116; 0.39; 0.534 |
| clip 1e-6 instead of 0.005: median \|asym\|; median \|J\| | – | 0.69; 2.14 | 3.59; 3.10 |

Mean couplings for the logically related pairs:

| mean J (nats) | Jev | 1.5B | 7B |
|---|---|---|---|
| J(mr \| rhf_ok) | −2.43 | +2.04 | −8.39 |
| J(rhf_ok \| mr) | −2.35 | +4.18 | +0.00 |
| J(t1 \| ccsdt_ok) | −0.72 | +1.93 | +0.00 |
| J(ccsdt_ok \| t1) | −1.10 | +2.90 | +2.93 |
| J(mr \| unstable) | +1.07 | +1.71 | +7.77 |
| J(unstable \| mr) | +0.98 | +2.16 | +4.05 |

What this shows:
- **1.5B.** All 20 mean couplings are positive, including the logically forced anticorrelation between mr and rhf_ok. Stating any fact as "TRUE" raises p(yes) for every other statement. That is acquiescence to the word TRUE, not a model of the chemistry.
- **7B.** Probabilities are strongly saturated. p(t1) is below 1.4e-5 in every marginal. When both conditionals in one direction sit at the same clip bound, J = 0 in that direction (e.g. J(rhf_ok|mr), J(t1|ccsdt_ok)), while the other direction can be several nats. So the 7B asymmetry distribution is dominated by saturation and depends on the clip: the median is 1.16 at clip 0.005 and 3.59 at clip 1e-6. On the 116 pair-cases where no conditional is clipped, the median asymmetry is 0.39 and 53% exceed 0.357.
- By every E1 measure, both Qwen models are **less** coherent than Jev: a larger asymmetry, a larger fraction above the Jev-noise null, and 3–8× larger total-probability residuals.

### E1b: negation sums p(s) + p(not s) (48 cases; Jev: 48 × 3 calls)

| statement | Jev: mean [min, max]; frac \|dev\|>0.1 | 1.5B | 7B |
|---|---|---|---|
| mr | 1.03 [0.94, 1.19]; 0.12 | 1.23 [1.03, 1.45]; 0.96 | 0.58 [0.21, 1.10]; 0.96 |
| rhf_ok | 1.12 [1.00, 1.25]; 0.68 | 1.51 [1.37, 1.59]; 1.00 | 0.99 [0.79, 1.28]; 0.23 |
| t1 | 1.13 [1.02, 1.21]; 0.80 | 0.81 [0.70, 1.09]; 0.92 | 0.03 [0.00, 0.11]; 1.00 |
| unstable | 0.96 [0.89, 1.05]; 0.01 | 1.53 [1.43, 1.62]; 1.00 | 1.47 [0.87, 1.69]; 1.00 |
| ccsdt_ok | 1.03 [0.94, 1.23]; 0.13 | 1.27 [1.15, 1.35]; 1.00 | 1.29 [0.79, 1.55]; 0.92 |

Overall ranges: Jev 0.89–1.25, 1.5B 0.70–1.62, 7B 0.00–1.69.
For 7B's t1 pair, the model says "No" to both "T1 exceeds 0.02" and "T1 does NOT exceed 0.02", so the sum is 0.03.

### E3: quantum-chemistry labels (marginal p(yes); AUROC / Brier / ECE, 10 bins)

Primary labels are `qc_labels_v2.json`: mr, t1_flag with the one None skipped (so n = 47), and rhf_unstable. Jev's v2 numbers are computed from `e1e3_results.json["Pm"]`. `e7_analyze.py` asserts they match `e3v2_results.json`.

| label (n, positive rate) | Jev | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| mr (48, 0.646) | 0.760 / 0.207 / 0.193 | 0.402 / 0.265 / 0.191 | 0.748 / 0.547 / 0.578 |
| t1 (47, 0.638) | 0.814 / 0.170 / 0.174 | 0.473 / 0.281 / 0.220 | 0.612 / 0.638 / 0.638 |
| unstable (48, 0.771) | 0.888 / 0.184 / 0.283 | 0.408 / 0.198 / 0.144 | 0.740 / 0.195 / 0.193 |

Ranges of the models' probabilities:
- 1.5B: mr 0.37–0.62, t1 0.35–0.58, unstable 0.54–0.71. These are nearly flat.
- 7B: mr 1e-4–0.50, t1 1.4e-6–1.4e-5, unstable 0.71–0.994.

The 7B t1 AUROC of 0.61 is a ranking of probabilities that are all near zero. Its Brier score of 0.638 equals the positive rate, because the model always says No.

For reference, the v2 leave-one-molecule-out baselines from `e3v2_results.json` (AUROC / Brier):

| label | base rate | stretch only | logistic (stretch, gap) |
|---|---|---|---|
| mr | 0.284 / 0.236 | 0.867 / 0.123 | 0.894 / 0.118 |
| t1 | 0.348 / 0.234 | 0.706 / 0.189 | 0.790 / 0.171 |
| unstable | 0.269 / 0.182 | 0.870 / 0.110 | 0.899 / 0.104 |

Secondary, with the original `qc_labels.json` labels (AUROC / Brier / ECE):

| label | Jev | 1.5B | 7B |
|---|---|---|---|
| mr | 0.740 / 0.207 / 0.132 | 0.418 / 0.262 / 0.167 | 0.709 / 0.489 / 0.516 |
| t1 | 0.848 / 0.168 / 0.156 | 0.544 / 0.276 / 0.207 | 0.587 / 0.625 / 0.625 |
| unstable | 0.888 / 0.184 / 0.283 | 0.408 / 0.198 / 0.144 | 0.740 / 0.195 / 0.193 |

### E2 / E2b: in-context Ising Hamiltonians (10 instances per n)

Spearman ρ(p, −E) is given as mean (s.e.m.). Top-1 is the fraction of records whose argmax is the ground state.

As stated, H = +Σh s + ΣJ s s:

| n | Jev ρ | Jev top-1 | 1.5B ρ | 1.5B top-1 | 7B ρ | 7B top-1 | chance |
|---|---|---|---|---|---|---|---|
| 3 | −0.258 (0.067) | 0.10 | −0.043 (0.123) | 0.10 | −0.071 (0.090) | 0.00 | 0.125 |
| 4 | −0.407 (0.046) | 0.00 | +0.001 (0.108) | 0.00 | −0.009 (0.064) | 0.10 | 0.0625 |
| 5 | −0.327 (0.035) | 0.00 | −0.000 (0.077) | 0.30 | −0.022 (0.078) | 0.10 | 0.0312 |
| 6 | −0.280 (0.037) | 0.00 | −0.017 (0.042) | 0.00 | −0.052 (0.054) | 0.00 | 0.0156 |
| 7 | −0.183 (0.015) | 0.00 | +0.022 (0.048) | 0.00 | +0.028 (0.051) | 0.00 | 0.0078 |

Sign-flipped, H = −Σh′ s − ΣJ′ s s with the same energies:

| n | Jev ρ | Jev top-1 | 1.5B ρ | 1.5B top-1 | 7B ρ | 7B top-1 | chance |
|---|---|---|---|---|---|---|---|
| 3 | +0.093 (0.088) | 0.33 | −0.029 (0.141) | 0.10 | −0.088 (0.100) | 0.00 | 0.125 |
| 4 | +0.341 (0.048) | 0.13 | +0.001 (0.105) | 0.00 | +0.008 (0.074) | 0.10 | 0.0625 |
| 5 | +0.344 (0.047) | 0.33 | +0.002 (0.077) | 0.30 | −0.021 (0.073) | 0.20 | 0.0312 |
| 6 | +0.237 (0.022) | 0.10 | −0.016 (0.044) | 0.00 | −0.056 (0.053) | 0.00 | 0.0156 |
| 7 | +0.129 (0.020) | 0.00 | +0.023 (0.049) | 0.00 | +0.027 (0.053) | 0.00 | 0.0078 |

Field/coupling decomposition in the as-stated form, ρ(p, +Σh s) / ρ(p, +ΣJ s s) for n = 3..7:
- Jev: +0.33/+0.09, +0.31/+0.21, +0.51/+0.05, +0.40/+0.10, +0.29/+0.04. These reproduce the paper's figures.
- 1.5B: +0.00/+0.01, −0.06/−0.03, −0.08/+0.03, −0.07/+0.05, +0.07/−0.05.
- 7B: +0.02/−0.00, −0.01/+0.03, −0.01/+0.04, −0.04/+0.08, +0.06/−0.04.

What this shows:
- Neither Qwen model's preferences track the energy (|ρ| ≤ 0.09 for every n and form) or the printed field signs.
- Rewriting the Hamiltonian in the other sign convention barely changes their ρ. The largest change is 0.017 (7B, n = 3 and n = 4), against Jev's swing of up to 0.75 (n = 4: −0.41 → +0.34).
- Their choices are dominated by label position or shape:
  - 1.5B picked the first option (all-plus) in 10/10 instances for n = 3 and n = 4 in both forms.
  - 7B picked the last option (all-minus) in 10/10 instances for n = 4 (both forms), in 10/10 (as stated) and 9/10 (flipped) for n = 5, and in 7/10 for flipped n = 3.
  - For n = 6 and 7, neither model's argmax was the first or last option.
- The sign-reading pattern is therefore **specific to Jev** among these three models. The open-weight models show no sensitivity to the Hamiltonian at all, not a sign-convention error.

## Precision check (1.5B, whole network in float32 vs bf16 body + fp32 head)

Same prompts, `E7_FP32=1`, file `e7_raw_Qwen2.5-1.5B-Instruct_fp32.json`.

| quantity | value |
|---|---|
| per-prompt \|Δp\|, E1 conditionals | median 0.0074, max 0.060 |
| per-pair-case \|Δ\|asym\|\| | median 0.091, 90th pct 0.195 nats |
| fp32 E1: median \|asym\| | 0.68 (bf16: 0.69) |
| fp32 E1: fraction > 0.357 | 0.800 (bf16: 0.798) |
| fp32 E1: residual median / 90th pct | 0.120 / 0.360 |
| fp32 E3 v2 AUROC/Brier: mr | 0.385 / 0.266 |
| fp32 E3 v2 AUROC/Brier: t1 | 0.443 / 0.284 |
| fp32 E3 v2 AUROC/Brier: unstable | 0.430 / 0.199 |
| fp32 E1b ranges | mr 1.06–1.44, rhf_ok 1.33–1.57, t1 0.65–1.05, unstable 1.41–1.58, ccsdt_ok 1.12–1.32 |
| fp32 Ising ρ, as stated, n = 3..7 | −0.043, −0.004, +0.005, −0.018, +0.019 |
| fp32 Ising ρ, flipped, n = 3..7 | −0.012, −0.004, +0.005, −0.018, +0.018 |
| fp32 Ising top-1 | identical to bf16 |

bf16 rounding in the body adds about 0.1 nats of scatter to individual asymmetries. It does not change any conclusion.
The 7B run was not repeated in fp32. It would need about 30 GB of weights, more than the machine's 24 GB.

## Code history of `e7_collect.py` and which runs are canonical

1. **First version, 22:52.** Logits came from `model(...).logits`, all in bf16. The first 1.5B run (23:11) used it and is **superseded**; its raw file was overwritten. Reason: bf16 logits around 16–32 are quantised in steps of 0.125, so log-odds, and therefore J, were quantised at a scale close to the 0.36-nat threshold. Examples: E1 median asym 0.75, fraction above 0.357 0.84. These numbers are not used anywhere.
2. **Around 23:12: fp32 output projection.** The last hidden state × `lm_head` is now computed in float32 (`head()`), and the `E7_FP32` switch was added. The canonical 1.5B E1/E1b data (run finished 23:24) and the fp32 check E1/E1b (23:38) come from this version. The canonical 7B E1/E1b data (saved at 01:00) also come from it, plus step 3's chunked prefill, which does not affect Yes/No prompts of about 200 tokens.
3. **After the first 7B attempt stalled, before the 7B run that produced the canonical E1/E1b data: chunked prefill and resumable saving.** The Choice prefill now runs in 1024-token chunks (later 512). Sections are saved as they finish, and a restart merges into the existing file. Reason: the first 7B Ising attempt stalled at n = 7 prompts of about 7,100 tokens. The machine went into swap, with about 9.6 GB of 10 GB swap used.
4. **01:32: tree-attention label scoring, and fp32 head computed chunk-wise without a resident fp32 copy.** Two reasons:
   - Per-label forwards with `cache.crop` took 25–40 s per label on the 7B n = 7 prompts under memory pressure. That is 128 labels × 20 instances, which was not feasible.
   - A resident 2.2 GB fp32 copy of the 7B `lm_head` added to that pressure.

   Converting bf16 to fp32 is exact, so the head output does not change. Tree attention computes exactly the same conditional log-probabilities; see the validation above.

   **All Ising data in all three raw files were re-collected with this final version:** 7B finished at 01:53, 1.5B at 01:56, and the fp32 1.5B check at 01:59. The earlier 1.5B Ising results from the per-label method were overwritten. They gave the same conclusions, with every ρ within about 0.01 of the canonical value.

Canonical data are therefore:
- `e7_raw_Qwen2.5-1.5B-Instruct.json`: E1/E1b from 23:24 (step 2), Ising from 01:56 (step 4).
- `e7_raw_Qwen2.5-7B-Instruct.json`: E1/E1b saved at 01:00 (steps 2–3), Ising from 01:53 (step 4).
- The `_fp32` file is the precision check only.

The `runtime_s` values are for these canonical sections. The first, abandoned 7B Ising attempts are not counted in them.

## Bottom line for the paper

- **Incoherence is not specific to Jev.** Both open-weight models violate coupling symmetry, total probability and negation normalisation by more than Jev does. Examples: median asymmetry 0.69 and 1.16 nats versus Jev's 0.26; total-probability residual medians 0.11 and 0.32 versus 0.041. The 1.5B model's couplings are all positive, including the logically forced one. The 7B model's are dominated by saturated probabilities.
- **Sign-reading is specific to Jev.** Given the same Ising Hamiltonians, the Qwen models' preferences do not correlate with the energy or with the printed field signs (|ρ| ≤ 0.09), and flipping the sign convention changes nothing. Jev's ρ goes from about −0.3 to about +0.3.
- **E3.** On the v2 labels Jev ranks better (AUROC 0.76 / 0.81 / 0.89) than 7B (0.75 / 0.61 / 0.74) and 1.5B (0.40 / 0.47 / 0.41, below chance). Its Brier scores are also far better than 7B's (0.21 / 0.17 / 0.18 versus 0.55 / 0.64 / 0.20).
- Caveats: this compares two small open models with one prompt template and no prompt tuning. The single-statement format differs from Jev's parallel queries. The analysis was not pre-registered.
