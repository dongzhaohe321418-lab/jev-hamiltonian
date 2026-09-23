"""E7 (exploratory): the E1/E1b/E3/E2/E2b probes applied to open-weight instruction models.

Usage (from repo root):
  HF_HOME=~/Documents/jev-hamiltonian-models .venv/bin/python experiments/e7_collect.py Qwen/Qwen2.5-7B-Instruct
Writes experiments/e7_raw_<short>.json. Deterministic (logits only, no sampling), so R = 1.

Readouts
- Yes/No (Noul analogue): one statement per prompt; p(yes) = softmax of the next-token logits at the first
  assistant position restricted to the token ids of 'Yes', ' Yes', 'No', ' No':
  p(yes) = (e^Yes + e^' Yes') / (e^Yes + e^' Yes' + e^No + e^' No'). The full-vocabulary probability mass on
  these four tokens is stored as `mass` for diagnostics.
- Choice (Ising): each option label (the +/- string keys Jev saw) is scored by the summed log-probability of its
  tokens as the assistant answer; scores are softmax-normalised over the 2^n options.
"""
import itertools, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from e1e3_collect import S, state_of      # __main__-guarded: importing only defines S, FACT, state_of
import e2_hamiltonian as e2                # __main__-guarded: defines ising_instances (module-level rng, seed 20260923)
from transformers import AutoModelForCausalLM, AutoTokenizer
CHUNK = 512
DTYPE = torch.float32 if os.environ.get("E7_FP32") == "1" else torch.bfloat16   # E7_FP32=1: precision check only

# Copied verbatim from experiments/e1b_e3b.py (that script has no __main__ guard, so it is not imported).
NEG = {"mr": "The ground state does NOT have strong multireference character: the Hartree-Fock determinant carries at least 90% of the exact wavefunction weight.",
       "rhf_ok": "Restricted Hartree-Fock is NOT a qualitatively correct single-determinant reference for this system.",
       "t1": "The CCSD T1 diagnostic does NOT exceed 0.02.",
       "unstable": "The RHF solution is stable: no spin-unrestricted (UHF) solution with lower energy exists.",
       "ccsdt_ok": "CCSD(T) will NOT be within 1 kcal/mol of the exact (full CI) energy in this basis."}

SYS_YN = "You are an expert in quantum chemistry and electronic-structure theory. Answer with a single word."
USER_YN = ("Input state (JSON):\n{state}\n\nStatement: {statement}\n\n"
           "Is this statement true for the system described in the input state? Answer with one word: Yes or No.")
SYS_CH = "You are an expert in statistical physics. Answer with the option label only."
USER_CH = ("Input state (JSON):\n{state}\n\nQuestion: {question}\n\nOptions:\n{options}\n\n"
           "Answer with the label of exactly one option (the string of + and - signs), nothing else.")
Q_ISING = "Which spin configuration is the ground state, i.e. minimises H(s)?"


def ising_state(h, J, flipped):
    """Same state dicts as e2_hamiltonian.ising_job (as stated) and e2b_convention.job (sign-flipped)."""
    if not flipped:
        return {"hamiltonian": "H(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j with spins s_i in {-1,+1}",
                "h": {f"h_{i}": float(h[i]) for i in range(len(h))},
                "J": {f"J_{i}{j}": v for (i, j), v in J.items()}}
    return {"hamiltonian": "H(s) = - sum_i h_i s_i - sum_{i<j} J_ij s_i s_j with spins s_i in {-1,+1}",
            "h": {f"h_{i}": float(-h[i]) for i in range(len(h))},
            "J": {f"J_{i}{j}": -v for (i, j), v in J.items()}}


class LM:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.dtype = DTYPE
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to("mps").eval()
        # Output projection in float32: bf16 logits of magnitude 16-32 are quantised in steps of 0.125, which would
        # quantise log-odds (and hence J asymmetries) at the scale of the 0.36-nat threshold. The transformer body
        # stays in DTYPE; only the final (post-norm) hidden state x lm_head is done in fp32.
        # (the fp32 copy is made chunk-wise on the fly, not stored: a resident 2.2 GB fp32 copy for 7B pushed the
        #  24 GB machine into swap on the 7k-token n=7 Ising prompts; bf16->fp32 conversion is exact, so the result
        #  is identical to a stored fp32 copy)
        self.Wb = self.model.lm_head.weight.detach()
        ids = {t: self.tok.encode(t, add_special_tokens=False) for t in ("Yes", " Yes", "No", " No")}
        assert all(len(v) == 1 for v in ids.values()), ids
        self.yn_ids = {t: v[0] for t, v in ids.items()}
        self.yn_tokens = {t: self.tok.convert_ids_to_tokens(v) for t, v in self.yn_ids.items()}

    def prompt(self, sys_msg, user):
        return self.tok.apply_chat_template([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True)

    def head(self, h):
        h = h.float()
        return torch.cat([h @ w.float().T for w in self.Wb.split(16384, 0)], -1)

    @torch.no_grad()
    def p_yes(self, state, statement):
        text = self.prompt(SYS_YN, USER_YN.format(state=json.dumps(state, indent=1), statement=statement))
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        logits = self.head(self.model.model(ids).last_hidden_state[0, -1])
        lp = torch.log_softmax(logits, -1)
        l = {t: logits[i].item() for t, i in self.yn_ids.items()}
        m = max(l.values()); e = {t: np.exp(v - m) for t, v in l.items()}
        py = (e["Yes"] + e[" Yes"]) / sum(e.values())
        mass = float(sum(np.exp(lp[i].item()) for i in self.yn_ids.values()))
        top = self.tok.convert_ids_to_tokens(int(logits.argmax()))
        return {"p": float(py), "mass": mass, "top": top, "n_prompt_tokens": ids.shape[1]}

    @torch.no_grad()
    def choice(self, state, question, crit):
        opts = "\n".join(f"{k}: {v}" for k, v in crit.items())
        text = self.prompt(SYS_CH, USER_CH.format(state=json.dumps(state, indent=1), question=question, options=opts))
        pids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        # Chunked prefill (CHUNK tokens at a time, same KV cache): n=7 prompts are ~7k tokens and a one-shot 7B
        # prefill materialises multi-GB attention buffers on MPS (it swapped and stalled on this 24 GB machine).
        cache = None
        for c0 in range(0, pids.shape[1], CHUNK):
            out = self.model.model(pids[:, c0:c0 + CHUNK], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        first = torch.log_softmax(self.head(out.last_hidden_state[0, -1]), -1)
        L = pids.shape[1]; scores = {}; ntok = {}; labs = {}
        for k in crit:
            full = self.tok(text + k, add_special_tokens=False).input_ids
            assert full[:L] == pids[0].tolist(), "label tokenisation crosses the prompt boundary"
            labs[k] = full[L:]; ntok[k] = len(labs[k]); scores[k] = first[labs[k][0]].item()
        # All continuation tokens in ONE forward over the prompt cache ("tree attention"): each label's tokens attend
        # to the whole prompt and to their own earlier tokens only, with positions L, L+1, ... This is exactly the
        # per-label conditional log-probability. (Per-label forwards with cache.crop stalled at 25-40 s each on the
        # 7k-token n=7 prompts under memory pressure.)
        seg = [(k, t) for k in crit for t in range(len(labs[k]) - 1)]
        if seg:
            T = len(seg); ids = torch.tensor([[labs[k][t] for k, t in seg]], device="mps")
            pos = torch.tensor([[L + t for _, t in seg]], device="mps")
            m = torch.zeros(T, L + T, dtype=torch.bool); m[:, :L] = True
            for a_, (ka, ta) in enumerate(seg):
                for b_, (kb, tb) in enumerate(seg):
                    if ka == kb and tb <= ta: m[a_, L + b_] = True
            o = self.model.model(ids, position_ids=pos, attention_mask=m[None, None].to("mps"), past_key_values=cache, use_cache=True)
            lps = torch.log_softmax(self.head(o.last_hidden_state[0]), -1)
            for a_, (k, t) in enumerate(seg): scores[k] += lps[a_, labs[k][t + 1]].item()
        del cache, out; torch.mps.empty_cache()
        v = np.array(list(scores.values())); p = np.exp(v - v.max()); p /= p.sum()
        return {"logp": scores, "p": dict(zip(scores, p.tolist())), "label_tokens": ntok, "n_prompt_tokens": L}


if __name__ == "__main__":
    name = sys.argv[1]; short = name.split("/")[-1] + ("_fp32" if DTYPE == torch.float32 else "")
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else ["e1", "neg", "ising"]
    rows = json.load(open("experiments/qc_labels.json"))
    insts = list(e2.ising_instances())    # single call -> identical instances to E2/E2b
    t0 = time.time(); lm = LM(name); t_load = time.time() - t0
    path = f"experiments/e7_raw_{short}.json"
    prev = json.load(open(path)) if os.path.exists(path) else {}   # sections are saved as they finish and merged
    out = prev | {"model": name, "yn_token_ids": lm.yn_ids, "yn_token_strings": lm.yn_tokens,
           "template": {"SYS_YN": SYS_YN, "USER_YN": USER_YN, "SYS_CH": SYS_CH, "USER_CH": USER_CH},
           "example_yn_prompt": lm.prompt(SYS_YN, USER_YN.format(state=json.dumps(state_of(rows[0], ("mr", True)), indent=1), statement=S["t1"])),
           "dtype": str(DTYPE), "lm_head_dtype": "float32", "choice_prefill_chunk": CHUNK, "torch": torch.__version__,
           "load_s": t_load, "runtime_s": prev.get("runtime_s", {})}
    save = lambda: json.dump(out, open(path, "w"), indent=1)
    try:
        import transformers; out["transformers"] = transformers.__version__
        from huggingface_hub import snapshot_download
        out["snapshot_path"] = snapshot_download(name, local_files_only=True)
    except Exception as ex:
        out["snapshot_err"] = repr(ex)
    if "e1" in only:
        t = time.time(); e1 = []
        for i, row in enumerate(rows):
            for cond in [None] + [(k, v) for k in S for v in (True, False)]:
                for q, txt in S.items():
                    if cond and q == cond[0]: continue
                    e1.append({"case": i, "cond": list(cond) if cond else None, "q": q, **lm.p_yes(state_of(row, cond), txt)})
            print("e1 case", i, round(time.time() - t), "s", flush=True)
        out["e1"] = e1; out["runtime_s"]["e1"] = time.time() - t; save()
    if "neg" in only:
        t = time.time()
        out["neg"] = [{"case": i, "q": k, **lm.p_yes(state_of(row), NEG[k])} for i, row in enumerate(rows) for k in S]
        out["runtime_s"]["neg"] = time.time() - t; save(); print("neg done", flush=True)
    if "ising" in only:
        t = time.time(); ising = out.get("ising_partial", [])     # resume: skip (form, n, k) already scored
        done = {(x["form"], x["n"], x["k"]) for x in ising}; acc0 = out["runtime_s"].get("ising_accum", 0.0)
        for flipped in (False, True):
            for n, k, h, J in insts:
                if ("flipped" if flipped else "stated", n, k) in done: continue
                confs = list(itertools.product([1, -1], repeat=n))
                crit = {e2.lab(s): "spins (" + ", ".join(f"s_{i}={s[i]:+d}" for i in range(n)) + ")" for s in confs}
                r = lm.choice(ising_state(h, J, flipped), Q_ISING, crit)
                ising.append({"form": "flipped" if flipped else "stated", "n": n, "k": k,
                              "E": {e2.lab(s): e2.energy(s, h, J) for s in confs},
                              "field": {e2.lab(s): float(sum(h[i] * s[i] for i in range(n))) for s in confs},
                              "coupling": {e2.lab(s): float(sum(v * s[i] * s[j] for (i, j), v in J.items())) for s in confs}, **r})
                out["ising_partial"] = ising; out["runtime_s"]["ising_accum"] = acc0 + time.time() - t; save(); torch.mps.empty_cache()
                print("ising", "flipped" if flipped else "stated", n, k, round(time.time() - t), "s", flush=True)
        out["ising"] = ising; out.pop("ising_partial", None)
        out["runtime_s"]["ising"] = acc0 + time.time() - t; out["runtime_s"].pop("ising_accum", None)
    save()
    print("done", {k: round(v) for k, v in out["runtime_s"].items()})
