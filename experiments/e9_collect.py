"""Controls requested in review round 1 (added after review; reported as such).
E9a positive control: a population with stated counts over three binary attributes; the exact joint is computable,
    so a coherent reader gives J_ab = J_ba. Same conditioning phrasing as E1 ("Established fact ...").
E9b fact uptake: state s_k as TRUE/FALSE and ask s_k itself.
E9c/E9d E3 ablations: name + geometry without the stretch note; stretch note + geometry without the name."""
import itertools, json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, "harness"); sys.path.insert(0, "experiments")
from jev import ask, noul
from e1e3_collect import S, FACT, state_of

rows = json.load(open("experiments/qc_labels.json"))
rng = np.random.default_rng(99)
ATTR = {"A": "The selected person is a smoker.", "B": "The selected person has high blood pressure.", "C": "The selected person exercises weekly."}
def tables():
    for t in range(24):
        w = rng.dirichlet(np.ones(8) * 0.7); counts = np.round(w * 1000).astype(int)
        yield t, {"".join(c): int(n) for c, n in zip(("".join(map(str, s)) for s in itertools.product([1, 0], repeat=3)), counts)}
def pop_state(counts, fact=None):
    desc = [f"{n} people: smoker={'yes' if k[0]=='1' else 'no'}, high blood pressure={'yes' if k[1]=='1' else 'no'}, exercises weekly={'yes' if k[2]=='1' else 'no'}"
            for k, n in counts.items()]
    st = {"population": "A town has exactly the following 1000 residents (counts by attribute combination).", "counts": desc,
          "selection": "One resident is selected uniformly at random."}
    if fact: st["known"] = FACT[fact[1]].format(ATTR[fact[0]])
    return st

def jobs():
    for t, c in tables():
        for r in range(3):
            yield ("e9a", t, r, None, pop_state(c), {k: noul(v) for k, v in ATTR.items()}, c)
            for k in ATTR:
                for v in (True, False):
                    yield ("e9a", t, r, (k, v), pop_state(c, (k, v)), {q: noul(x) for q, x in ATTR.items() if q != k}, c)
    for i, row in enumerate(rows):
        for k in S:
            for v in (True, False):
                yield ("e9b", i, 0, (k, v), state_of(row, (k, v)), {k: noul(S[k])}, None)
        for r in range(3):
            st = state_of(row); st.pop("geometry_note")
            yield ("e9c", i, r, None, st, {k: noul(S[k]) for k in ("mr", "t1", "unstable")}, None)
            st = state_of(row); st["system"] = "a small molecule"
            yield ("e9d", i, r, None, st, {k: noul(S[k]) for k in ("mr", "t1", "unstable")}, None)

def run(j):
    kind, i, r, cond, st, qs, c = j
    a = ask(st, qs, rep=r)["answers"]
    return {"kind": kind, "case": i, "rep": r, "cond": cond, "p": {k: v["noul"] for k, v in a.items()}, "counts": c}

if __name__ == "__main__":
    J = list(jobs()); print("jobs", len(J), flush=True)
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(run, J))
    json.dump(out, open("experiments/e9_raw.json", "w")); print("done", len(out))
