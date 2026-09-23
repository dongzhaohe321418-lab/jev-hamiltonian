# Is there a Hamiltonian inside a decision model?

Code, data and paper for a black-box study of **Jev** (TypeSafe AI, `jev-1.13.0`), a "System One" model that returns
calibrated probabilities over a typed answer schema instead of text. Any such output can be read as local energies
E = -log p. We ask whether those energies form one Hamiltonian-like landscape and whether they are useful in quantum
chemistry.

The short answer: the local energies are approximately but measurably incoherent, they do not follow a Hamiltonian that
is written into the input (Jev reads coefficient signs and recalls textbook answers), and for chemistry decisions most of
the signal comes from a stretch factor stated in the prompt. Jev adds a small amount of information on top of a
two-descriptor model for some labels, and its determinant rankings for selected CI are a fixed template that loses to
iterative CIPSI.

The paper is `paper/main.pdf` (arXiv category cs.LG, cross-lists physics.chem-ph and quant-ph).

## Layout

| Path | What it holds |
|---|---|
| `harness/jev.py` | API client; every request and response is cached under `data/cache/` (one JSON per call) |
| `experiments/PLAN.md` | Pre-registered plan, amendments, and the disclosed deviations |
| `experiments/qc_labels_v2.py` | PySCF ground truth (stable RHF, full-valence singlet CASCI, frozen-core CCSD, multi-start UHF) |
| `experiments/e*_collect.py`, `e*_*.py` | Data collection for each experiment (E0-E9) |
| `experiments/analyze_*.py` | Offline analyses that produce every number in the paper |
| `experiments/figures.py` | All data figures |
| `review/` | Pre-release review: three blind referee reports simulated by separate AI agents and a code audit by a second AI system (Codex); no human peer review |
| `survey/` | Literature survey, verified bibliography and verification log |

## Reproducing

```bash
./reproduce.sh            # rebuilds analyses, figures and the PDF from the cache
```

`reproduce.sh` sets `JEV_OFFLINE=1`, so a missing cache entry raises an error instead of calling the API. Collecting new
data needs an API key in `~/.config/typesafe/key` or `TYPESAFE_API_KEY`. The API only exposes the alias `jev-latest`;
every cached response records the concrete model version it came from (`jev-1.13.0` for all data here).

The study made 10,145 distinct API calls (all answered by `jev-1.13.0`) at a total cost of about US$0.28.

## Licence

Code: MIT. Cached API responses are released for reproducibility of this study.
