# Verification log (2026-09-23)

Method. For DOI entries we fetched `https://api.crossref.org/works/<doi>` and compared authors, title, year, venue, volume, issue and pages. For arXiv entries we used the export API (`https://export.arxiv.org/api/query?id_list=...`), checking title, authors and first-version date. Other entries were checked against the publisher or host page. Abstracts were read (arXiv API, CrossRef, OpenAlex or Semantic Scholar) wherever the survey states a finding. Result: 57 entries verified, 0 in the bib unverified. Named items dropped or changed are listed at the end.

## Verified entries

| key | verified against | notes |
|---|---|---|
| lecun2006tutorial | http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf (title page) | Chapter in *Predicting Structured Data*, MIT Press 2006; no DOI |
| grathwohl2019your | https://arxiv.org/abs/1912.03263 | Cited as arXiv; ICLR 2020 venue not independently verified (OpenReview blocked), so omitted |
| hopfield1982neural | https://api.crossref.org/works/10.1073/pnas.79.8.2554 | |
| ackley1985learning | https://api.crossref.org/works/10.1207/s15516709cog0901_7 | |
| ramsauer2020hopfield | https://arxiv.org/abs/2008.02217 | Cited as arXiv; ICLR 2021 venue not independently verified |
| du2019implicit | https://arxiv.org/abs/1903.08689 | Added (not in brief); cited as arXiv |
| winer2025deep | https://arxiv.org/abs/2503.23982 | Confirmed to exist: "Deep Neural Nets as Hamiltonians", Winer & Hanin, 2025-03-31 |
| besag1974spatial | https://api.crossref.org/works/10.1111/j.2517-6161.1974.tb00999.x | CrossRef pages 192–225 (the discussion runs to 236; we use the CrossRef range) |
| hammersley1971markov | https://ora.ox.ac.uk/objects/uuid:4ea849da-1511-4578-bb88-6a8d02f457a6 ; scan at https://www.statslab.cam.ac.uk/~grg1000/books/hammfest/hamm-cliff.pdf | Title and authors read from the scan's first page; the scan carries no date, and the 1971 date comes from the Oxford ORA record |
| arnold1989compatible | https://api.crossref.org/works/10.1080/01621459.1989.10478750 | |
| heckerman2000dependency | https://www.jmlr.org/papers/v1/heckerman00a.html | JMLR 1(Oct):49–75, 2000 |
| fluri2023evaluating | https://arxiv.org/abs/2306.09983 | Found by us (brief asked for real LLM-coherence papers) |
| paleka2025consistency | https://arxiv.org/abs/2412.18544 ; OpenReview API search (forum r5IXBlTCGc, venue "ICLR 2025 Oral") | |
| calanzone2024towards | https://arxiv.org/abs/2404.12843 | Workshop venue taken from the arXiv comment |
| zhu2024incoherent | https://arxiv.org/abs/2401.16646 | CogSci 2024 per the arXiv journal_ref |
| kassner2021beliefbank | https://aclanthology.org/2021.emnlp-main.697/ | pp. 8849–8861 |
| guo2017calibration | https://proceedings.mlr.press/v70/guo17a.html | PMLR 70:1321–1330 |
| kadavath2022language | https://arxiv.org/abs/2207.05221 | Full 36-author list from arXiv |
| brier1950verification | https://api.crossref.org/works/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2 | |
| gneiting2007strictly | https://api.crossref.org/works/10.1198/016214506000001437 | |
| naeini2015obtaining | https://api.crossref.org/works/10.1609/aaai.v29i1.9602 | AAAI 2015, vol. 29(1) |
| tian2023just | https://aclanthology.org/2023.emnlp-main.330/ | pp. 5433–5442 |
| lucas2014ising | https://api.crossref.org/works/10.3389/fphy.2014.00005 | Article 5 |
| kadowaki1998quantum | https://api.crossref.org/works/10.1103/PhysRevE.58.5355 | |
| anshu2021sample | https://api.crossref.org/works?query.bibliographic=Sample-efficient+learning+of+interacting+quantum+systems -> 10.1038/s41567-021-01232-0 | **Corrected**: the brief's title "...quantum many-body systems" is the FOCS 2020 version (10.1109/focs46700.2020.00069); the Nature Physics 2021 article is "Sample-efficient learning of interacting quantum systems", 17:931–935. The DOI we first guessed (10.1038/s41567-020-01086-6) returned 404 |
| nguyen2017inverse | https://api.crossref.org/works/10.1080/00018732.2017.1341604 | |
| wiebe2014hamiltonian | https://api.crossref.org/works/10.1103/PhysRevLett.112.190501 | Added |
| peruzzo2014variational | https://api.crossref.org/works/10.1038/ncomms5213 | |
| mcardle2020quantum | https://api.crossref.org/works/10.1103/RevModPhys.92.015003 | |
| cao2019quantum | https://api.crossref.org/works/10.1021/acs.chemrev.8b00803 | |
| jordan1928pauli | https://api.crossref.org/works/10.1007/BF01331938 | Z. Phys. 47:631–651 |
| omalley2016scalable | https://api.crossref.org/works/10.1103/PhysRevX.6.031007 | Full 32-author list from CrossRef |
| carleo2017solving | https://api.crossref.org/works/10.1126/science.aag2302 | |
| pfau2020ab | https://api.crossref.org/works/10.1103/PhysRevResearch.2.033429 | FermiNet |
| hermann2020deep | https://api.crossref.org/works/10.1038/s41557-020-0544-y | PauliNet |
| xia2018electronic | https://api.crossref.org/works/10.1021/acs.jpcb.7b10371 | Title is "Electronic Structure Calculations and the Ising Hamiltonian" |
| genin2019quantum | https://arxiv.org/abs/1901.04715 | No journal_ref on arXiv; cited as a preprint |
| lee1989diagnostic | https://api.crossref.org/works/10.1002/qua.560360824 ; Wiley landing page (via search) lists 1989 | CrossRef "issued" shows 2009 (the online digitisation date); the volume 36/S23 print year is 1989 |
| jiang2012multireference | https://api.crossref.org/works?query.bibliographic=... -> 10.1021/ct2006852 ; abstract via OpenAlex | **Corrected**: the DOI we first guessed (10.1021/ct200418x) returned 404. This is a JCTC research article, not a review as the brief suggested |
| sayfutyarova2017automated | https://api.crossref.org/works/10.1021/acs.jctc.7b00128 | AVAS |
| jeong2020automation | https://api.crossref.org/works/10.1021/acs.jctc.9b01297 | |
| duan2020data | https://api.crossref.org/works/10.1021/acs.jctc.0c00358 | "Duan & Kulik 2020" item (1 of 2) |
| liu2020rapid | https://api.crossref.org/works/10.1021/acs.jpclett.0c02288 | **Note**: first author is Fang Liu (Liu, Duan, Kulik), so the key was changed from duan2020* |
| janet2017predicting | https://api.crossref.org/works/10.1039/C7SC01247K | Added for spin-state prediction |
| huron1973iterative | https://api.crossref.org/works/10.1063/1.1679199 | CIPSI |
| holmes2016heat | https://api.crossref.org/works/10.1021/acs.jctc.6b00407 | |
| coe2018machine | https://api.crossref.org/works/10.1021/acs.jctc.8b00849 ; abstract via Semantic Scholar | |
| bran2024augmenting | https://api.crossref.org/works/10.1038/s42256-024-00832-8 | ChemCrow |
| boiko2023autonomous | https://api.crossref.org/works/10.1038/s41586-023-06792-0 | Coscientist |
| mirza2025framework | https://api.crossref.org/works/10.1038/s41557-025-01815-x | ChemBench, Nat. Chem. 17:1027–1034 (35 authors; bib truncated with "and others") |
| mirzadeh2024gsm | https://arxiv.org/abs/2410.05229 | Numeric-reasoning limits; cited as arXiv |
| typesafe2026introducing | https://typesafe.ai/blog/introducing-system-one-models-and-jev (HTTP 200; title "Introducing System One Models & Jev") | Author Diogo Almeida, dated 2026-09-15 |
| typesafe2026api | https://docs.typesafe.ai/api (HTTP 200; title "API reference") | Noul/Choice(≤255)/Score(2–10); jev-1.13.0 appears in example responses |
| typesafe2026systemone | https://docs.typesafe.ai/concepts/system-one (HTTP 200) | Added |
| typesafe2026jaggedness | https://docs.typesafe.ai/model-jaggedness/jev-1.13 (HTTP 200; title "Jev 1.13 jaggedness") | Added: the "Common-sense structural invariants" section (negation pair 0.72 + 0.47 = 1.19) is central to our framing |
| willison2026jev | https://simonwillison.net/2026/Sep/21/jev/ (HTTP 200) | Dated 21 Sep 2026 |
| fernholz2026new | https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/ (HTTP 200) | By Tim Fernholz, 2026-09-18 |

## Dropped or changed relative to the brief

- **"Duan & Kulik ML for multireference character (2020)"**: no paper has exactly those two authors. We cite the two 2020 Kulik-group papers that match (duan2020data; liu2020rapid, whose first author is Liu).
- **"Anshu et al. 2021 sample-efficient learning of quantum many-body systems"**: the title differs between the conference and journal versions. We cite the Nature Physics 2021 version under its actual title.
- **"Jiang/Wilson multireference diagnostics review (2012 JCTC)"**: this is a research article, not a review. It is kept.
- **Venue claims for ICLR papers (Grathwohl 2020, Ramsauer 2021) and NeurIPS (Du & Mordatch 2019)**: OpenReview and DBLP were unreachable (challenge / 429), so these are cited as arXiv preprints. No venue is asserted.
- **Hammersley–Clifford**: this is an unpublished manuscript. It is kept, with the Oxford ORA record as the source for its 1971 date.
- **Discrepancy to flag in the paper**: the training method is called "Reinforcement Learning *for* Calibrated Decisions (RLCD)" in the vendor blog and "reinforcement learning *from* calibrated decisions" in TechCrunch. Quote the vendor form.
- Nothing named in the brief turned out to be fabricated. arXiv 2503.23982 is real.

## Added in revision (2026-09-24)
- sun2020recent: CrossRef 10.1063/5.0006074 (J. Chem. Phys. 153(2), article 024109, 2020)
- mcclean2020openfermion: CrossRef 10.1088/2058-9565/ab8ebc (Quantum Sci. Technol. 5(3) 034014, 2020)
- benjamini1995controlling: CrossRef 10.1111/j.2517-6161.1995.tb02031.x (JRSS B 57(1) 289-300)
- lin2007note: CrossRef 10.1007/s10994-007-5018-6 (Machine Learning 68(3) 267-276). Cited for Platt scaling because Platt's 1999 chapter could not be verified via CrossRef.
- qwen2024qwen25: arXiv 2412.15115 (export.arxiv.org API, published 2024-12-19)
