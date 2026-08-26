# TabuLM: Morphology-Aware Tabular Pre-training for Low-Resource Languages

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20%26%20Data-yellow)](https://huggingface.co/TabuLM-Research/tabulm)

**Ireddi Rakshitha** (Software Engineer, Barclays) · **Devavarapu Yashwanth** (Software Engineer, Barclays) · **Pierre Ntakirutimana** (Research Associate, Carnegie Mellon University)

> **arXiv preprint · August 2026**

---

## Overview

**TabuLM** is the first language model pre-trained on Kinyarwanda tabular data. It extends KinyaBERT-large with:

- **Row, column, and cell-type embeddings** — additive structural signal at every Tier 2 layer
- **Table-structure attention bias** — learned per-head scalars for same-row, same-column, and header signals
- **Masked Cell Recovery (MCR)** — masks entire cells; model reconstructs from row/column context
- **Column Type Prediction (CTP)** — masks column headers; model predicts numeric/text/categorical/date type

Pre-training corpus: **172 Rwandan government tables** (~35,000 cells) from NISR, RAB, REB, MoH open-data portals.

**TabQA-kin** — the first native Kinyarwanda table QA benchmark: 526 QA pairs, 31 tables, 4 question types.

---

## Results on TabQA-kin

| Model | Overall EM | Lookup | Comparison | Aggregation |
|---|---|---|---|---|
| GPT-4o (zero-shot) | 64.0% | **82.9%** | **79.2%** | 25.9% |
| GPT-4o-mini (zero-shot) | 64.0% | 85.7% | 70.8% | 29.6% |
| mBERT | 49.3% | 16.7% | 50.0% | 80.8% |
| XLM-R | 50.0% | 19.2% | 44.4% | 85.2% |
| KinyaBERT-large | 56.3% | 26.7% | 59.1% | **88.9%** |
| **TabuLM (ours)** | **62.0%** | **28.6%** | **66.7%** | 79.2% |

All fine-tuned models substantially outperform GPT-4o on aggregation — fine-tuning closes a structural gap zero-shot LLMs cannot overcome.

---

## Repository Contents

```
tabulm-arxiv/
├── main.tex            ← arXiv-ready LaTeX source
├── tabulm_refs.bib     ← BibTeX references
├── splncs04.bst        ← bibliography style
├── llncs.cls           ← required by splncs04.bst
└── README.md
```

## Compile

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Model & Data

Checkpoint (751 MB), TabQA-kin benchmark, and pre-training tables:

**[huggingface.co/TabuLM-Research/tabulm](https://huggingface.co/TabuLM-Research/tabulm)**

---

## Citation

```bibtex
@article{ireddi2026tabulm,
  title   = {{TabuLM}: Morphology-Aware Tabular Pre-training for Low-Resource Languages},
  author  = {Ireddi, Rakshitha and Devavarapu, Yashwanth and Ntakirutimana, Pierre},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code: MIT · TabQA-kin: CC-BY 4.0 · Pre-training tables: Rwandan government open-data (public domain)
