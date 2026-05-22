# TabuLM: Morphology-Aware Tabular Pre-training for Low-Resource Languages

> **Paper under review — EMNLP 2026 ARR May cycle.**

**TabuLM** is the first pre-trained language model that jointly captures
morphological richness and tabular relational structure for a low-resource
language. Built on top of [KinyaBERT](https://github.com/anzeyimana/KinyaBERT),
it introduces three tabular-aware additions to the sequence transformer:

| Component | What it does |
|---|---|
| Row / Col / CellType Embeddings | Additive embeddings encoding each token's grid position and cell type |
| Table-Structure Attention Bias | Learned per-head scalars boosting same-row, same-column, and header attention |
| MCR + CTP objectives | Masked Cell Recovery and Column Type Prediction pre-training tasks |

We also release **TabQA-kin**, the first native Kinyarwanda table
question-answering benchmark (526 QA pairs across 31 government tables).

---

## Results on TabQA-kin (dev set)

| Model | Lookup | Comparison | Aggregation | Overall EM |
|---|:-:|:-:|:-:|:-:|
| GPT-4o (zero-shot) | 82.9 | 79.2 | 25.9 | 64.0 |
| GPT-4o-mini (zero-shot) | 85.7 | 70.8 | 29.6 | 64.0 |
| mBERT (fine-tuned) | 16.7 | 50.0 | 80.8 | 49.3 |
| XLM-R (fine-tuned) | 19.2 | 44.4 | 85.2 | 50.0 |
| KinyaBERT-large (fine-tuned) | 26.7 | 59.1 | 88.9 | 56.3 |
| **TabuLM (ours)** | **28.6** | **66.7** | **79.2** | **62.0** |

**Key finding:** GPT-4o and GPT-4o-mini both score 64.0% — a scale-independent
LLM ceiling driven by aggregation failure (25–30%). All fine-tuned models
break through this ceiling on aggregation (non-overlapping 95% Wilson CIs,
statistically significant).

---

## Repository Structure

```
TabuLM/
├── code/
│   ├── train_tabulm.py            # Pre-training (distributed, LAMB optimizer)
│   ├── finetune_tabqa.py          # Fine-tune + evaluate on TabQA-kin
│   ├── eval_baselines.py          # mBERT / XLM-R / KinyaBERT baselines
│   ├── eval_llm_baseline.py       # GPT-4o / GPT-4o-mini zero-shot eval
│   └── eval_llm_agg_fewshot.py   # 3-shot aggregation experiment
├── data/
│   ├── tables/                    # 172 Kinyarwanda pre-training tables (CSV)
│   └── tabqa_kin.json             # TabQA-kin benchmark (526 QA pairs)
├── results/
│   ├── finetune_tabqa_v3_results.json
│   ├── baseline_results.json
│   ├── llm_baseline_v2_results.json
│   └── llm_baseline_mini_results.json
└── paper/
    ├── tabulm_emnlp2026.tex
    └── tabulm_refs.bib
```

---

## Setup

```bash
git clone https://github.com/TabuLM-Research/TabuLM.git
cd TabuLM

conda create -n tabulm python=3.9
conda activate tabulm
pip install torch transformers youtokentome openai tqdm scipy
```

> **Morphological analyzer note:** Tier 1 morphological analysis requires
> `libkinlp.so` from the
> [KinyaBERT repository](https://github.com/anzeyimana/KinyaBERT).
> Domain tokens (numerals, entity names) fall back to BPE automatically —
> all tabular training and evaluation runs on the BPE fallback path without
> the binary.

---

## Pre-training

```bash
CUDA_VISIBLE_DEVICES=0 python code/train_tabulm.py \
    -g 1 \
    --batch-size 8 \
    --accumulation-steps 8 \
    --number-of-load-batches 24 \
    --num-iters 10000 \
    --warmup-iter 500 \
    --seq-tr-nhead 8
```

Pre-training takes ~7 hours on a single NVIDIA RTX 3090 (24 GB),
warm-started from a KinyaBERT checkpoint.

---

## Fine-tuning on TabQA-kin

```bash
python code/finetune_tabqa.py \
    --checkpoint data/tabulm_model_pretrained.pt \
    --output-prefix data/finetune_tabqa
```

Fine-tuning runs for 20 epochs with AdamW lr=2e-5, top-4 layers unfrozen.
Converges in under 30 minutes on a single GPU.

---

## Evaluating Baselines

```bash
# Fine-tuned text models
python code/eval_baselines.py --model mbert
python code/eval_baselines.py --model xlmr
python code/eval_baselines.py --model kinyabert

# Zero-shot LLM baselines (bring your own key)
python code/eval_llm_baseline.py --provider openai --api-key YOUR_KEY
python code/eval_llm_baseline.py --provider openai --model gpt-4o-mini --api-key YOUR_KEY
```

---

## Pre-trained Checkpoints

Checkpoints will be released on
[Hugging Face](https://huggingface.co/TabuLM-Research) upon paper acceptance.

| File | Size | Description |
|---|---|---|
| `tabulm_pretrained.pt` | 751 MB | Pre-trained TabuLM encoder (10K iters) |
| `tabulm_tabqa_finetuned.pt` | 751 MB | Fine-tuned on TabQA-kin (best dev EM 62.0%) |

---

## Citation

```bibtex
@inproceedings{tabulm2026emnlp,
  title     = {TabuLM: Morphology-Aware Tabular Pre-training for Low-Resource Languages},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

- **Code:** MIT License
- **Pre-training data:** Sourced from Rwanda government open-data portals (public domain)
- **TabQA-kin benchmark:** CC BY 4.0
