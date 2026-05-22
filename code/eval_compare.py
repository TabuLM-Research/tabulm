#!/usr/bin/env python3
"""Paired per-item evaluation: TabuLM vs KinyaBERT on the same dev items.

Outputs:
  data/compare_results.json  — per-item predictions for both models
  Prints bootstrap 95% CI for both models
  Prints comparison examples (TabuLM correct, KinyaBERT wrong) for error analysis
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DATA_DIR   = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/data'
CODE_DIR   = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/code'
TABQA_FILE = os.path.join(DATA_DIR, 'tabqa_kin.json')
CSV_DIR    = os.path.join(DATA_DIR, 'tables')

TABULM_CKPT  = os.path.join(DATA_DIR, 'finetune_tabqa_v3_best.pt')
KINYABERT_CKPT = os.path.join(DATA_DIR, 'baseline_kinyabert_best.pt')

sys.path.insert(0, CODE_DIR)

import youtokentome as yttm
from morpho_data_loaders import KBVocab
from tabular_serializer import serialize_csv, table_cells_to_text, TableCell
from morpho_stub import parse_text_stub
from tabulm_model import TabuLM, tabulm_base


# ── Shared gold-cell lookup ────────────────────────────────────────────────────

def find_gold_cell(cells: List[TableCell], answer_text: str,
                   question_text: str = '') -> Optional[Tuple[int, int]]:
    answer_norm = answer_text.strip().lower()
    matches = [(c.row_id, c.col_id) for c in cells
               if c.row_id > 1 and c.col_id > 0
               and c.content.strip() == answer_text.strip()]
    if not matches:
        matches = [(c.row_id, c.col_id) for c in cells
                   if c.row_id > 1 and c.col_id > 0
                   and c.content.strip().lower() == answer_norm]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if question_text:
        q_words = set(question_text.lower().split())
        row_labels  = {c.row_id: c.content.strip().lower()
                       for c in cells if c.col_id == 1 and c.row_id > 1}
        col_headers = {c.col_id: c.content.strip().lower()
                       for c in cells if c.row_id == 1 and c.col_id > 0}
        best, best_score = None, (-1, -1)
        for (row_id, col_id) in matches:
            r_sc = len(q_words & set(row_labels.get(row_id,  '').split()))
            c_sc = len(q_words & set(col_headers.get(col_id, '').split()))
            if (r_sc, c_sc) > best_score:
                best_score, best = (r_sc, c_sc), (row_id, col_id)
        if best_score[0] > 0 or best_score[1] > 0:
            return best
    return min(matches, key=lambda rc: (rc[0], rc[1]))


def _predict_lookup(scores, valid_cells, cells, question_text):
    q_words = set(question_text.lower().split())
    row_labels  = {c.row_id: c.content.strip().lower()
                   for c in cells if c.col_id == 1 and c.row_id > 1}
    col_headers = {c.col_id: c.content.strip().lower()
                   for c in cells if c.row_id == 1 and c.col_id > 0}
    if not q_words:
        return valid_cells[scores.argmax().item()]
    row_score = {r: len(q_words & set(lbl.split())) for r, lbl in row_labels.items()}
    col_score = {c: len(q_words & set(hdr.split())) for c, hdr in col_headers.items()}
    best_row = max(row_score, key=row_score.get) if row_score else None
    best_col = max(col_score, key=col_score.get) if col_score else None
    has_row  = best_row is not None and row_score[best_row] > 0
    has_col  = best_col is not None and col_score[best_col] > 0
    if not has_row and not has_col:
        return valid_cells[scores.argmax().item()]
    row_indices = [i for i, rc in enumerate(valid_cells) if rc[0] == best_row] if has_row \
                  else list(range(len(valid_cells)))
    if not row_indices:
        return valid_cells[scores.argmax().item()]
    if has_col:
        both = [i for i in row_indices if valid_cells[i][1] == best_col]
        if both:
            return valid_cells[max(both, key=lambda i: scores[i].item())]
    return valid_cells[max(row_indices, key=lambda i: scores[i].item())]


# ── TabuLM inference ───────────────────────────────────────────────────────────

def load_tabulm(device):
    from finetune_tabqa import (detect_arch_from_state, encode_item, TabQAModel)
    bpe = yttm.BPE(model=os.path.join(DATA_DIR, 'BPE-30k.mdl'))
    kv  = KBVocab()
    kv.load_state_dict(torch.load(
        os.path.join(DATA_DIR, 'kb_vocab_state_dict_2021-02-07.pt'),
        map_location='cpu'))

    saved = torch.load(TABULM_CKPT, map_location='cpu')
    state = saved['model_state_dict']
    pretrain_state = {k: v for k, v in state.items()
                      if not k.startswith('cell_head')}
    if all(k.startswith('module.') for k in pretrain_state):
        pretrain_state = {k[7:]: v for k, v in pretrain_state.items()}

    args = detect_arch_from_state(pretrain_state)
    tabulm_obj = tabulm_base(kv, None, None, device, args, saved_model_file=None)
    model = TabQAModel(tabulm_obj).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    return model, args, kv, bpe, encode_item


def tabulm_predict(model, args, kv, bpe, encode_item, cells, question, atype, device):
    from finetune_tabqa import encode_item as _enc
    enc = _enc(cells, question, kv, bpe)
    if enc is None:
        return None
    (pos_tags, stems, affixes, tokens_lengths,
     row_ids, col_ids, cell_types, ordered_cells, cell_to_positions) = enc
    with torch.no_grad():
        scores, valid_cells = model(
            args, pos_tags, stems, affixes,
            tokens_lengths, row_ids, col_ids, cell_types,
            ordered_cells, cell_to_positions, device)
    if scores is None:
        return None
    if atype == 'lookup':
        return _predict_lookup(scores, valid_cells, cells, question)
    return valid_cells[scores.argmax().item()]


# ── KinyaBERT inference ────────────────────────────────────────────────────────

class KBCellScorer(nn.Module):
    def __init__(self, encoder, d_model):
        super().__init__()
        self.encoder   = encoder
        self.cell_head = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask, ordered_cells,
                cell_to_positions, device):
        out    = self.encoder(
            input_ids=input_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device))
        hidden = out.last_hidden_state[0]
        S      = hidden.size(0)
        cell_embeds, valid_cells = [], []
        for rc in ordered_cells:
            positions = [p for p in cell_to_positions[rc] if p < S]
            if not positions:
                continue
            cell_embeds.append(hidden[positions].mean(0))
            valid_cells.append(rc)
        if not cell_embeds:
            return None, []
        scores = self.cell_head(torch.stack(cell_embeds)).squeeze(-1)
        return scores, valid_cells


def load_kinyabert(device):
    from eval_baselines import serialize_table_for_bert
    hf_name   = 'jean-paul/KinyaBERT-large'
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    encoder   = AutoModel.from_pretrained(hf_name)
    d_model   = encoder.config.hidden_size
    model     = KBCellScorer(encoder, d_model).to(device)
    saved     = torch.load(KINYABERT_CKPT, map_location='cpu')
    model.load_state_dict(saved['model_state_dict'], strict=False)
    model.eval()
    return model, tokenizer, serialize_table_for_bert


def kb_predict(model, tokenizer, serialize_fn, cells, question, atype, device):
    enc = serialize_fn(cells, question, tokenizer)
    if enc is None:
        return None
    input_ids, attn_mask, ordered_cells, cell_to_positions = enc
    with torch.no_grad():
        scores, valid_cells = model(
            input_ids, attn_mask, ordered_cells, cell_to_positions, device)
    if scores is None:
        return None
    if atype == 'lookup':
        return _predict_lookup(scores, valid_cells, cells, question)
    return valid_cells[scores.argmax().item()]


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(hits: List[int], n_boot: int = 10000, alpha: float = 0.05):
    hits_arr = np.array(hits, dtype=float)
    n = len(hits_arr)
    boot_means = [np.mean(np.random.choice(hits_arr, size=n, replace=True))
                  for _ in range(n_boot)]
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(hits_arr)), lo, hi


def paired_bootstrap_pvalue(hits_a: List[int], hits_b: List[int],
                             n_boot: int = 10000) -> float:
    """One-sided test: P(A > B) under H0 that A and B have same mean."""
    a = np.array(hits_a, dtype=float)
    b = np.array(hits_b, dtype=float)
    n = len(a)
    assert len(b) == n
    obs_diff = np.mean(a) - np.mean(b)
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.array(diffs)
    # two-sided p-value
    p = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))
    return p


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--out', default=os.path.join(DATA_DIR, 'compare_results.json'))
    args_cli = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[CMP] Device: {device}')

    print('[CMP] Loading TabuLM...')
    tabulm_model, tabulm_args, kv, bpe, enc_fn = load_tabulm(device)
    print('[CMP] Loading KinyaBERT...')
    kb_model, kb_tok, kb_ser = load_kinyabert(device)

    with open(TABQA_FILE) as f:
        all_items = json.load(f)
    random.shuffle(all_items)
    dev_items = all_items[int(0.8 * len(all_items)):]
    print(f'[CMP] {len(dev_items)} dev items')

    records = []
    for item in dev_items:
        csv_path = os.path.join(CSV_DIR, item['table_file'])
        atype    = item.get('answer_type', '?')
        if atype == 'count' or not os.path.exists(csv_path):
            continue

        cells = serialize_csv(csv_path)
        if not cells:
            continue

        gold_rc = find_gold_cell(cells, item['answer'], item['question'])
        if gold_rc is None:
            continue

        # TabuLM prediction
        t_pred = tabulm_predict(tabulm_model, tabulm_args, kv, bpe, enc_fn,
                                cells, item['question'], atype, device)
        t_hit  = int(t_pred == gold_rc) if t_pred is not None else None

        # KinyaBERT prediction
        k_pred = kb_predict(kb_model, kb_tok, kb_ser,
                            cells, item['question'], atype, device)
        k_hit  = int(k_pred == gold_rc) if k_pred is not None else None

        # only count items both models can evaluate
        if t_hit is None or k_hit is None:
            continue

        # get cell text for readable output
        cell_map = {(c.row_id, c.col_id): c.content.strip() for c in cells}
        records.append({
            'question':    item['question'],
            'table':       item['table_file'],
            'answer_type': atype,
            'gold':        item['answer'],
            'gold_rc':     list(gold_rc),
            'tabulm_rc':   list(t_pred),
            'tabulm_pred': cell_map.get(t_pred, '?'),
            'tabulm_hit':  t_hit,
            'kb_rc':       list(k_pred),
            'kb_pred':     cell_map.get(k_pred, '?'),
            'kb_hit':      k_hit,
        })

    print(f'\n[CMP] Paired items: {len(records)}')

    # ── Per-type breakdown ──────────────────────────────────────────────────
    by_type: Dict[str, Dict] = {}
    for r in records:
        t = r['answer_type']
        by_type.setdefault(t, {'tabulm': [], 'kb': []})
        by_type[t]['tabulm'].append(r['tabulm_hit'])
        by_type[t]['kb'].append(r['kb_hit'])

    print(f'\n{"Type":<14} {"N":>4}  {"TabuLM":>8}  {"KinyaBERT":>10}')
    print('-' * 44)
    all_t, all_k = [], []
    for t, d in sorted(by_type.items()):
        n = len(d['tabulm'])
        t_em = sum(d['tabulm']) / n
        k_em = sum(d['kb']) / n
        all_t.extend(d['tabulm']); all_k.extend(d['kb'])
        print(f'{t:<14} {n:>4}  {t_em:>8.3f}  {k_em:>10.3f}')
    n_all = len(all_t)
    print(f'{"Overall":<14} {n_all:>4}  {sum(all_t)/n_all:>8.3f}  {sum(all_k)/n_all:>10.3f}')

    # ── Bootstrap CI ───────────────────────────────────────────────────────
    print(f'\n[CMP] Bootstrap CI (n_boot={args_cli.n_boot}):')
    t_em, t_lo, t_hi = bootstrap_ci(all_t, args_cli.n_boot)
    k_em, k_lo, k_hi = bootstrap_ci(all_k, args_cli.n_boot)
    p_val = paired_bootstrap_pvalue(all_t, all_k, args_cli.n_boot)
    print(f'  TabuLM:    {t_em:.3f}  95% CI [{t_lo:.3f}, {t_hi:.3f}]')
    print(f'  KinyaBERT: {k_em:.3f}  95% CI [{k_lo:.3f}, {k_hi:.3f}]')
    print(f'  Two-sided paired bootstrap p-value: {p_val:.4f}')

    # ── Error analysis: TabuLM correct, KinyaBERT wrong ────────────────────
    wins = [r for r in records if r['tabulm_hit'] == 1 and r['kb_hit'] == 0]
    losses = [r for r in records if r['tabulm_hit'] == 0 and r['kb_hit'] == 1]
    print(f'\n[CMP] TabuLM wins (T=1, KB=0): {len(wins)}')
    print(f'[CMP] TabuLM losses (T=0, KB=1): {len(losses)}')

    print('\n=== TabuLM WINS (comparison type preferred) ===')
    cmp_wins = [r for r in wins if r['answer_type'] == 'comparison']
    for r in cmp_wins[:5]:
        print(f'\n  Table : {r["table"]}')
        print(f'  Q     : {r["question"]}')
        print(f'  Type  : {r["answer_type"]}')
        print(f'  Gold  : {r["gold"]}  (row={r["gold_rc"][0]}, col={r["gold_rc"][1]})')
        print(f'  TabuLM: {r["tabulm_pred"]}  ✓')
        print(f'  KinyaB: {r["kb_pred"]}  ✗')

    print('\n=== TabuLM LOSSES ===')
    for r in losses[:3]:
        print(f'\n  Table : {r["table"]}')
        print(f'  Q     : {r["question"]}')
        print(f'  Type  : {r["answer_type"]}')
        print(f'  Gold  : {r["gold"]}')
        print(f'  TabuLM: {r["tabulm_pred"]}  ✗')
        print(f'  KinyaB: {r["kb_pred"]}  ✓')

    # ── Save ───────────────────────────────────────────────────────────────
    out_data = {
        'n_paired':       n_all,
        'tabulm_em':      round(t_em, 4),
        'kb_em':          round(k_em, 4),
        'tabulm_ci':      [round(t_lo, 4), round(t_hi, 4)],
        'kb_ci':          [round(k_lo, 4), round(k_hi, 4)],
        'p_value':        round(p_val, 4),
        'by_type':        {t: {
                               'n':           len(d['tabulm']),
                               'tabulm_em':   round(sum(d['tabulm'])/len(d['tabulm']), 4),
                               'kb_em':       round(sum(d['kb'])/len(d['kb']), 4),
                           } for t, d in by_type.items()},
        'tabulm_wins':    wins,
        'tabulm_losses':  losses,
        'records':        records,
    }
    with open(args_cli.out, 'w') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f'\n[CMP] Saved to {args_cli.out}')


if __name__ == '__main__':
    main()
