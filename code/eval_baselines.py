#!/usr/bin/env python3
"""Baseline evaluation on TabQA-kin using mBERT, XLM-R, and KinyaBERT.

Each baseline uses the same cell-selection approach as TabuLM:
- Serialize table as flat text, track (row, col) token positions
- Pool hidden states per cell → linear cell scorer
- Same 80/20 train/dev split (seed=42), 20 epochs, lr=2e-5
- Gold cell found by text-matching (same as finetune_tabqa.py)
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel

DATA_DIR = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/data'
CODE_DIR = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/code'
TABQA_FILE = os.path.join(DATA_DIR, 'tabqa_kin.json')
CSV_DIR    = os.path.join(DATA_DIR, 'tables')

sys.path.insert(0, CODE_DIR)
from tabular_serializer import serialize_csv, TableCell

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    'mbert':     'bert-base-multilingual-cased',
    'xlmr':      'xlm-roberta-base',
    'kinyabert': 'jean-paul/KinyaBERT-large',
}


# ── Gold cell lookup (same as finetune_tabqa.py) ───────────────────────────────

def find_gold_cell(cells: List[TableCell], answer_text: str,
                   question_text: str = '') -> Optional[Tuple[int, int]]:
    answer_norm = answer_text.strip().lower()
    matches = [(c.row_id, c.col_id) for c in cells
               if c.row_id > 1 and c.col_id > 0 and c.content.strip() == answer_text.strip()]
    if not matches:
        matches = [(c.row_id, c.col_id) for c in cells
                   if c.row_id > 1 and c.col_id > 0 and c.content.strip().lower() == answer_norm]
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


# ── Table serialization for HF tokenizers ─────────────────────────────────────

def serialize_table_for_bert(cells: List[TableCell], question: str,
                              tokenizer, max_seq_len: int = 512):
    """
    Linearize table + question for a BERT-style model.
    Format: [CLS] h1 | h2 | ... [SEP] v1 | v2 | ... [SEP] ... question [SEP]
    Tracks which token positions belong to each (row_id, col_id) data cell.
    Returns (input_ids, attn_mask, ordered_cells, cell_to_positions) or None.
    """
    rows: Dict[int, Dict[int, str]] = {}
    for c in cells:
        if c.row_id > 0 and c.col_id > 0:
            rows.setdefault(c.row_id, {})[c.col_id] = c.content.strip()

    if not rows:
        return None

    sorted_rows = sorted(rows.keys())
    col_ids_sorted = sorted(rows[sorted_rows[0]].keys())

    tokens: List[int] = [tokenizer.cls_token_id]
    cell_to_positions: Dict[Tuple[int, int], List[int]] = {}

    pipe_id = tokenizer.convert_tokens_to_ids('|')
    if pipe_id == tokenizer.unk_token_id:
        pipe_id = None   # skip pipe separator if not in vocab

    for row_id in sorted_rows:
        for i, col_id in enumerate(col_ids_sorted):
            content = rows[row_id].get(col_id, '')
            sub = tokenizer.encode(content, add_special_tokens=False)
            if sub:
                start = len(tokens)
                tokens.extend(sub)
                if row_id > 1:   # data rows only (header row_id=1 not scored)
                    cell_to_positions.setdefault((row_id, col_id), []).extend(
                        range(start, len(tokens))
                    )
            if pipe_id is not None and i < len(col_ids_sorted) - 1:
                tokens.append(pipe_id)
        tokens.append(tokenizer.sep_token_id)

    # Question
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    tokens.extend(q_ids)
    tokens.append(tokenizer.sep_token_id)

    # Truncate
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len - 1] + [tokenizer.sep_token_id]
        cell_to_positions = {
            k: [p for p in v if p < max_seq_len]
            for k, v in cell_to_positions.items()
        }
        cell_to_positions = {k: v for k, v in cell_to_positions.items() if v}

    input_ids = torch.tensor(tokens, dtype=torch.long)
    attn_mask = torch.ones(len(tokens), dtype=torch.long)
    ordered_cells = sorted(cell_to_positions.keys(),
                           key=lambda rc: min(cell_to_positions[rc]))

    return input_ids, attn_mask, ordered_cells, cell_to_positions


# ── Cell-scoring model ─────────────────────────────────────────────────────────

class BaselineCellScorer(nn.Module):
    def __init__(self, encoder, d_model: int):
        super().__init__()
        self.encoder   = encoder
        self.cell_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.cell_head.weight, std=0.02)
        nn.init.zeros_(self.cell_head.bias)

    def forward(self, input_ids, attention_mask, ordered_cells,
                cell_to_positions, device):
        out = self.encoder(
            input_ids=input_ids.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
        )
        hidden = out.last_hidden_state[0]   # (S, d)
        S = hidden.size(0)

        cell_embeds, valid_cells = [], []
        for rc in ordered_cells:
            positions = [p for p in cell_to_positions[rc] if p < S]
            if not positions:
                continue
            h = hidden[positions].mean(0)
            cell_embeds.append(h)
            valid_cells.append(rc)

        if not cell_embeds:
            return None, []

        scores = self.cell_head(torch.stack(cell_embeds)).squeeze(-1)
        return scores, valid_cells


# ── Lookup prediction helper ───────────────────────────────────────────────────

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

    if has_row:
        row_indices = [i for i, rc in enumerate(valid_cells) if rc[0] == best_row]
    else:
        row_indices = list(range(len(valid_cells)))

    if not row_indices:
        return valid_cells[scores.argmax().item()]

    if has_col:
        both = [i for i in row_indices if valid_cells[i][1] == best_col]
        if both:
            return valid_cells[max(both, key=lambda i: scores[i].item())]

    return valid_cells[max(row_indices, key=lambda i: scores[i].item())]


def _predict_comparison(scores, valid_cells, cells, question_text):
    """For comparison questions: restrict to col-1 (entity name) cells of the
    top-2 question-relevant rows. The answer is the winning entity name."""
    q_words = set(question_text.lower().split())
    row_labels = {c.row_id: c.content.strip().lower()
                  for c in cells if c.col_id == 1 and c.row_id > 1}

    if not q_words or not row_labels:
        return valid_cells[scores.argmax().item()]

    row_score = {r: len(q_words & set(lbl.split())) for r, lbl in row_labels.items()}
    top2 = sorted([r for r, s in row_score.items() if s > 0],
                  key=lambda r: row_score[r], reverse=True)[:2]

    if not top2:
        return valid_cells[scores.argmax().item()]

    cand = [i for i, (r, c) in enumerate(valid_cells) if r in top2 and c == 1]
    if not cand:
        cand = [i for i, (r, c) in enumerate(valid_cells) if r in top2]
    if not cand:
        return valid_cells[scores.argmax().item()]

    return valid_cells[max(cand, key=lambda i: scores[i].item())]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, items, csv_dir, device, split='dev'):
    model.eval()
    correct, total, skipped = 0, 0, 0
    by_type: Dict[str, List[int]] = {}

    with torch.no_grad():
        for item in items:
            csv_path = os.path.join(csv_dir, item['table_file'])
            if not os.path.exists(csv_path):
                skipped += 1; continue
            cells = serialize_csv(csv_path)
            if not cells:
                skipped += 1; continue

            question = item['question']
            atype    = item.get('answer_type', '?')

            gold_rc = find_gold_cell(cells, item['answer'], question_text=question)
            if gold_rc is None:
                skipped += 1; continue

            enc = serialize_table_for_bert(cells, question, tokenizer)
            if enc is None:
                skipped += 1; continue
            input_ids, attn_mask, ordered_cells, cell_to_positions = enc

            if gold_rc not in cell_to_positions:
                skipped += 1; continue

            scores, valid_cells = model(input_ids, attn_mask, ordered_cells,
                                        cell_to_positions, device)
            if scores is None or gold_rc not in valid_cells:
                skipped += 1; continue

            if atype == 'lookup':
                pred_rc = _predict_lookup(scores, valid_cells, cells, question)
            elif atype == 'comparison':
                pred_rc = _predict_comparison(scores, valid_cells, cells, question)
            else:
                pred_rc = valid_cells[scores.argmax().item()]
            hit = int(pred_rc == gold_rc)
            correct += hit; total += 1
            by_type.setdefault(atype, []).append(hit)

    em = correct / total if total > 0 else 0.0
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'  [{ts}] {split} EM={em:.4f} ({correct}/{total}, {skipped} skipped)')
    for t, hits in sorted(by_type.items()):
        print(f'    {t}: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.3f}')
    return em, {t: round(sum(h)/len(h), 4) for t, h in by_type.items()}


# ── Training ───────────────────────────────────────────────────────────────────

def train_baseline(model_name, hf_name, train_items, dev_items, csv_dir,
                   device, num_epochs=20, lr=2e-5):
    print(f'\n[BL] ===== {model_name.upper()} ({hf_name}) =====')
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    encoder   = AutoModel.from_pretrained(hf_name)
    d_model   = encoder.config.hidden_size
    model     = BaselineCellScorer(encoder, d_model).to(device)

    print(f'[BL] d_model={d_model}  '
          f'params={sum(p.numel() for p in model.parameters()):,}')

    print('[BL] Zero-shot baseline:')
    best_em, best_by_type = evaluate(model, tokenizer, dev_items,
                                     csv_dir, device, 'zero-shot')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history   = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        random.shuffle(train_items)
        total_loss, n_ok, n_skip = 0.0, 0, 0

        for item in train_items:
            csv_path = os.path.join(csv_dir, item['table_file'])
            if not os.path.exists(csv_path):
                n_skip += 1; continue
            cells = serialize_csv(csv_path)
            if not cells:
                n_skip += 1; continue

            gold_rc = find_gold_cell(cells, item['answer'])
            if gold_rc is None:
                n_skip += 1; continue

            enc = serialize_table_for_bert(cells, item['question'], tokenizer)
            if enc is None:
                n_skip += 1; continue
            input_ids, attn_mask, ordered_cells, cell_to_positions = enc

            if gold_rc not in cell_to_positions:
                n_skip += 1; continue

            scores, valid_cells = model(input_ids, attn_mask, ordered_cells,
                                        cell_to_positions, device)
            if scores is None or gold_rc not in valid_cells:
                n_skip += 1; continue

            gold_idx = torch.tensor([valid_cells.index(gold_rc)],
                                    dtype=torch.long, device=device)
            loss = criterion(scores.unsqueeze(0), gold_idx)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item(); n_ok += 1

        ts  = datetime.now().strftime('%H:%M:%S')
        avg = total_loss / max(n_ok, 1)
        print(f'[{ts}] Epoch {epoch}/{num_epochs}  loss={avg:.4f}  '
              f'trained={n_ok}  skipped={n_skip}')

        em, by_type = evaluate(model, tokenizer, dev_items, csv_dir, device)
        history.append({'epoch': epoch, 'loss': avg, 'dev_em': em})

        if em > best_em:
            best_em, best_by_type = em, by_type
            sp = os.path.join(DATA_DIR, f'baseline_{model_name}_best.pt')
            torch.save({'model_state_dict': model.state_dict(), 'em': em}, sp)
            print(f'  ** New best EM={best_em:.4f}')

    print(f'\n[BL] {model_name.upper()} done. Best EM={best_em:.4f}')
    return best_em, best_by_type, history


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['mbert'],
                        choices=list(MODELS.keys()))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr',     type=float, default=2e-5)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[BL] Device: {device}')

    with open(TABQA_FILE) as f:
        all_items = json.load(f)
    random.shuffle(all_items)
    split_n     = int(0.8 * len(all_items))
    train_items = all_items[:split_n]
    dev_items   = all_items[split_n:]
    print(f'[BL] {len(all_items)} items  Train={len(train_items)}  Dev={len(dev_items)}')

    all_results = {}
    for model_name in args.models:
        hf_name = MODELS[model_name]
        best_em, best_by_type, history = train_baseline(
            model_name, hf_name, train_items, dev_items, CSV_DIR, device,
            num_epochs=args.epochs, lr=args.lr,
        )
        all_results[model_name] = {
            'hf_name': hf_name, 'best_em': best_em,
            'by_type': best_by_type, 'history': history,
        }

    out = os.path.join(DATA_DIR, 'baseline_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[BL] Results saved to {out}')

    print('\n' + '='*58)
    print(f'{"Model":<18} {"EM":>6}  {"Agg":>5}  {"Cmp":>5}  {"Lkp":>5}')
    print('-'*58)
    for name, r in all_results.items():
        bt = r['by_type']
        print(f'{name:<18} {r["best_em"]:>6.3f}  '
              f'{bt.get("aggregation",0):>5.3f}  '
              f'{bt.get("comparison",0):>5.3f}  '
              f'{bt.get("lookup",0):>5.3f}')
    print(f'{"TabuLM (ours)":<18} {"0.560":>6}  '
          f'{"0.792":>5}  {"0.417":>5}  {"0.286":>5}')


if __name__ == '__main__':
    main()
