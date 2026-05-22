#!/usr/bin/env python3
"""Fine-tune TabuLM for cell-selection on TabQA-kin (526 Kinyarwanda table QA pairs).

Strategy: Given a (table, question) pair, encode both through the TabuLM encoder
and predict which table cell (row_id, col_id) contains the answer via a linear
scoring head over per-cell pooled hidden states.
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

# ── Server paths ───────────────────────────────────────────────────────────────
CODE_DIR = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/code'
DATA_DIR = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/data'
CHECKPOINT = os.path.join(DATA_DIR, 'tabulm_model_2026-05-13_pos@1_stem@1_afsets@False@10000.pt')
TABQA_FILE = os.path.join(DATA_DIR, 'tabqa_kin.json')
CSV_DIR    = os.path.join(DATA_DIR, 'tables')
BEST_MODEL = os.path.join(DATA_DIR, 'finetune_tabqa_best.pt')
RESULTS    = os.path.join(DATA_DIR, 'finetune_tabqa_results.json')

sys.path.insert(0, CODE_DIR)

import youtokentome as yttm
from morpho_data_loaders import KBVocab
from tabular_serializer import serialize_csv, table_cells_to_text, TableCell
from morpho_stub import parse_text_stub
from tabulm_model import TabuLM, tabulm_base


# ── Architecture auto-detection ────────────────────────────────────────────────

def detect_arch_from_state(state: dict) -> argparse.Namespace:
    """Infer all architecture hyperparameters by inspecting checkpoint tensor shapes."""
    ns = argparse.Namespace()

    # Sequence transformer model dimension
    ns.seq_tr_d_model = int(state['encoder.row_embedding.weight'].shape[1])
    d = ns.seq_tr_d_model

    # Stem / morpho dims
    ns.stem_dim  = int(state['encoder.s_stem_embedding.weight'].shape[1])
    if 'encoder.m1_pos_embedding.weight' in state:
        ns.morpho_dim = int(state['encoder.m1_pos_embedding.weight'].shape[1])
    elif 'encoder.m_stem_embedding.weight' in state:
        ns.morpho_dim = int(state['encoder.m_stem_embedding.weight'].shape[1])
    else:
        ns.morpho_dim = 128

    # Sequence transformer layers
    layer_ids = {
        int(k.split('.layers.')[1].split('.')[0])
        for k in state
        if 'encoder.seq_transformer_encoder.layers.' in k and '.layers.' in k
    }
    ns.seq_tr_nlayers = max(layer_ids) + 1 if layer_ids else 12

    # Heads from tabular bias parameter
    if 'encoder.row_attn_bias' in state:
        ns.seq_tr_nhead = int(state['encoder.row_attn_bias'].shape[0])
    else:
        ns.seq_tr_nhead = 8

    # Feed-forward dim
    ff_key = 'encoder.seq_transformer_encoder.layers.0.linear1.weight'
    ns.seq_tr_dim_feedforward = int(state[ff_key].shape[0]) if ff_key in state else d * 4

    # Morpho transformer layers
    m_ids = {
        int(k.split('.layers.')[1].split('.')[0])
        for k in state
        if 'encoder.morpho_transformer_encoder.layers.' in k and '.layers.' in k
    }
    ns.morpho_tr_nlayers = max(m_ids) + 1 if m_ids else 4

    mff_key = 'encoder.morpho_transformer_encoder.layers.0.linear1.weight'
    ns.morpho_tr_dim_feedforward = int(state[mff_key].shape[0]) if mff_key in state else ns.morpho_dim * 4

    ns.morpho_tr_nhead = max(1, ns.morpho_dim // 32)  # head_dim=32 default
    ns.morpho_tr_dropout = 0.1
    ns.seq_tr_dropout    = 0.1
    ns.layernorm_epsilon = 1e-6
    ns.max_seq_len = 512

    # Tabular embedding flags
    ns.num_pos_m_embeddings  = 1
    ns.num_stem_m_embeddings = 1
    ns.use_afsets            = False
    ns.afset_dict_size       = 10000
    ns.use_morpho_encoder    = True

    morpho_part     = d - ns.stem_dim
    tot_morpho_vecs = morpho_part // ns.morpho_dim if ns.morpho_dim else 2
    tot_morpho_idx  = ns.num_pos_m_embeddings + ns.num_stem_m_embeddings
    ns.use_affix_bow_m_embedding = (tot_morpho_vecs > tot_morpho_idx)

    ns.use_tupe_rel_pos_bias       = 'encoder.tupe_rel_pos_bias.weight' in state \
                                     or any('tupe' in k for k in state)
    ns.use_pos_aware_rel_pos_bias  = any('pos_aware' in k for k in state)
    ns.use_pos_aware_rel           = ns.use_pos_aware_rel_pos_bias
    ns.predict_affixes             = False

    # Needed by tabulm_base factory
    ns.gpus            = 0
    ns.world_size      = 1
    ns.exploratory_model_load = None

    return ns


# ── Gold cell lookup via text matching ────────────────────────────────────────

def find_gold_cell(cells: List[TableCell], answer_text: str,
                   question_text: str = '') -> Optional[Tuple[int, int]]:
    """
    Find the (row_id, col_id) of the cell whose content matches answer_text.
    When multiple cells share the same value, uses question-word overlap against
    both first-column row labels AND header-row column names to disambiguate.
    """
    answer_norm = answer_text.strip().lower()

    # Pass 1: exact match
    matches = [(c.row_id, c.col_id) for c in cells
               if c.row_id > 1 and c.col_id > 0 and c.content.strip() == answer_text.strip()]
    if len(matches) == 1:
        return matches[0]

    # Pass 2: case-insensitive
    if not matches:
        matches = [(c.row_id, c.col_id) for c in cells
                   if c.row_id > 1 and c.col_id > 0 and c.content.strip().lower() == answer_norm]

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # Multiple cells match — score by (row overlap, col overlap) jointly
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


# ── Sequence encoding (no masking) ────────────────────────────────────────────

def _add_special(key, kv, pos_tags, stems, tokens_lengths, row_ids, col_ids, cell_types):
    pos_tags.append(kv.pos_tag_vocab[key])
    stems.append(kv.reduced_stem_vocab[key])
    tokens_lengths.append(0)
    row_ids.append(0)
    col_ids.append(0)
    cell_types.append(0)


def encode_item(
    cells,
    question_text: str,
    kv: KBVocab,
    bpe,
    max_seq_len: int = 512,
) -> Optional[Tuple]:
    """
    Encode a table (List[TableCell]) + question string into flat lists for the
    TabuLM encoder. Returns None if the table serialization fails.

    Returns:
      pos_tags, stems, affixes, tokens_lengths,
      row_ids, col_ids, cell_types,
      ordered_cells, cell_to_positions
    where ordered_cells is a sorted list of unique (row_id, col_id) tuples, and
    cell_to_positions maps each (row_id, col_id) → [token_idx, ...].
    """
    text, word_meta = table_cells_to_text(cells)
    parsed_table = parse_text_stub(text, kv, bpe)
    if len(parsed_table) != len(word_meta):
        return None

    parsed_question = parse_text_stub(question_text, kv, bpe)

    pos_tags, stems, affixes, tokens_lengths = [], [], [], []
    row_ids, col_ids, cell_types = [], [], []
    cell_to_positions: Dict[Tuple[int, int], List[int]] = {}

    # CLS
    _add_special('<CLS>', kv, pos_tags, stems, tokens_lengths, row_ids, col_ids, cell_types)

    # Table tokens — each ParsedToken may have multiple sub-stems (BPE)
    for pt, (r, c, ct) in zip(parsed_table, word_meta):
        for sidx in pt.stem_idx:
            seq_idx = len(pos_tags)
            pos_tags.append(pt.pos_tag_idx)
            stems.append(kv.mapped_stem_vocab_idx[sidx])
            affixes.extend(pt.affixes_idx)
            tokens_lengths.append(len(pt.affixes_idx))
            row_ids.append(r)
            col_ids.append(c)
            cell_types.append(ct)
            if r > 0 and c > 0:
                cell_to_positions.setdefault((r, c), []).append(seq_idx)

    # SEP between table and question
    _add_special('<SEP>', kv, pos_tags, stems, tokens_lengths, row_ids, col_ids, cell_types)

    # Question tokens (row_id=0, col_id=0 → not scored as table cells)
    for pt in parsed_question:
        for sidx in pt.stem_idx:
            pos_tags.append(pt.pos_tag_idx)
            stems.append(kv.mapped_stem_vocab_idx[sidx])
            affixes.extend(pt.affixes_idx)
            tokens_lengths.append(len(pt.affixes_idx))
            row_ids.append(0)
            col_ids.append(0)
            cell_types.append(0)

    # Final SEP
    _add_special('<SEP>', kv, pos_tags, stems, tokens_lengths, row_ids, col_ids, cell_types)

    # Truncate to max_seq_len
    if len(pos_tags) > max_seq_len:
        pos_tags      = pos_tags[:max_seq_len]
        stems         = stems[:max_seq_len]
        tokens_lengths = tokens_lengths[:max_seq_len]
        row_ids       = row_ids[:max_seq_len]
        col_ids       = col_ids[:max_seq_len]
        cell_types    = cell_types[:max_seq_len]
        affixes       = affixes[:sum(tokens_lengths)]
        cell_to_positions = {
            k: [p for p in v if p < max_seq_len]
            for k, v in cell_to_positions.items()
        }
        cell_to_positions = {k: v for k, v in cell_to_positions.items() if v}

    ordered_cells = sorted(cell_to_positions.keys(),
                           key=lambda rc: min(cell_to_positions[rc]))

    return (pos_tags, stems, affixes, tokens_lengths,
            row_ids, col_ids, cell_types,
            ordered_cells, cell_to_positions)


# ── Fine-tuning model ──────────────────────────────────────────────────────────

class TabQAModel(nn.Module):
    """TabuLM encoder + linear cell-selection head."""

    def __init__(self, tabulm: TabuLM):
        super().__init__()
        self.encoder   = tabulm.encoder
        d = self.encoder.seq_tr_d_model
        self.cell_head = nn.Linear(d, 1)
        nn.init.normal_(self.cell_head.weight, std=0.02)
        nn.init.zeros_(self.cell_head.bias)

    def get_hidden(self, args, pos_tags, stems, affixes,
                   tokens_lengths, row_ids, col_ids, cell_types, device):
        pos_t  = torch.tensor(pos_tags,     dtype=torch.long, device=device)
        stem_t = torch.tensor(stems,        dtype=torch.long, device=device)
        afx_t  = (torch.tensor(affixes, dtype=torch.long, device=device)
                  if affixes else torch.zeros(0, dtype=torch.long, device=device))
        row_t  = torch.tensor(row_ids,      dtype=torch.long, device=device)
        col_t  = torch.tensor(col_ids,      dtype=torch.long, device=device)
        ct_t   = torch.tensor(cell_types,   dtype=torch.long, device=device)

        hidden = self.encoder.forward(
            args,
            rel_pos_arr=None,
            tokens_lengths=tokens_lengths,
            input_sequence_lengths=[len(pos_tags)],
            pos_tags=pos_t, stems=stem_t, afsets=None, affixes=afx_t,
            row_ids=row_t, col_ids=col_t, cell_types=ct_t,
        )  # (S, 1, d)
        return hidden[:, 0, :]  # (S, d)

    def forward(self, args, pos_tags, stems, affixes,
                tokens_lengths, row_ids, col_ids, cell_types,
                ordered_cells, cell_to_positions, device):
        """
        Returns (scores, valid_cells) where scores is a (C,) tensor of logits
        and valid_cells is the subset of ordered_cells that had token positions.
        """
        hidden = self.get_hidden(args, pos_tags, stems, affixes,
                                 tokens_lengths, row_ids, col_ids, cell_types, device)
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

        cell_embeds = torch.stack(cell_embeds)          # (C, d)
        scores      = self.cell_head(cell_embeds).squeeze(-1)  # (C,)
        return scores, valid_cells


# ── Evaluation ────────────────────────────────────────────────────────────────

def _predict_lookup(scores, valid_cells, cells, question_text):
    """For lookup questions: restrict to (best_row, best_col) using question-word
    overlap against row labels (first column) and column headers (header row)."""
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

    # Narrow to best row first
    if has_row:
        row_indices = [i for i, rc in enumerate(valid_cells) if rc[0] == best_row]
    else:
        row_indices = list(range(len(valid_cells)))

    if not row_indices:
        return valid_cells[scores.argmax().item()]

    # Further narrow to best column if signal available
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

    # Prefer col-1 cells (entity names) in those rows; fall back to any cell
    cand = [i for i, (r, c) in enumerate(valid_cells) if r in top2 and c == 1]
    if not cand:
        cand = [i for i, (r, c) in enumerate(valid_cells) if r in top2]
    if not cand:
        return valid_cells[scores.argmax().item()]

    return valid_cells[max(cand, key=lambda i: scores[i].item())]


def evaluate(model, args, items, kv, bpe, csv_dir, device, split='dev'):
    model.eval()
    correct  = 0
    total    = 0
    skipped  = 0
    by_type: Dict[str, List[int]] = {}

    with torch.no_grad():
        for item in items:
            csv_path = os.path.join(csv_dir, item['table_file'])
            if not os.path.exists(csv_path):
                skipped += 1
                continue
            cells = serialize_csv(csv_path)
            if not cells:
                skipped += 1
                continue

            question = item['question']
            atype    = item.get('answer_type', 'unknown')

            # Find gold cell — use question context to resolve duplicate values
            gold_rc = find_gold_cell(cells, item['answer'], question_text=question)
            if gold_rc is None:
                skipped += 1
                continue

            enc = encode_item(cells, question, kv, bpe)
            if enc is None:
                skipped += 1
                continue

            (pos_tags, stems, affixes, tokens_lengths,
             row_ids, col_ids, cell_types, ordered_cells, cell_to_positions) = enc

            if gold_rc not in cell_to_positions:
                skipped += 1
                continue

            scores, valid_cells = model(args, pos_tags, stems, affixes,
                                        tokens_lengths, row_ids, col_ids, cell_types,
                                        ordered_cells, cell_to_positions, device)
            if scores is None or gold_rc not in valid_cells:
                skipped += 1
                continue

            if atype == 'lookup':
                pred_rc = _predict_lookup(scores, valid_cells, cells, question)
            elif atype == 'comparison':
                pred_rc = _predict_comparison(scores, valid_cells, cells, question)
            else:
                pred_rc = valid_cells[scores.argmax().item()]
            hit     = int(pred_rc == gold_rc)
            correct += hit
            total   += 1
            by_type.setdefault(atype, []).append(hit)

    em = correct / total if total > 0 else 0.0
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'  [{ts}] {split} EM={em:.4f} ({correct}/{total} correct, {skipped} skipped)')
    for atype, hits in sorted(by_type.items()):
        print(f'    {atype}: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.3f}')
    return em


# ── Training ──────────────────────────────────────────────────────────────────

def train_finetune(args, model, train_items, dev_items, kv, bpe, csv_dir, device,
                   num_epochs=20, lr=2e-5, best_model_path=BEST_MODEL):
    # Fine-tune top-4 seq transformer layers + cell head; freeze the rest
    for name, p in model.named_parameters():
        if 'cell_head' in name:
            p.requires_grad = True
        elif 'seq_transformer_encoder.layers.' in name:
            # Extract layer index
            try:
                layer_idx = int(name.split('seq_transformer_encoder.layers.')[1].split('.')[0])
                total_layers = model.encoder.seq_tr_nlayers \
                    if hasattr(model.encoder, 'seq_tr_nlayers') else 12
                p.requires_grad = (layer_idx >= total_layers - 4)
            except (IndexError, ValueError):
                p.requires_grad = False
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[FT] Trainable parameters: {trainable:,}')

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    best_em  = 0.0
    history  = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        random.shuffle(train_items)
        total_loss = 0.0
        n_ok       = 0
        n_skip     = 0

        for item in train_items:
            csv_path = os.path.join(csv_dir, item['table_file'])
            if not os.path.exists(csv_path):
                n_skip += 1
                continue
            cells = serialize_csv(csv_path)
            if not cells:
                n_skip += 1
                continue

            # Find gold cell by text matching (bypasses broken coord system)
            gold_rc = find_gold_cell(cells, item['answer'])
            if gold_rc is None:
                n_skip += 1
                continue

            enc = encode_item(cells, item['question'], kv, bpe)
            if enc is None:
                n_skip += 1
                continue

            (pos_tags, stems, affixes, tokens_lengths,
             row_ids, col_ids, cell_types, ordered_cells, cell_to_positions) = enc

            if gold_rc not in cell_to_positions:
                n_skip += 1
                continue

            scores, valid_cells = model(args, pos_tags, stems, affixes,
                                        tokens_lengths, row_ids, col_ids, cell_types,
                                        ordered_cells, cell_to_positions, device)
            if scores is None or gold_rc not in valid_cells:
                n_skip += 1
                continue

            gold_idx = torch.tensor([valid_cells.index(gold_rc)],
                                    dtype=torch.long, device=device)
            loss = criterion(scores.unsqueeze(0), gold_idx)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_ok       += 1

        ts = datetime.now().strftime('%H:%M:%S')
        avg_loss = total_loss / max(n_ok, 1)
        print(f'[{ts}] Epoch {epoch}/{num_epochs}  loss={avg_loss:.4f}  '
              f'trained={n_ok}  skipped={n_skip}')

        em = evaluate(model, args, dev_items, kv, bpe, csv_dir, device, split='dev')
        history.append({'epoch': epoch, 'train_loss': avg_loss, 'dev_em': em})

        if em > best_em:
            best_em = em
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'dev_em': em,
            }, best_model_path)
            print(f'  ** New best EM={best_em:.4f}  saved to {best_model_path}')

    return best_em, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT,
                        help='Path to pre-trained TabuLM checkpoint (.pt)')
    parser.add_argument('--output-prefix', type=str, default='finetune_tabqa',
                        help='Prefix for best-model and results files (no extension)')
    cli = parser.parse_args()

    best_model_path = os.path.join(DATA_DIR, f'{cli.output_prefix}_best.pt')
    results_path    = os.path.join(DATA_DIR, f'{cli.output_prefix}_results.json')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[FT] Device: {device}')

    # Load vocab + BPE
    print('[FT] Loading vocab and BPE...')
    bpe = yttm.BPE(model=os.path.join(DATA_DIR, 'BPE-30k.mdl'))
    kv  = KBVocab()
    kv.load_state_dict(torch.load(os.path.join(DATA_DIR, 'kb_vocab_state_dict_2021-02-07.pt'),
                                  map_location='cpu'))

    # Load checkpoint and detect architecture
    print(f'[FT] Loading checkpoint from {cli.checkpoint}')
    ckpt  = torch.load(cli.checkpoint, map_location='cpu')
    state = ckpt['model_state_dict']
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}

    args = detect_arch_from_state(state)
    print(f'[FT] Detected: d_model={args.seq_tr_d_model}  nlayers={args.seq_tr_nlayers}  '
          f'nhead={args.seq_tr_nhead}  ff={args.seq_tr_dim_feedforward}  '
          f'morpho_dim={args.morpho_dim}  stem_dim={args.stem_dim}  '
          f'bow={args.use_affix_bow_m_embedding}')

    # Build model and load weights
    tabulm = tabulm_base(kv, None, None, device, args, saved_model_file=None)
    missing, unexpected = tabulm.load_state_dict(state, strict=False)
    print(f'[FT] Loaded encoder: {len(missing)} missing, {len(unexpected)} unexpected keys')

    model = TabQAModel(tabulm).to(device)

    # Load and split TabQA-kin
    print(f'[FT] Loading TabQA-kin from {TABQA_FILE}')
    with open(TABQA_FILE) as f:
        all_items = json.load(f)
    print(f'[FT] {len(all_items)} QA items total')

    random.shuffle(all_items)
    split_n     = int(0.8 * len(all_items))
    train_items = all_items[:split_n]
    dev_items   = all_items[split_n:]
    print(f'[FT] Train={len(train_items)}  Dev={len(dev_items)}')

    print('[FT] Baseline (pre-training weights, no task training):')
    evaluate(model, args, dev_items, kv, bpe, CSV_DIR, device, split='dev-baseline')

    best_em, history = train_finetune(
        args, model, train_items, dev_items, kv, bpe, CSV_DIR, device,
        num_epochs=20, lr=2e-5, best_model_path=best_model_path,
    )

    print(f'\n[FT] Done. Best dev EM = {best_em:.4f}')

    with open(results_path, 'w') as f:
        json.dump({'best_em': best_em, 'epochs': history,
                   'train_size': len(train_items), 'dev_size': len(dev_items)}, f, indent=2)
    print(f'[FT] Results saved to {results_path}')


if __name__ == '__main__':
    main()
