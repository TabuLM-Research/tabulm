"""
finetune_tableqa.py — Fine-tune TabuLM on TabQA-kin benchmark.

Fine-tuning approach:
  - Serialize table + question → token string via tabular_serializer
  - Append question tokens after [SEP]
  - Predict answer as span (start/end token indices) for lookup/comparison/aggregation
  - Predict integer for count questions
  - Metrics: exact match (EM) and token-level F1

Usage:
    python code/finetune_tableqa.py \
        --checkpoint checkpoints/tabulm_step120000.pt \
        --tabqa     data/tabqa_kin.json \
        --tables    data/tables/ \
        --output    checkpoints/tabulm_tableqa/
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

random.seed(42)
torch.manual_seed(42)

# ── Span prediction head ──────────────────────────────────────────────────────

class SpanHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.start_linear = nn.Linear(hidden_size, 1)
        self.end_linear   = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        start_logits = self.start_linear(hidden_states).squeeze(-1)
        end_logits   = self.end_linear(hidden_states).squeeze(-1)
        return start_logits, end_logits


class TabuLM_QA(nn.Module):
    def __init__(self, encoder, hidden_size: int = 768):
        super().__init__()
        self.encoder   = encoder
        self.span_head = SpanHead(hidden_size)
        self.count_head = nn.Linear(hidden_size, 100)

    def forward(self, batch, answer_type: str = 'lookup'):
        hidden = self.encoder(batch)['last_hidden_state']
        if answer_type == 'count':
            cls_hidden = hidden[:, 0, :]
            return self.count_head(cls_hidden)
        else:
            return self.span_head(hidden)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TabQADataset(Dataset):
    def __init__(self, items: List[Dict], tables_dir: Path, kb_vocab, bpe):
        from tabular_serializer import serialize_csv, table_cells_to_text
        from morpho_stub import parse_text_stub

        self.samples = []
        for item in items:
            table_path = tables_dir / item['table_file']
            if not table_path.exists():
                continue
            cells = serialize_csv(str(table_path))
            table_text, word_meta = table_cells_to_text(cells)

            question = item['question']
            full_text = table_text + ' [SEP] ' + question

            tokens = parse_text_stub(full_text, kb_vocab, bpe)
            self.samples.append({
                'tokens':      tokens,
                'word_meta':   word_meta,
                'answer':      item['answer'],
                'answer_type': item['answer_type'],
                'answer_row':  item.get('answer_row', -1),
                'answer_col':  item.get('answer_col', -1),
                'id':          item['id'],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Metrics ───────────────────────────────────────────────────────────────────

def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())


def token_f1(pred: str, gold: str) -> float:
    pred_toks = pred.strip().lower().split()
    gold_toks = gold.strip().lower().split()
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    p = len(common) / len(pred_toks) if pred_toks else 0
    r = len(common) / len(gold_toks) if gold_toks else 0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    em_sum = f1_sum = total = 0
    with torch.no_grad():
        for batch in dataloader:
            answer_type = batch['answer_type'][0]
            preds = model(batch, answer_type=answer_type)

            if answer_type == 'count':
                pred_counts = preds.argmax(dim=-1).tolist()
                for pred, gold in zip(pred_counts, batch['answer']):
                    em_sum  += exact_match(str(pred), gold)
                    f1_sum  += token_f1(str(pred), gold)
                    total   += 1
            else:
                start_logits, end_logits = preds
                for i in range(len(batch['answer'])):
                    s = start_logits[i].argmax().item()
                    e = end_logits[i].argmax().item()
                    tokens = batch['tokens'][i]
                    pred_text = ' '.join(t.stem for t in tokens[s:e+1]) if s <= e else ''
                    gold_text = batch['answer'][i]
                    em_sum  += exact_match(pred_text, gold_text)
                    f1_sum  += token_f1(pred_text, gold_text)
                    total   += 1

    return {
        'exact_match': em_sum / total if total else 0,
        'f1':          f1_sum / total if total else 0,
        'total':       total,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.tabqa, encoding='utf-8') as f:
        all_items = json.load(f)

    random.shuffle(all_items)
    split = int(0.8 * len(all_items))
    train_items = all_items[:split]
    dev_items   = all_items[split:]
    print(f'Train: {len(train_items)}  Dev: {len(dev_items)}')

    import sys; sys.path.insert(0, 'code')
    from morpho_common import KBVocab
    from morpho_stub   import BPEFallback

    kb_vocab = KBVocab.load(args.kb_vocab)
    bpe      = BPEFallback(args.bpe_codes)

    train_ds = TabQADataset(train_items, Path(args.tables), kb_vocab, bpe)
    dev_ds   = TabQADataset(dev_items,   Path(args.tables), kb_vocab, bpe)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    from tabulm_model import tabulm_base
    model_core = tabulm_base(kb_vocab)
    model_core.load_state_dict(checkpoint['model'], strict=False)

    model = TabuLM_QA(model_core.encoder).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ce_loss   = nn.CrossEntropyLoss()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_em = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)):
            optimizer.zero_grad()
            answer_type = batch['answer_type'][0]
            preds = model(batch, answer_type=answer_type)

            if answer_type == 'count':
                labels = torch.tensor([int(a) for a in batch['answer']], device=device)
                loss = ce_loss(preds, labels)
            else:
                start_logits, end_logits = preds
                start_labels = torch.tensor(batch['answer_row'], device=device)
                end_labels   = torch.tensor(batch['answer_row'], device=device)
                loss = ce_loss(start_logits, start_labels) + ce_loss(end_logits, end_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0:
                print(f'  epoch {epoch+1} step {step}  loss={total_loss/(step+1):.4f}')

        metrics = evaluate(model, DataLoader(dev_ds, batch_size=32), device)
        print(f'Epoch {epoch+1}  EM={metrics["exact_match"]:.4f}  F1={metrics["f1"]:.4f}')

        if metrics['exact_match'] > best_em:
            best_em = metrics['exact_match']
            ckpt_path = Path(args.output) / 'best_tableqa.pt'
            torch.save({'model': model.state_dict(), 'metrics': metrics}, ckpt_path)
            print(f'  → saved best checkpoint  EM={best_em:.4f}')

    print(f'\nBest EM: {best_em:.4f}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tabqa',      default='data/tabqa_kin.json')
    parser.add_argument('--tables',     default='data/tables')
    parser.add_argument('--output',     default='checkpoints/tabulm_tableqa')
    parser.add_argument('--kb_vocab',   default='conf/kb_vocab_state_dict_2021-02-07.pt')
    parser.add_argument('--bpe_codes',  default='conf/bpe_codes.txt')
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=2e-5)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
