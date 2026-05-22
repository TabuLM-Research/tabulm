#!/usr/bin/env python3
"""Zero-shot LLM baseline on TabQA-kin dev set. Supports OpenAI and Anthropic.

Usage:
  # OpenAI (GPT-4o)
  python eval_llm_baseline.py --provider openai --api-key sk-...
  python eval_llm_baseline.py --provider openai --api-key sk-... --model gpt-4o-mini

  # Anthropic (Claude Haiku)
  python eval_llm_baseline.py --provider anthropic --api-key sk-ant-...
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

DATA_DIR   = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/data'
CODE_DIR   = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/code'
TABQA_FILE = os.path.join(DATA_DIR, 'tabqa_kin.json')
CSV_DIR    = os.path.join(DATA_DIR, 'tables')

sys.path.insert(0, CODE_DIR)
from tabular_serializer import serialize_csv, TableCell

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL    = "https://api.openai.com/v1/chat/completions"


# ── Table formatting ───────────────────────────────────────────────────────────

def csv_to_markdown(csv_path: str) -> Optional[str]:
    try:
        with open(csv_path, encoding='utf-8') as f:
            rows = list(csv.reader(f))
        if not rows:
            return None
        lines = [' | '.join(r) for r in rows]
        # insert separator after header
        lines.insert(1, ' | '.join(['---'] * len(rows[0])))
        return '\n'.join(lines)
    except Exception:
        return None


# ── Gold cell lookup (same logic as finetune_tabqa.py v3) ─────────────────────

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


# ── Prompt construction ────────────────────────────────────────────────────────

TYPE_HINTS = {
    'lookup':      'The answer is a specific value (number or text) from the table. Find the row matching the entity named in the question, then return the value in the relevant column.',
    'comparison':  'The answer is the NAME of one of the two entities mentioned in the question — NOT a number. Find both entities in the first column, compare their values in the specified data column, and return ONLY the entity name that has the higher or lower value as asked.',
    'aggregation': 'The answer is the NAME of an entity from the first column of the table — NOT a number. Identify which entity has the highest or lowest value in the relevant column, then return that entity\'s name exactly as it appears in the table.',
}

def make_prompt(table_md: str, question: str, answer_type: str) -> str:
    hint = TYPE_HINTS.get(answer_type, '')
    return (
        f"You are answering a question about a data table written in Kinyarwanda.\n\n"
        f"Table:\n{table_md}\n\n"
        f"Question: {question}\n\n"
        f"{hint}\n\n"
        f"Reply with ONLY the exact value from the table that answers the question. "
        f"No explanation, no added words."
    )


# ── API call ───────────────────────────────────────────────────────────────────

def call_llm(prompt: str, api_key: str, model: str, provider: str,
             retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            if provider == 'openai':
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                }
                payload = {
                    "model":      model,
                    "max_tokens": 64,
                    "messages":   [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
                r = requests.post(OPENAI_URL, headers=headers,
                                  json=payload, timeout=30)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            else:  # anthropic
                headers = {
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                }
                payload = {
                    "model":      model,
                    "max_tokens": 64,
                    "messages":   [{"role": "user", "content": prompt}],
                }
                r = requests.post(ANTHROPIC_URL, headers=headers,
                                  json=payload, timeout=30)
                if r.status_code == 529:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()["content"][0]["text"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [API error] {e}")
    return None


# ── EM with normalization ──────────────────────────────────────────────────────

def em_match(pred: str, gold: str) -> bool:
    def norm(s: str) -> str:
        s = s.strip().lower().replace(',', '').replace(' ', '')
        try:
            return f'{float(s):g}'
        except ValueError:
            return s
    return norm(pred) == norm(gold)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key',  required=True, help='OpenAI or Anthropic API key')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic'])
    parser.add_argument('--model',    default=None,
                        help='Model name (default: gpt-4o for OpenAI, claude-haiku-4-5-20251001 for Anthropic)')
    parser.add_argument('--out',      default=os.path.join(DATA_DIR, 'llm_baseline_results.json'))
    parser.add_argument('--max-items', type=int, default=None,
                        help='Limit number of dev items (for testing)')
    args = parser.parse_args()
    if args.model is None:
        args.model = 'gpt-4o' if args.provider == 'openai' else 'claude-haiku-4-5-20251001'

    random.seed(42)
    with open(TABQA_FILE) as f:
        all_items = json.load(f)
    random.shuffle(all_items)
    dev_items = all_items[int(0.8 * len(all_items)):]
    if args.max_items:
        dev_items = dev_items[:args.max_items]
    print(f'[LLM] Provider: {args.provider}  Model: {args.model}')
    print(f'[LLM] {len(dev_items)} dev items')

    correct, total, skipped = 0, 0, 0
    by_type: Dict[str, List[int]] = {}
    records = []

    for i, item in enumerate(dev_items):
        csv_path = os.path.join(CSV_DIR, item['table_file'])
        atype    = item.get('answer_type', '?')

        # skip count questions (answer not a cell value)
        if atype == 'count':
            skipped += 1; continue

        if not os.path.exists(csv_path):
            skipped += 1; continue

        cells = serialize_csv(csv_path)
        if not cells:
            skipped += 1; continue

        # verify gold cell exists (same filter as our fine-tuning eval)
        gold_rc = find_gold_cell(cells, item['answer'], item['question'])
        if gold_rc is None:
            skipped += 1; continue

        table_md = csv_to_markdown(csv_path)
        if not table_md:
            skipped += 1; continue

        prompt = make_prompt(table_md, item['question'], atype)
        pred   = call_llm(prompt, args.api_key, args.model, args.provider)
        if pred is None:
            skipped += 1; continue

        hit = int(em_match(pred, item['answer']))
        correct += hit
        total   += 1
        by_type.setdefault(atype, []).append(hit)
        records.append({
            'idx':          i,
            'question':     item['question'],
            'table':        item['table_file'],
            'answer_type':  atype,
            'gold':         item['answer'],
            'pred':         pred,
            'hit':          hit,
        })

        if (i + 1) % 10 == 0:
            running_em = correct / total if total else 0.0
            print(f'  [{i+1:3d}/{len(dev_items)}] running EM={running_em:.3f}  '
                  f'correct={correct}/{total}  skipped={skipped}')

    em = correct / total if total else 0.0
    print(f'\n[LLM] Final EM = {em:.4f} ({correct}/{total}, {skipped} skipped)')
    for t, hits in sorted(by_type.items()):
        print(f'  {t}: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.3f}')

    result = {
        'model':    args.model,
        'em':       round(em, 4),
        'correct':  correct,
        'total':    total,
        'skipped':  skipped,
        'by_type':  {t: round(sum(h)/len(h), 4) for t, h in by_type.items()},
        'records':  records,
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[LLM] Results saved to {args.out}')


if __name__ == '__main__':
    main()
