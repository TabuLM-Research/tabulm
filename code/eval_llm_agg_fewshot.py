#!/usr/bin/env python3
"""3-shot GPT-4o experiment on aggregation items only.
Tests whether few-shot prompting closes the aggregation gap vs TabuLM.

Usage:
  python eval_llm_agg_fewshot.py --api-key sk-...
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import List, Optional

import requests

DATA_DIR   = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/data'
CODE_DIR   = '/shared/scratch/0/tmp/v_ireddi_rakshitha_results/tabulm/code'
TABQA_FILE = os.path.join(DATA_DIR, 'tabqa_kin.json')
CSV_DIR    = os.path.join(DATA_DIR, 'tables')
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

sys.path.insert(0, CODE_DIR)
from tabular_serializer import serialize_csv, TableCell
from eval_llm_baseline import csv_to_markdown, find_gold_cell, em_match


FEW_SHOT_EXAMPLES = [
    {
        "table": "| Inzego | Umubare_w_Abaturage | Umubare_w_Inzu |\n|---|---|---|\n| Gasabo | 542000 | 89000 |\n| Kicukiro | 312000 | 54000 |\n| Nyarugenge | 274000 | 48000 |",
        "question": "Ni iki gifite Umubare_w_Abaturage nyinshi?",
        "answer": "Gasabo",
        "explanation": "Gasabo has the highest Umubare_w_Abaturage (542000).",
    },
    {
        "table": "| Igihingwa | Umusaruro_Toni | Akarere |\n|---|---|---|\n| Ibigori | 45200 | Musanze |\n| Ibishyimbo | 12300 | Musanze |\n| Ibirayi | 89000 | Musanze |",
        "question": "Ni iki gifite Umusaruro_Toni nke cyane?",
        "answer": "Ibishyimbo",
        "explanation": "Ibishyimbo has the lowest Umusaruro_Toni (12300).",
    },
    {
        "table": "| Umwaka | Umusaruro_Miliyari_Frw | Impinduka_% |\n|---|---|---|\n| 2020 | 1100.2 | +3.1 |\n| 2021 | 1189.5 | +8.1 |\n| 2022 | 1209.3 | +1.7 |",
        "question": "Ni iki gifite Impinduka_% ntoya muri bose?",
        "answer": "2022",
        "explanation": "2022 has the lowest Impinduka_% (+1.7).",
    },
]


def make_fewshot_prompt(table_md: str, question: str) -> str:
    examples = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples += (
            f"Table:\n{ex['table']}\n\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n\n"
        )
    return (
        f"You are answering questions about data tables written in Kinyarwanda.\n"
        f"The answer is always the NAME of an entity from the first column — NOT a number.\n"
        f"Identify which entity has the highest or lowest value in the relevant column.\n\n"
        f"Examples:\n\n{examples}"
        f"Now answer the following:\n\n"
        f"Table:\n{table_md}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def call_openai(prompt: str, api_key: str, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": "gpt-4o", "max_tokens": 64, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}
            r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [API error] {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--out', default=os.path.join(DATA_DIR, 'llm_agg_fewshot_results.json'))
    args = parser.parse_args()

    random.seed(42)
    with open(TABQA_FILE) as f:
        all_items = json.load(f)
    random.shuffle(all_items)
    dev_items = [x for x in all_items[int(0.8 * len(all_items)):]
                 if x.get('answer_type') == 'aggregation']
    print(f'[FewShot-Agg] {len(dev_items)} aggregation dev items')

    correct, total, skipped = 0, 0, 0
    records = []

    for i, item in enumerate(dev_items):
        csv_path = os.path.join(CSV_DIR, item['table_file'])
        if not os.path.exists(csv_path):
            skipped += 1; continue
        cells = serialize_csv(csv_path)
        if not cells:
            skipped += 1; continue
        gold_rc = find_gold_cell(cells, item['answer'], item['question'])
        if gold_rc is None:
            skipped += 1; continue
        table_md = csv_to_markdown(csv_path)
        if not table_md:
            skipped += 1; continue

        prompt = make_fewshot_prompt(table_md, item['question'])
        pred   = call_openai(prompt, args.api_key)
        if pred is None:
            skipped += 1; continue

        hit = int(em_match(pred, item['answer']))
        correct += hit; total += 1
        records.append({'question': item['question'], 'gold': item['answer'],
                        'pred': pred, 'hit': hit})
        print(f'  [{i+1}/{len(dev_items)}] gold={item["answer"]!r:20s} pred={pred!r:20s} hit={hit}')

    em = correct / total if total else 0.0
    print(f'\n[FewShot-Agg] EM = {em:.4f} ({correct}/{total}, {skipped} skipped)')
    with open(args.out, 'w') as f:
        json.dump({'em': round(em, 4), 'correct': correct, 'total': total,
                   'skipped': skipped, 'records': records}, f, indent=2, ensure_ascii=False)
    print(f'[FewShot-Agg] Saved to {args.out}')


if __name__ == '__main__':
    main()
