# TabuLM — tabular dataset and data loading utilities
# Extends KinyaBERT's data pipeline with:
#   - TabularParsedToken: ParsedToken + (row_id, col_id, cell_type)
#   - process_tabular_sentence: masking aware of cell boundaries
#   - TabularKBCorpusDataset: loads CSV tables, no libkinlp.so required
#   - tabular_collate_wrapper: extends morpho_seq_collate_wrapper
#   - tabulm_model_forward: unpacks tabular batch for the model

from __future__ import print_function, division

import math
import os
import random
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import youtokentome as yttm
from torch.utils.data import Dataset

from kinyabert_utils import time_now
from morpho_data_loaders import (
    KBVocab, AffixSetVocab, ParsedToken, morpho_seq_collate_wrapper
)
from tabular_serializer import (
    CellType, NUM_CELL_TYPES, TableCell,
    serialize_csv, table_cells_to_text, WordMeta
)
from morpho_stub import parse_text_stub

warnings.filterwarnings("ignore")


# ── TabularParsedToken ────────────────────────────────────────────────────────

class TabularParsedToken(ParsedToken):
    """ParsedToken extended with table-grid coordinates."""

    def __init__(self, row_id: int, col_id: int, cell_type: int, **kwargs):
        super().__init__(**kwargs)
        self.row_id = row_id
        self.col_id = col_id
        self.cell_type = cell_type


# ── process_tabular_sentence ─────────────────────────────────────────────────

def process_tabular_sentence(
    args,
    parsed_tokens_list: List[TabularParsedToken],
    add_cls: bool,
    kv: KBVocab,
    affix_set_vocab: AffixSetVocab,
    mcr_masked_cells: set,          # set of (row_id, col_id) selected for MCR masking
    ctp_col_labels: dict,           # {col_id: cell_type_int} for CTP targets
):
    """
    Extended version of process_parsed_sentence that additionally:
      - Tracks row_id / col_id / cell_type per sequence position
      - Applies MCR (cell-level) masking in addition to word-level masking
      - Collects CTP prediction targets (column type labels)
    """
    pos_tags = []
    stems = []
    afsets = [] if args.use_afsets else None
    affixes = []
    tokens_lengths = []
    row_ids = []
    col_ids = []
    cell_types = []

    predicted_stems = []
    predicted_afsets = [] if args.use_afsets else None
    predicted_affixes = [] if args.predict_affixes else None
    predicted_tokens_idx = []
    predicted_tokens_affixes_idx = [] if args.predict_affixes else None
    predicted_tokens_affixes_lengths = [] if args.predict_affixes else None

    # MCR-specific bookkeeping
    mcr_predicted_stems = []
    mcr_predicted_tokens_idx = []

    def _add_special(vocab_key: str, r: int = 0, c: int = 0, ct: int = 0):
        pos_tags.append(kv.pos_tag_vocab[vocab_key])
        stems.append(kv.reduced_stem_vocab[vocab_key])
        if args.use_afsets:
            afsets.append(affix_set_vocab.affix_set_to_idx(vocab_key))
        tokens_lengths.append(0)
        row_ids.append(r)
        col_ids.append(c)
        cell_types.append(ct)

    if add_cls:
        _add_special('<CLS>')

    if not parsed_tokens_list:
        _add_special('<SEP>')
        return _pack_result(
            pos_tags, stems, afsets, affixes, tokens_lengths,
            row_ids, col_ids, cell_types,
            predicted_stems, predicted_afsets, predicted_affixes,
            predicted_tokens_idx, predicted_tokens_affixes_idx,
            predicted_tokens_affixes_lengths,
            mcr_predicted_stems, mcr_predicted_tokens_idx,
        )

    for ptoken in parsed_tokens_list:
        is_mcr_masked = (ptoken.row_id, ptoken.col_id) in mcr_masked_cells
        r, c, ct = ptoken.row_id, ptoken.col_id, ptoken.cell_type

        for sidx in ptoken.stem_idx:
            unchanged = True
            predict_word = False
            rval = random.random()

            if is_mcr_masked:
                # MCR: mask the entire cell unconditionally
                pos_tags.append(kv.pos_tag_vocab['<MSK>'])
                stems.append(kv.reduced_stem_vocab['<MSK>'])
                if args.use_afsets:
                    afsets.append(affix_set_vocab.affix_set_to_idx('<MSK>'))
                tokens_lengths.append(0)
                mcr_predicted_stems.append(kv.mapped_stem_vocab_idx[sidx])
                mcr_predicted_tokens_idx.append(len(tokens_lengths) - 1)
                unchanged = False

            else:
                # Standard word-level masking (15%)
                if rval <= 0.15:
                    predict_word = True
                    rval /= 0.15
                    if rval < 0.8:
                        unchanged = False
                        pos_tags.append(kv.pos_tag_vocab['<MSK>'])
                        stems.append(kv.reduced_stem_vocab['<MSK>'])
                        if args.use_afsets:
                            afsets.append(affix_set_vocab.affix_set_to_idx('<MSK>'))
                        if (rval / 0.8) < 0.3:
                            affixes.extend(ptoken.affixes_idx)
                            tokens_lengths.append(len(ptoken.affixes_idx))
                        else:
                            tokens_lengths.append(0)
                    elif rval < 0.9:
                        unchanged = False
                        rnd_pos = random.randint(kv.pos_tag_vocab['<UNK>'], len(kv.pos_tag_vocab) - 1)
                        rnd_stem = random.randint(kv.reduced_stem_vocab['<UNK>'], len(kv.reduced_stem_vocab) - 1)
                        pos_tags.append(rnd_pos)
                        stems.append(rnd_stem)
                        if args.use_afsets:
                            afsets.append(affix_set_vocab.random_idx())
                        tokens_lengths.append(0)

            if unchanged:
                pos_tags.append(ptoken.pos_tag_idx)
                stems.append(kv.mapped_stem_vocab_idx[sidx])
                if args.use_afsets:
                    afsets.append(affix_set_vocab.affix_set_to_idx(ptoken.affix_set_key()))
                affixes.extend(ptoken.affixes_idx)
                tokens_lengths.append(len(ptoken.affixes_idx))

            row_ids.append(r)
            col_ids.append(c)
            cell_types.append(ct)

            if predict_word:
                predicted_stems.append(kv.mapped_stem_vocab_idx[sidx])
                predicted_tokens_idx.append(len(tokens_lengths) - 1)
                if args.use_afsets:
                    predicted_afsets.append(affix_set_vocab.affix_set_to_idx(ptoken.affix_set_key()))
                if args.predict_affixes:
                    predicted_affixes.extend(ptoken.affixes_idx)
                    if ptoken.affixes_idx:
                        predicted_tokens_affixes_idx.append(len(predicted_tokens_idx) - 1)
                        predicted_tokens_affixes_lengths.append(len(ptoken.affixes_idx))

    return _pack_result(
        pos_tags, stems, afsets, affixes, tokens_lengths,
        row_ids, col_ids, cell_types,
        predicted_stems, predicted_afsets, predicted_affixes,
        predicted_tokens_idx, predicted_tokens_affixes_idx,
        predicted_tokens_affixes_lengths,
        mcr_predicted_stems, mcr_predicted_tokens_idx,
    )


def _pack_result(pos_tags, stems, afsets, affixes, tokens_lengths,
                 row_ids, col_ids, cell_types,
                 predicted_stems, predicted_afsets, predicted_affixes,
                 predicted_tokens_idx, predicted_tokens_affixes_idx,
                 predicted_tokens_affixes_lengths,
                 mcr_predicted_stems, mcr_predicted_tokens_idx):
    return (
        pos_tags, stems, afsets, affixes, tokens_lengths,
        row_ids, col_ids, cell_types,
        predicted_stems, predicted_afsets, predicted_affixes,
        predicted_tokens_idx, predicted_tokens_affixes_idx,
        predicted_tokens_affixes_lengths,
        mcr_predicted_stems, mcr_predicted_tokens_idx,
    )


# ── gather_tabular_itemized_data ─────────────────────────────────────────────

def gather_tabular_itemized_data(
    args,
    table_cells_list: List[List[TableCell]],  # list of tables, each a list of cells
    kb_vocab: KBVocab,
    affix_set_vocab: AffixSetVocab,
    bpe: yttm.BPE,
    max_seq_len: int,
    max_batch_items: int,
    mcr_cell_mask_rate: float = 0.15,
    ctp_mask_rate: float = 0.50,
):
    """
    Builds a list of itemized training data items from a list of tables.
    Each item is a tuple compatible with tabular_collate_wrapper.
    """
    itemized_data = []
    table_idx = 0

    seq_pos_tags, seq_stems, seq_afsets, seq_affixes, seq_tokens_lengths = [], [], [], [], []
    seq_row_ids, seq_col_ids, seq_cell_types = [], [], []
    seq_predicted_stems = []
    seq_predicted_afsets = [] if args.use_afsets else None
    seq_predicted_affixes = [] if args.predict_affixes else None
    seq_predicted_tokens_idx = []
    seq_predicted_tokens_affixes_idx = [] if args.predict_affixes else None
    seq_predicted_tokens_affixes_lengths = [] if args.predict_affixes else None
    seq_mcr_predicted_stems, seq_mcr_predicted_tokens_idx = [], []
    seq_ctp_labels: List[Tuple[int, int]] = []   # (position_in_sequence, cell_type_label)
    seq_rrp_pair: Optional[Tuple[int, int, int]] = None  # (rowA_start, rowB_start, label)

    add_cls = True
    random.shuffle(table_cells_list)

    for cells in table_cells_list:
        if not cells:
            continue

        text, word_meta = table_cells_to_text(cells)
        parsed_tokens_raw = parse_text_stub(text, kb_vocab, bpe)

        if len(parsed_tokens_raw) != len(word_meta):
            # Length mismatch can happen with empty cells — skip
            continue

        # Attach table coordinates to each ParsedToken
        tabular_tokens: List[TabularParsedToken] = []
        for pt, (r, c, ct) in zip(parsed_tokens_raw, word_meta):
            tp = TabularParsedToken(
                row_id=r, col_id=c, cell_type=ct,
                surface_form=pt.surface_form,
                decode_prob=pt.decode_prob,
                tf_idf=pt.tf_idf,
                pos_tag_id=pt.pos_tag_idx,
                stem_ids=pt.stem_idx,
            )
            tp.morpho_slots_idx = pt.morpho_slots_idx
            tp.affixes_idx = pt.affixes_idx
            tabular_tokens.append(tp)

        # Choose MCR cells to mask (15% of unique cells by (row_id, col_id))
        unique_cells = set((t.row_id, t.col_id) for t in tabular_tokens if t.row_id > 0)
        num_mcr = max(1, int(mcr_cell_mask_rate * len(unique_cells)))
        mcr_masked_cells = set(random.sample(sorted(unique_cells), min(num_mcr, len(unique_cells))))

        # CTP: determine column type labels (majority vote per column)
        col_type_votes: dict = {}
        for t in tabular_tokens:
            if t.row_id > 1 and t.col_id > 0 and t.cell_type != int(CellType.PAD):
                col_type_votes.setdefault(t.col_id, []).append(t.cell_type)

        ctp_labels_for_table = {}
        for col_id, votes in col_type_votes.items():
            from collections import Counter
            majority = Counter(votes).most_common(1)[0][0]
            ctp_labels_for_table[col_id] = majority

        (pos_tags, stems, afsets, affixes, tokens_lengths,
         row_ids, col_ids, cell_types,
         predicted_stems, predicted_afsets, predicted_affixes,
         predicted_tokens_idx, predicted_tokens_affixes_idx,
         predicted_tokens_affixes_lengths,
         mcr_predicted_stems, mcr_predicted_tokens_idx) = process_tabular_sentence(
            args, tabular_tokens, add_cls, kb_vocab, affix_set_vocab,
            mcr_masked_cells, ctp_labels_for_table,
        )
        add_cls = False

        if (len(seq_tokens_lengths) + len(tokens_lengths)) > max_seq_len:
            # Flush existing buffer (only if non-empty)
            if seq_tokens_lengths:
                item = _pack_tabular_item(
                    max_seq_len,
                    seq_pos_tags, seq_stems, seq_afsets, seq_affixes, seq_tokens_lengths,
                    seq_row_ids, seq_col_ids, seq_cell_types,
                    seq_predicted_stems, seq_predicted_afsets, seq_predicted_affixes,
                    seq_predicted_tokens_idx, seq_predicted_tokens_affixes_idx,
                    seq_predicted_tokens_affixes_lengths,
                    seq_mcr_predicted_stems, seq_mcr_predicted_tokens_idx,
                    seq_ctp_labels,
                )
                itemized_data.append(item)
                if len(itemized_data) >= max_batch_items:
                    return itemized_data

            # Reset buffer then fall through to add current tokens below
            seq_pos_tags, seq_stems, seq_afsets, seq_affixes, seq_tokens_lengths = [], [], [], [], []
            seq_row_ids, seq_col_ids, seq_cell_types = [], [], []
            seq_predicted_stems = []
            seq_predicted_afsets = [] if args.use_afsets else None
            seq_predicted_affixes = [] if args.predict_affixes else None
            seq_predicted_tokens_idx = []
            seq_predicted_tokens_affixes_idx = [] if args.predict_affixes else None
            seq_predicted_tokens_affixes_lengths = [] if args.predict_affixes else None
            seq_mcr_predicted_stems, seq_mcr_predicted_tokens_idx = [], []
            seq_ctp_labels = []
            add_cls = True
            # Truncate current table to max_seq_len before adding to fresh buffer
            tokens_lengths = tokens_lengths[:max_seq_len]
            pos_tags = pos_tags[:max_seq_len]
            stems = stems[:max_seq_len]
            row_ids = row_ids[:max_seq_len]
            col_ids = col_ids[:max_seq_len]
            cell_types = cell_types[:max_seq_len]
            if afsets is not None:
                afsets = afsets[:max_seq_len]
            # Re-compute affix flat list up to the truncated tokens
            affix_end = sum(tokens_lengths)
            affixes = affixes[:affix_end]
            # Trim prediction indices that are out of range
            predicted_tokens_idx = [i for i in predicted_tokens_idx if i < max_seq_len]
            predicted_stems = predicted_stems[:len(predicted_tokens_idx)]
            if predicted_afsets is not None:
                predicted_afsets = predicted_afsets[:len(predicted_tokens_idx)]
            if predicted_tokens_affixes_idx is not None:
                predicted_tokens_affixes_idx = [i for i in predicted_tokens_affixes_idx if i < len(predicted_tokens_idx)]
                predicted_tokens_affixes_lengths = predicted_tokens_affixes_lengths[:len(predicted_tokens_affixes_idx)] if predicted_tokens_affixes_lengths else []
                predicted_affixes = predicted_affixes[:sum(predicted_tokens_affixes_lengths)] if predicted_affixes else []
            mcr_predicted_tokens_idx = [i for i in mcr_predicted_tokens_idx if i < max_seq_len]
            mcr_predicted_stems = mcr_predicted_stems[:len(mcr_predicted_tokens_idx)]

        offset = len(seq_predicted_tokens_idx)
        if args.predict_affixes and predicted_tokens_affixes_idx:
            seq_predicted_tokens_affixes_idx.extend(
                [offset + i for i in predicted_tokens_affixes_idx]
            )
        seq_predicted_tokens_idx.extend(
            [len(seq_tokens_lengths) + i for i in predicted_tokens_idx]
        )

        # CTP: record (absolute_position_of_first_header_token_in_col, label)
        header_positions = {}
        for abs_i, (r, c, ct_int) in enumerate(zip(row_ids, col_ids, cell_types)):
            if ct_int == int(CellType.HEADER) and c not in header_positions:
                header_positions[c] = len(seq_tokens_lengths) + abs_i
        for col_id, label in ctp_labels_for_table.items():
            if random.random() < ctp_mask_rate and col_id in header_positions:
                seq_ctp_labels.append((header_positions[col_id], label))

        seq_pos_tags.extend(pos_tags)
        seq_stems.extend(stems)
        if args.use_afsets and afsets is not None:
            seq_afsets.extend(afsets)
        seq_affixes.extend(affixes)
        seq_tokens_lengths.extend(tokens_lengths)
        seq_row_ids.extend(row_ids)
        seq_col_ids.extend(col_ids)
        seq_cell_types.extend(cell_types)
        seq_predicted_stems.extend(predicted_stems)
        if args.use_afsets and predicted_afsets is not None:
            seq_predicted_afsets.extend(predicted_afsets)
        if args.predict_affixes and predicted_affixes is not None:
            seq_predicted_affixes.extend(predicted_affixes)
            if predicted_tokens_affixes_lengths:
                seq_predicted_tokens_affixes_lengths.extend(predicted_tokens_affixes_lengths)
        seq_mcr_predicted_stems.extend(mcr_predicted_stems)
        seq_mcr_predicted_tokens_idx.extend(
            [len(seq_tokens_lengths) - len(tokens_lengths) + i for i in mcr_predicted_tokens_idx]
        )

    return itemized_data


def _pack_tabular_item(
    max_seq_len,
    pos_tags, stems, afsets, affixes, tokens_lengths,
    row_ids, col_ids, cell_types,
    predicted_stems, predicted_afsets, predicted_affixes,
    predicted_tokens_idx, predicted_tokens_affixes_idx,
    predicted_tokens_affixes_lengths,
    mcr_predicted_stems, mcr_predicted_tokens_idx,
    ctp_labels,
):
    return (
        max_seq_len,
        # Original KinyaBERT fields
        None,           # rel_pos_arr (unused for tabular)
        pos_tags, stems, afsets, affixes, tokens_lengths,
        predicted_stems, predicted_afsets, predicted_affixes,
        predicted_tokens_idx, predicted_tokens_affixes_idx,
        predicted_tokens_affixes_lengths,
        # Tabular-specific fields
        row_ids, col_ids, cell_types,
        mcr_predicted_stems, mcr_predicted_tokens_idx,
        ctp_labels,
    )


# ── TabularKBCorpusDataset ────────────────────────────────────────────────────

class TabularKBCorpusDataset(Dataset):
    """
    Dataset that loads CSV tables from a directory and builds
    tabular training items. Compatible with tabular_collate_wrapper.
    """

    def __init__(
        self,
        args,
        kb_vocab: KBVocab,
        affix_set_vocab: AffixSetVocab,
        bpe_encoder: yttm.BPE,
        csv_dir: str,
        max_batch_items: int,
        max_seq_len: int = 512,
        max_rows: int = 64,
        max_cols: int = 24,
        rank: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.max_batch_items = max_batch_items

        # Discover CSV files
        csv_files = [
            os.path.join(csv_dir, f)
            for f in os.listdir(csv_dir)
            if f.lower().endswith('.csv')
        ]
        if not csv_files:
            raise FileNotFoundError(f'No CSV files found in {csv_dir}')

        if rank == 0:
            print(time_now(), f'Loading {len(csv_files)} CSV tables from {csv_dir}')

        # Load all tables
        all_tables: List[List[TableCell]] = []
        for fp in csv_files:
            cells = serialize_csv(fp, max_rows=max_rows, max_cols=max_cols)
            if cells:
                all_tables.append(cells)

        if rank == 0:
            print(time_now(), f'{len(all_tables)} tables loaded successfully')

        self.itemized_data = gather_tabular_itemized_data(
            args, all_tables, kb_vocab, affix_set_vocab, bpe_encoder,
            max_seq_len=max_seq_len,
            max_batch_items=max_batch_items,
        )

        if rank == 0:
            print(time_now(), f'{len(self.itemized_data)} training items prepared')

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]


# ── tabular_collate_wrapper ───────────────────────────────────────────────────

def tabular_collate_wrapper(batch_items):
    """
    Collate function for TabularKBCorpusDataset.
    Extends morpho_seq_collate_wrapper with tabular fields.
    """
    batch_input_sequence_lengths = []
    batch_pos_tags, batch_stems, batch_afsets = [], [], []
    batch_affixes, batch_tokens_lengths = [], []
    batch_predicted_stems, batch_predicted_afsets = [], []
    batch_predicted_affixes = []
    batch_predicted_tokens_idx = []
    batch_predicted_tokens_affixes_idx = []
    batch_predicted_tokens_affixes_lengths = []

    batch_row_ids, batch_col_ids, batch_cell_types = [], [], []
    batch_mcr_predicted_stems, batch_mcr_predicted_tokens_idx = [], []
    batch_ctp_labels: List[Tuple[int, int]] = []

    for bidx, item in enumerate(batch_items):
        (max_seq_len, _rel_pos_arr,
         pos_tags, stems, afsets, affixes, tokens_lengths,
         predicted_stems, predicted_afsets, predicted_affixes,
         predicted_tokens_idx, predicted_tokens_affixes_idx,
         predicted_tokens_affixes_lengths,
         row_ids, col_ids, cell_types,
         mcr_predicted_stems, mcr_predicted_tokens_idx,
         ctp_labels) = item

        # Offset affix-prediction indices
        if predicted_tokens_affixes_idx is not None:
            batch_predicted_tokens_affixes_idx.extend(
                [(len(batch_predicted_tokens_idx) + t) for t in predicted_tokens_affixes_idx]
            )

        batch_predicted_tokens_idx.extend(
            [(t, len(batch_input_sequence_lengths)) for t in predicted_tokens_idx]
        )

        # Offset MCR indices the same way
        batch_mcr_predicted_tokens_idx.extend(
            [(t, len(batch_input_sequence_lengths)) for t in mcr_predicted_tokens_idx]
        )

        # CTP labels: offset the absolute position
        seq_start = sum(batch_input_sequence_lengths)  # not directly, but captured below
        batch_ctp_labels.extend(ctp_labels)

        batch_pos_tags.extend(pos_tags)
        batch_stems.extend(stems)
        if afsets is not None:
            batch_afsets.extend(afsets)
        batch_affixes.extend(affixes)
        batch_tokens_lengths.extend(tokens_lengths)
        batch_predicted_stems.extend(predicted_stems)
        if predicted_afsets is not None:
            batch_predicted_afsets.extend(predicted_afsets)
        if predicted_affixes is not None:
            batch_predicted_affixes.extend(predicted_affixes)
        if predicted_tokens_affixes_lengths is not None:
            batch_predicted_tokens_affixes_lengths.extend(predicted_tokens_affixes_lengths)

        batch_row_ids.extend(row_ids)
        batch_col_ids.extend(col_ids)
        batch_cell_types.extend(cell_types)
        batch_mcr_predicted_stems.extend(mcr_predicted_stems)
        batch_input_sequence_lengths.append(len(tokens_lengths))

    return (
        batch_input_sequence_lengths,
        None,  # rel_pos_arr unused
        batch_pos_tags, batch_stems, batch_afsets,
        batch_affixes, batch_tokens_lengths,
        batch_predicted_stems, batch_predicted_afsets, batch_predicted_affixes,
        batch_predicted_tokens_idx,
        batch_predicted_tokens_affixes_idx,
        batch_predicted_tokens_affixes_lengths,
        # Tabular additions
        batch_row_ids, batch_col_ids, batch_cell_types,
        batch_mcr_predicted_stems, batch_mcr_predicted_tokens_idx,
        batch_ctp_labels,
    )


# ── tabulm_model_forward ──────────────────────────────────────────────────────

def tabulm_model_forward(args, data_item, model, device, tot_num_affixes):
    """
    Unpacks a tabular data_item and runs one forward pass through TabuLM.
    Returns (total_loss, stem_loss, afset_loss, affix_loss, mcr_loss, ctp_loss, rrp_loss).
    """
    (batch_input_sequence_lengths,
     _rel_pos_arr,
     batch_pos_tags, batch_stems, batch_afsets,
     batch_affixes, batch_tokens_lengths,
     batch_predicted_stems, batch_predicted_afsets, batch_predicted_affixes,
     batch_predicted_tokens_idx,
     batch_predicted_tokens_affixes_idx,
     batch_predicted_tokens_affixes_lengths,
     batch_row_ids, batch_col_ids, batch_cell_types,
     batch_mcr_predicted_stems, batch_mcr_predicted_tokens_idx,
     batch_ctp_labels) = data_item

    pos_tags = torch.tensor(batch_pos_tags, dtype=torch.long).to(device)
    stems = torch.tensor(batch_stems, dtype=torch.long).to(device)
    afsets = torch.tensor(batch_afsets, dtype=torch.long).to(device) if args.use_afsets and batch_afsets else None
    affixes = torch.tensor(batch_affixes, dtype=torch.long).to(device)
    row_ids = torch.tensor(batch_row_ids, dtype=torch.long).to(device)
    col_ids = torch.tensor(batch_col_ids, dtype=torch.long).to(device)
    cell_types = torch.tensor(batch_cell_types, dtype=torch.long).to(device)

    max_seq = max(batch_input_sequence_lengths)
    predicted_tokens_idx = torch.tensor(
        [s * max_seq + t for t, s in batch_predicted_tokens_idx], dtype=torch.long
    ).to(device)

    predicted_tokens_affixes_idx = (
        torch.tensor(batch_predicted_tokens_affixes_idx, dtype=torch.long).to(device)
        if args.predict_affixes and batch_predicted_tokens_affixes_idx
        else None
    )

    predicted_affixes_prob = None
    if args.predict_affixes and batch_predicted_affixes and predicted_tokens_affixes_idx is not None:
        from itertools import accumulate
        lengths = batch_predicted_tokens_affixes_lengths
        affix_groups = []
        prev = 0
        for ln in lengths:
            sub = batch_predicted_affixes[prev: prev + ln]
            vec = torch.zeros(tot_num_affixes)
            for idx in sub:
                if 0 < idx < tot_num_affixes:
                    vec[idx] = 1.0
            affix_groups.append(vec)
            prev += ln
        if affix_groups:
            predicted_affixes_prob = torch.stack(affix_groups).to(device)
            predicted_affixes_prob = predicted_affixes_prob / (predicted_affixes_prob.sum(1, keepdim=True).clamp(min=1.0))

    predicted_stems = torch.tensor(batch_predicted_stems, dtype=torch.long).to(device)
    predicted_afsets = (
        torch.tensor(batch_predicted_afsets, dtype=torch.long).to(device)
        if args.use_afsets and batch_predicted_afsets else None
    )

    # MCR indices and targets
    mcr_tokens_idx = (
        torch.tensor([s * max_seq + t for t, s in batch_mcr_predicted_tokens_idx], dtype=torch.long).to(device)
        if batch_mcr_predicted_tokens_idx else None
    )
    mcr_stems = (
        torch.tensor(batch_mcr_predicted_stems, dtype=torch.long).to(device)
        if batch_mcr_predicted_stems else None
    )

    # CTP indices and labels
    ctp_positions = (
        torch.tensor([pos for pos, _ in batch_ctp_labels], dtype=torch.long).to(device)
        if batch_ctp_labels else None
    )
    ctp_labels_t = (
        torch.tensor([lbl for _, lbl in batch_ctp_labels], dtype=torch.long).to(device)
        if batch_ctp_labels else None
    )

    # Ablation flags: disable objectives or structural embeddings
    if getattr(args, 'no_mcr', False):
        mcr_tokens_idx = None
        mcr_stems = None
    if getattr(args, 'no_ctp', False):
        ctp_positions = None
        ctp_labels_t = None
    if getattr(args, 'no_tabular_emb', False):
        # padding_idx=0 → embedding lookup returns zeros, so zeroing IDs kills all structural signal
        row_ids = torch.zeros_like(row_ids)
        col_ids = torch.zeros_like(col_ids)
        cell_types = torch.zeros_like(cell_types)

    return model(
        args,
        rel_pos_arr=None,
        tokens_lengths=batch_tokens_lengths,
        input_sequence_lengths=batch_input_sequence_lengths,
        pos_tags=pos_tags,
        stems=stems,
        afsets=afsets,
        affixes=affixes,
        row_ids=row_ids,
        col_ids=col_ids,
        cell_types=cell_types,
        predicted_tokens_idx=predicted_tokens_idx,
        predicted_tokens_affixes_idx=predicted_tokens_affixes_idx,
        predicted_stems=predicted_stems,
        predicted_afsets=predicted_afsets,
        predicted_affixes_prob=predicted_affixes_prob,
        mcr_tokens_idx=mcr_tokens_idx,
        mcr_stems=mcr_stems,
        ctp_positions=ctp_positions,
        ctp_labels=ctp_labels_t,
    )
