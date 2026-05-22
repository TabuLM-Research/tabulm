# TabuLM — Table-Aware Morphologically-Rich Language Model
# Extends KinyaBERT_MorphoEncoder with:
#   1. Row / column / cell-type embeddings (additive, same dim as seq_tr_d_model)
#   2. Table structure attention bias (same-row, same-col, header bias per head)
#   3. MCR head  — Masked Cell Recovery (reuses stem decoder)
#   4. CTP head  — Column Type Prediction (4-class linear)
#
# Architecture change summary vs KinyaBERT:
#   seq_tr_d_model   unchanged (768 with default args)
#   attn_bias        = position_bias + table_structure_bias  (injected at morpho_transformer.py:333)
#   input to seq-tr  = morpho_output + stem_embed + row_embed + col_embed + cell_type_embed  (additive)
#   new losses       = MCR (NLL) + CTP (NLL)

from __future__ import print_function, division

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd
from torch.nn.utils.rnn import pad_sequence

from morpho_transformer import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from morpho_model import (
    gelu, init_bert_params, BertLayerNorm,
    KinyaBERT_MorphoEncoder,
)
from tabular_serializer import NUM_CELL_TYPES

warnings.filterwarnings("ignore")

# ── Maximum table dimensions supported ───────────────────────────────────────
MAX_ROWS = 128
MAX_COLS = 64


# ── Column type label set for CTP ────────────────────────────────────────────
# We predict: NUMERIC(0) / TEXT(1) / CATEGORICAL(2) / DATE(3)
# (HEADER is the header row itself, not a prediction target)
CTP_NUM_LABELS = 4
CTP_TYPE_MAP = {2: 0, 3: 1, 4: 2, 5: 3}   # CellType int → CTP label index


class TabuLM_Encoder(KinyaBERT_MorphoEncoder):
    """
    KinyaBERT_MorphoEncoder extended with tabular structure awareness.

    Three new components are added to the sequence transformer tier:

    (a) Additive embeddings: row_embedding, col_embedding, cell_type_embedding
        — same dimension as seq_tr_d_model, added element-wise to each word
          vector before it enters the sequence transformer (same way BERT adds
          absolute position embeddings).

    (b) Table structure attention bias: a per-head learned scalar that is added
        to the pre-softmax attention weights whenever two tokens share the same
        row, same column, or when either token is in the header row.
        Injected at morpho_transformer.py line 333 on top of position bias.

    (c) MCR prediction head: reuses the inherited stem decoder to score
        cell-level masked tokens (no new parameters needed).

    (d) CTP prediction head: a 2-layer linear that maps from seq_tr_d_model
        to CTP_NUM_LABELS classes for column type prediction.
    """

    def __init__(self, *args,
                 max_rows: int = MAX_ROWS,
                 max_cols: int = MAX_COLS,
                 **kwargs):
        super().__init__(*args, **kwargs)

        d = self.seq_tr_d_model

        # (a) Tabular embeddings (additive, padding_idx=0 is the PAD/special token slot)
        self.row_embedding = nn.Embedding(max_rows + 2, d, padding_idx=0)
        self.col_embedding = nn.Embedding(max_cols + 2, d, padding_idx=0)
        self.cell_type_embedding = nn.Embedding(NUM_CELL_TYPES + 1, d, padding_idx=0)

        # (b) Table structure bias: one scalar per attention head, for each relation
        self.row_attn_bias = nn.Parameter(torch.zeros(self.seq_tr_nhead))
        self.col_attn_bias = nn.Parameter(torch.zeros(self.seq_tr_nhead))
        self.header_attn_bias = nn.Parameter(torch.zeros(self.seq_tr_nhead))

        # (d) CTP head
        self.ctp_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            BertLayerNorm(d // 2, eps=1e-6),
            nn.Linear(d // 2, CTP_NUM_LABELS),
        )

        self.apply(init_bert_params)

    # ── Position bias override (safe for rel_pos_arr=None) ───────────────────

    def get_position_attn_bias(self, rel_pos_arr, seq_len, batch_size, device):
        # When rel_pos_arr is None (no POS-aware dictionary available),
        # skip that term rather than crashing.
        tupe_bias = self.get_tupe_rel_pos_bias(seq_len, device) if self.use_tupe_rel_pos_bias else None

        weight = self.pos_ln(self.pos.weight[:seq_len + 1, :])
        pos_q = self.pos_q_linear(weight).view(seq_len + 1, self.seq_tr_nhead, -1).transpose(0, 1) * self.pos_scaling
        pos_k = self.pos_k_linear(weight).view(seq_len + 1, self.seq_tr_nhead, -1).transpose(0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))

        cls_2_other = abs_pos_bias[:, 0, 0]
        other_2_cls = abs_pos_bias[:, 1, 1]
        abs_pos_bias = abs_pos_bias[:, 1:, 1:]
        abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
        abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)

        if tupe_bias is not None:
            abs_pos_bias = abs_pos_bias + tupe_bias

        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, seq_len, seq_len)

        if self.use_pos_aware_rel_pos_bias and rel_pos_arr is not None:
            abs_pos_bias = abs_pos_bias + self.get_pos_aware_rel_pos_bias(rel_pos_arr, seq_len)

        return abs_pos_bias

    # ── Table structure bias ──────────────────────────────────────────────────

    def get_table_structure_bias(
        self,
        row_ids_padded: torch.Tensor,   # (N, S) — padded row ids
        col_ids_padded: torch.Tensor,   # (N, S)
        seq_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Returns (N*H, S, S) tensor to be added to the position attn_bias.
        row_ids_padded == 1 is the header row.
        """
        H = self.seq_tr_nhead
        # (N, S, S) boolean: do positions i and j share the same row / col?
        row_match = (row_ids_padded.unsqueeze(2) == row_ids_padded.unsqueeze(1)).float()
        col_match = (col_ids_padded.unsqueeze(2) == col_ids_padded.unsqueeze(1)).float()
        # Header: either token i or token j is in the header row (row_id == 1)
        is_header = (row_ids_padded == 1)   # (N, S)
        header_match = (is_header.unsqueeze(2) | is_header.unsqueeze(1)).float()  # (N, S, S)

        # Broadcast head-specific scalars: (H,) → (H, 1, 1)
        rb = self.row_attn_bias.view(H, 1, 1)
        cb = self.col_attn_bias.view(H, 1, 1)
        hb = self.header_attn_bias.view(H, 1, 1)

        # (N, 1, S, S) * (H, 1, 1) broadcast → (N, H, S, S)
        bias = (
            rb * row_match.unsqueeze(1) +
            cb * col_match.unsqueeze(1) +
            hb * header_match.unsqueeze(1)
        )
        return bias.reshape(batch_size * H, seq_len, seq_len).contiguous()

    # ── Forward ───────────────────────────────────────────────────────────────

    @custom_fwd
    def forward(self, args, rel_pos_arr,
                tokens_lengths, input_sequence_lengths,
                pos_tags, stems, afsets, affixes,
                row_ids=None, col_ids=None, cell_types=None):
        """
        All parameters identical to KinyaBERT_MorphoEncoder.forward() plus:
          row_ids, col_ids, cell_types — flat tensors of length L (total tokens).
        """
        device = stems.device

        # ── Tier 1: Morpho transformer (unchanged from KinyaBERT) ─────────────
        x_embed = None
        if self.num_pos_m_embeddings > 0:
            xm_pos1 = self.m1_pos_embedding(pos_tags)
            x_embed = torch.unsqueeze(xm_pos1, 0)
        if self.num_pos_m_embeddings > 1:
            xm_pos2 = self.m2_pos_embedding(pos_tags)
            x_embed = torch.cat((x_embed, torch.unsqueeze(xm_pos2, 0)), 0)
        if self.num_pos_m_embeddings > 2:
            xm_pos3 = self.m3_pos_embedding(pos_tags)
            x_embed = torch.cat((x_embed, torch.unsqueeze(xm_pos3, 0)), 0)
        if self.num_stem_m_embeddings > 0:
            xm_stem = self.m_stem_embedding(stems)
            xm_stem = torch.unsqueeze(xm_stem, 0)
            x_embed = torch.cat((x_embed, xm_stem), 0) if x_embed is not None else xm_stem
        if args.use_afsets and afsets is not None:
            xm_afset = self.m_afset_embedding(afsets)
            xm_afset = torch.unsqueeze(xm_afset, 0)
            x_embed = torch.cat((x_embed, xm_afset), 0) if x_embed is not None else xm_afset

        m_masks_padded = None
        has_morphemes = False
        morpho_input = None

        if args.use_morpho_encoder:
            afx = [a for a in affixes.split(tokens_lengths) if a.numel() > 0]
            afx_padded = pad_sequence(afx, batch_first=False) if afx else torch.empty(0, device=device)
            if afx_padded.nelement() > 0:
                has_morphemes = True
                xm_affix = self.m_affix_embedding(afx_padded)
                x_embed = torch.cat((x_embed, xm_affix), 0) if x_embed is not None else xm_affix
                m_masks = [torch.zeros(x + self.tot_morpho_idx, dtype=torch.bool, device=device)
                           for x in tokens_lengths]
                m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1)

        if x_embed is not None and args.use_morpho_encoder:
            morpho_out = self.morpho_transformer_encoder(x_embed, src_key_padding_mask=m_masks_padded)
            if self.tot_morpho_idx > 0 and self.use_affix_bow_m_embedding:
                heads = morpho_out[:self.tot_morpho_idx]
                bow = torch.sum(morpho_out[self.tot_morpho_idx:], 0, keepdim=True) if has_morphemes \
                    else torch.zeros((1, stems.size(0), self.morpho_dim), device=device)
                morpho_input = torch.cat((heads, bow), 0)
            elif self.tot_morpho_idx > 0:
                morpho_input = morpho_out[:self.tot_morpho_idx]
            elif self.use_affix_bow_m_embedding:
                morpho_input = torch.sum(morpho_out[self.tot_morpho_idx:], 0, keepdim=True) if has_morphemes \
                    else torch.zeros((1, stems.size(0), self.morpho_dim), device=device)
        elif self.use_affix_bow_m_embedding:
            morpho_input = torch.zeros((1, stems.size(0), self.morpho_dim), device=device)

        # ── Build input to sequence transformer ──────────────────────────────
        input_sequences = self.s_stem_embedding(stems)   # (L, stem_dim)

        if morpho_input is not None:
            morpho_input = morpho_input.permute(1, 0, 2).contiguous()
            L = morpho_input.size(0)
            morpho_input = morpho_input.view(L, -1)
            input_sequences = torch.cat((morpho_input, input_sequences), 1)   # (L, seq_tr_d_model)

        lists = input_sequences.split(input_sequence_lengths, 0)
        tr_padded = pad_sequence(lists, batch_first=False)   # (S, N, d)

        seq_len = tr_padded.size(0)
        batch_size = tr_padded.size(1)

        # ── Tabular additive embeddings ───────────────────────────────────────
        row_ids_padded = None
        col_ids_padded = None
        if row_ids is not None:
            row_lists = row_ids.split(input_sequence_lengths, 0)
            row_ids_padded = pad_sequence(row_lists, batch_first=True, padding_value=0)  # (N, S)
            col_lists = col_ids.split(input_sequence_lengths, 0)
            col_ids_padded = pad_sequence(col_lists, batch_first=True, padding_value=0)
            ct_lists = cell_types.split(input_sequence_lengths, 0)
            ct_padded = pad_sequence(ct_lists, batch_first=True, padding_value=0)

            # (N, S, d) → (S, N, d)
            row_embed = self.row_embedding(row_ids_padded).permute(1, 0, 2)
            col_embed = self.col_embedding(col_ids_padded).permute(1, 0, 2)
            ct_embed  = self.cell_type_embedding(ct_padded).permute(1, 0, 2)
            tr_padded = tr_padded + row_embed + col_embed + ct_embed

        # ── Position bias (TUPE + POS-aware) — unchanged ─────────────────────
        abs_pos_bias = self.get_position_attn_bias(rel_pos_arr, seq_len, batch_size, device)

        # ── Table structure bias ──────────────────────────────────────────────
        if row_ids_padded is not None and not getattr(args, 'no_bias', False):
            table_bias = self.get_table_structure_bias(
                row_ids_padded, col_ids_padded, seq_len, batch_size
            )
            abs_pos_bias = abs_pos_bias + table_bias

        # ── Tier 2: Sequence transformer ──────────────────────────────────────
        masks = [torch.zeros(x, dtype=torch.bool, device=device) for x in input_sequence_lengths]
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=1)

        transformer_output = self.seq_transformer_encoder(
            tr_padded,
            attn_bias=abs_pos_bias,
            src_key_padding_mask=masks_padded,
        )   # (S, N, d)

        return transformer_output

    # ── CTP prediction ────────────────────────────────────────────────────────

    def predict_ctp(self, hidden_states: torch.Tensor,
                    ctp_positions: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (S, N, d)
        ctp_positions: (num_ctp,) — flat indices into the (N*S) token space
        Returns: (num_ctp, CTP_NUM_LABELS) logits
        """
        flat = hidden_states.permute(1, 0, 2).reshape(-1, self.seq_tr_d_model)
        selected = torch.index_select(flat, 0, ctp_positions)
        return self.ctp_head(selected)


# ── TabuLM (full model with all prediction heads) ─────────────────────────────

class TabuLM(nn.Module):
    """Full TabuLM pre-training model."""

    def __init__(self, args,
                 num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_pos_m_embeddings, num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias, use_tupe_rel_pos_bias,
                 max_seq_len=512,
                 morpho_dim=128, stem_dim=256,
                 morpho_tr_nhead=4, morpho_tr_nlayers=4,
                 morpho_tr_dim_feedforward=512, morpho_tr_dropout=0.1,
                 morpho_tr_activation='gelu',
                 seq_tr_nhead=12, seq_tr_nlayers=12,
                 seq_tr_dim_feedforward=3072, seq_tr_dropout=0.1,
                 seq_tr_activation='gelu',
                 layernorm_epsilon=1e-6,
                 tupe_rel_pos_bins=32, tupe_max_rel_pos=128,
                 max_rows=MAX_ROWS, max_cols=MAX_COLS):
        super().__init__()

        self.encoder = TabuLM_Encoder(
            args,
            num_stems, num_afsets, num_pos_tags, num_affixes,
            num_rel_pos_dict_size,
            num_pos_m_embeddings, num_stem_m_embeddings,
            use_affix_bow_m_embedding,
            use_pos_aware_rel_pos_bias, use_tupe_rel_pos_bias,
            max_seq_len=max_seq_len,
            morpho_dim=morpho_dim, stem_dim=stem_dim,
            morpho_tr_nhead=morpho_tr_nhead, morpho_tr_nlayers=morpho_tr_nlayers,
            morpho_tr_dim_feedforward=morpho_tr_dim_feedforward,
            morpho_tr_dropout=morpho_tr_dropout, morpho_tr_activation=morpho_tr_activation,
            seq_tr_nhead=seq_tr_nhead, seq_tr_nlayers=seq_tr_nlayers,
            seq_tr_dim_feedforward=seq_tr_dim_feedforward,
            seq_tr_dropout=seq_tr_dropout, seq_tr_activation=seq_tr_activation,
            tupe_rel_pos_bins=tupe_rel_pos_bins, tupe_max_rel_pos=tupe_max_rel_pos,
            max_rows=max_rows, max_cols=max_cols,
        )

        d = self.encoder.seq_tr_d_model

        from morpho_model import MorphoHeadPredictor
        self.head_predictor = MorphoHeadPredictor(
            args,
            stem_embedding_weights=self.encoder.s_stem_embedding.weight,
            afset_embedding_weights=(
                self.encoder.m_afset_embedding.weight
                if hasattr(self.encoder, 'm_afset_embedding') else
                torch.zeros(1, morpho_dim)
            ),
            affix_embedding_weights=(
                self.encoder.m_affix_embedding.weight
                if hasattr(self.encoder, 'm_affix_embedding') else
                torch.zeros(1, morpho_dim)
            ),
            tr_d_model=d,
            tr_dropout=seq_tr_dropout,
            layernorm_epsilon=layernorm_epsilon,
        )

    def forward(self, args, rel_pos_arr,
                tokens_lengths, input_sequence_lengths,
                pos_tags, stems, afsets, affixes,
                row_ids, col_ids, cell_types,
                predicted_tokens_idx, predicted_tokens_affixes_idx,
                predicted_stems, predicted_afsets, predicted_affixes_prob,
                mcr_tokens_idx=None, mcr_stems=None,
                ctp_positions=None, ctp_labels=None):

        hidden = self.encoder(
            args, rel_pos_arr, tokens_lengths, input_sequence_lengths,
            pos_tags, stems, afsets, affixes,
            row_ids=row_ids, col_ids=col_ids, cell_types=cell_types,
        )

        # ── Standard KinyaBERT losses (stem + afset + affix) ──────────────────
        n_stems = self.head_predictor.stem_decoder.weight.size(0)
        predicted_stems = predicted_stems.clamp(0, n_stems - 1)
        loss, stem_loss, afset_loss, affix_loss = self.head_predictor(
            args, hidden,
            predicted_tokens_idx, predicted_tokens_affixes_idx,
            predicted_stems, predicted_afsets, predicted_affixes_prob,
        )

        # ── MCR loss — reuse stem decoder on cell-masked tokens ───────────────
        mcr_loss = torch.tensor(0.0, device=loss.device)
        if mcr_tokens_idx is not None and mcr_stems is not None and mcr_tokens_idx.numel() > 0:
            flat = hidden.permute(1, 0, 2).reshape(-1, hidden.size(-1))
            mcr_hidden = torch.index_select(flat, 0, mcr_tokens_idx)
            mcr_state = self.head_predictor.stem_transform(mcr_hidden)
            mcr_scores = self.head_predictor.stem_decoder(mcr_state) + self.head_predictor.stem_decoder_bias
            n_stem_cls = mcr_scores.size(1)
            mcr_stems_clamped = mcr_stems.clamp(0, n_stem_cls - 1)
            mcr_loss = F.nll_loss(F.log_softmax(mcr_scores, dim=1), mcr_stems_clamped)
            loss = loss + mcr_loss

        # ── CTP loss ──────────────────────────────────────────────────────────
        ctp_loss = torch.tensor(0.0, device=loss.device)
        if ctp_positions is not None and ctp_labels is not None and ctp_positions.numel() > 0:
            ctp_logits = self.encoder.predict_ctp(hidden, ctp_positions)
            n_ctp_cls = ctp_logits.size(1)
            ctp_labels_clamped = ctp_labels.clamp(0, n_ctp_cls - 1)
            ctp_loss = F.cross_entropy(ctp_logits, ctp_labels_clamped)
            loss = loss + ctp_loss

        return loss, stem_loss, afset_loss, affix_loss, mcr_loss, ctp_loss

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)


# ── Factory ───────────────────────────────────────────────────────────────────

def tabulm_base(kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                device, args, saved_model_file=None):
    """
    Construct TabuLM-base model.  Mirrors kinyabert_base() from morpho_model.py.
    All default hyperparameters match KinyaBERT-base so a KinyaBERT checkpoint
    can be used to warm-start the encoder (tabular components start from scratch).
    """
    num_stems = len(kb_vocab.reduced_stem_vocab) + 1
    num_afsets = (len(affix_set_vocab.affix_set_vocab_idx) + 1) if affix_set_vocab is not None else 0
    num_pos_tags = len(kb_vocab.pos_tag_vocab) + 1
    num_affixes = len(kb_vocab.affix_vocab) + 1
    num_rel_pos_dict_size = (len(morpho_rel_pos_dict) + 1) if morpho_rel_pos_dict is not None else 0

    model = TabuLM(
        args,
        num_stems=num_stems,
        num_afsets=num_afsets,
        num_pos_tags=num_pos_tags,
        num_affixes=num_affixes,
        num_rel_pos_dict_size=num_rel_pos_dict_size,
        num_pos_m_embeddings=args.num_pos_m_embeddings,
        num_stem_m_embeddings=args.num_stem_m_embeddings,
        use_affix_bow_m_embedding=args.use_affix_bow_m_embedding,
        use_pos_aware_rel_pos_bias=args.use_pos_aware_rel_pos_bias,
        use_tupe_rel_pos_bias=args.use_tupe_rel_pos_bias,
        max_seq_len=args.max_seq_len,
        morpho_dim=args.morpho_dim,
        stem_dim=args.stem_dim,
        morpho_tr_nhead=args.morpho_tr_nhead,
        morpho_tr_nlayers=args.morpho_tr_nlayers,
        morpho_tr_dim_feedforward=args.morpho_tr_dim_feedforward,
        morpho_tr_dropout=args.morpho_tr_dropout,
        seq_tr_nhead=args.seq_tr_nhead,
        seq_tr_nlayers=args.seq_tr_nlayers,
        seq_tr_dim_feedforward=args.seq_tr_dim_feedforward,
        seq_tr_dropout=args.seq_tr_dropout,
        layernorm_epsilon=args.layernorm_epsilon,
    ).to(device)

    if saved_model_file is not None:
        state = torch.load(saved_model_file, map_location=device)
        # Warm-start from KinyaBERT checkpoint (ignore tabular-specific keys)
        own_state = model.state_dict()
        pretrained = state.get('model_state_dict', state)
        matched, skipped = 0, 0
        for k, v in pretrained.items():
            # Map KinyaBERT encoder keys to TabuLM encoder keys
            new_k = k.replace('module.encoder.', 'encoder.')
            if new_k not in own_state:
                new_k = k
            if new_k in own_state and own_state[new_k].shape == v.shape:
                own_state[new_k].copy_(v)
                matched += 1
            else:
                skipped += 1
        print(f'[tabulm_base] warm-started {matched} layers, skipped {skipped}')
        model.load_state_dict(own_state)

    return model
