# TabuLM — morphological analyzer stub
# Drop-in replacement for parse_raw_text_lines that works without libkinlp.so.
# Uses regex heuristics to assign word types and falls back to BPE for all stems.

import re
from typing import List

import youtokentome as yttm

_NUMERIC_RE = re.compile(
    r'^[\d,.\s]+(%|Frw|RWF|km|kg|ha|m|m²|L|MW|USD|acres|ha)?$', re.IGNORECASE
)
_DATE_RE = re.compile(
    r'^\d{4}(-\d{2}(-\d{2})?)?$|^\d{1,2}/\d{1,2}/\d{2,4}$'
)
_SPECIAL_TOKEN_RE = re.compile(r'^\[.*\]$')
_UPPER_START_RE = re.compile(r'^[A-Z][a-z]')


def _classify_word_type(word: str) -> str:
    """Return the KBVocab word-type prefix for a given surface form."""
    if _SPECIAL_TOKEN_RE.match(word):
        return 'T'
    if _DATE_RE.match(word) or _NUMERIC_RE.match(word):
        return 'NU'
    if _UPPER_START_RE.match(word):
        return 'NP'
    return 'T'


def parse_text_stub(text: str, kb_vocab, bpe: yttm.BPE) -> List:
    """
    Stub replacement for parse_raw_text_lines from morpho_data_loaders.
    Takes a space-separated text string and returns a list of ParsedToken objects.
    No libkinlp.so / CFFI required — all stems resolved via BPE fallback.
    """
    from morpho_data_loaders import ParsedToken

    unk_pos = kb_vocab.pos_tag_vocab.get('<UNK>', 1)
    unk_stem = kb_vocab._stem_vocab.get('<UNK>', 1)

    parsed_tokens: List[ParsedToken] = []

    for word in text.split():
        if not word:
            continue

        word_type = _classify_word_type(word)

        if word_type == 'NU':
            stem_key = f'NU:{word}'
            si = kb_vocab._stem_vocab.get(stem_key, unk_stem)
            ptoken = ParsedToken(
                word, decode_prob=1.0, tf_idf=0.001,
                pos_tag_id=unk_pos, stem_ids=[si]
            )
            parsed_tokens.append(ptoken)
            continue

        try:
            subwords = bpe.encode(word, output_type=yttm.OutputType.SUBWORD)
        except Exception:
            subwords = []

        if not subwords:
            stem_key = f'{word_type}:{word}'
            si = kb_vocab._stem_vocab.get(stem_key, unk_stem)
            subwords_sids = [si]
        else:
            subwords_sids = []
            for sw in subwords:
                stem_key = f'{word_type}:{sw}'
                si = kb_vocab._stem_vocab.get(stem_key, unk_stem)
                subwords_sids.append(si)

        ptoken = ParsedToken(
            word, decode_prob=1.0, tf_idf=0.001,
            pos_tag_id=unk_pos, stem_ids=subwords_sids
        )
        parsed_tokens.append(ptoken)

    return parsed_tokens
