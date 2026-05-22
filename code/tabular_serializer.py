# TabuLM — tabular data serializer
# Converts CSV / dict-of-records tables into a flat token string
# plus a parallel word_metadata list for row/column tracking.

import csv
import re
from enum import IntEnum
from typing import List, Tuple

# ── Cell type taxonomy ────────────────────────────────────────────────────────

class CellType(IntEnum):
    PAD         = 0   # padding / special tokens with no cell identity
    HEADER      = 1   # column header row
    NUMERIC     = 2   # quantity, count, percentage, measurement
    TEXT        = 3   # free-form text longer than a label
    CATEGORICAL = 4   # short label / enum value
    DATE        = 5   # year or full date

NUM_CELL_TYPES = 6

# ── Regex heuristics ──────────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(
    r'^[\d,.\s]+(%|Frw|RWF|km|kg|ha|m|m²|L|MW|USD|acres|ha)?$',
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r'^(1[89]\d{2}|2[012]\d{2})(-\d{2}(-\d{2})?)?$|^\d{1,2}/\d{1,2}/\d{2,4}$'
)
_NULL_RE = re.compile(r'^[-–]$|^n/?a$|^null$|^none$|^$', re.IGNORECASE)


# ── Core data structures ──────────────────────────────────────────────────────

class TableCell:
    """One cell in a table, with its grid coordinates and semantic type."""
    __slots__ = ('content', 'row_id', 'col_id', 'cell_type')

    def __init__(self, content: str, row_id: int, col_id: int, cell_type: CellType):
        self.content = content.strip()
        self.row_id = row_id     # 1-based; 1 = header row
        self.col_id = col_id     # 1-based
        self.cell_type = cell_type

    def __repr__(self):
        return (f'TableCell(r={self.row_id}, c={self.col_id}, '
                f'type={self.cell_type.name}, "{self.content[:20]}")')


# ── Cell type detection ───────────────────────────────────────────────────────

def detect_cell_type(value: str, is_header: bool = False) -> CellType:
    if is_header:
        return CellType.HEADER
    v = value.strip()
    if _NULL_RE.match(v):
        return CellType.TEXT
    if _DATE_RE.match(v):
        return CellType.DATE
    if _NUMERIC_RE.match(v):
        return CellType.NUMERIC
    if len(v) <= 40 and '\n' not in v and ' ' not in v:
        return CellType.CATEGORICAL
    return CellType.TEXT


# ── Table loading ─────────────────────────────────────────────────────────────

def serialize_csv(filepath: str,
                  max_rows: int = 64,
                  max_cols: int = 24) -> List[TableCell]:
    """Read a CSV file and return an ordered list of TableCell objects."""
    cells: List[TableCell] = []
    try:
        with open(filepath, newline='', encoding='utf-8-sig') as f:
            rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    except Exception:
        return cells

    if not rows:
        return cells

    header = rows[0][:max_cols]
    for col_id, h in enumerate(header, start=1):
        cells.append(TableCell(
            h or f'col_{col_id}', row_id=1, col_id=col_id,
            cell_type=CellType.HEADER,
        ))

    for row_offset, row in enumerate(rows[1: max_rows + 1], start=2):
        for col_id, val in enumerate(row[:max_cols], start=1):
            cells.append(TableCell(
                val, row_id=row_offset, col_id=col_id,
                cell_type=detect_cell_type(val),
            ))

    return cells


def serialize_records(records: List[dict],
                      max_rows: int = 64,
                      max_cols: int = 24) -> List[TableCell]:
    """Convert a list of dicts (e.g. from pandas .to_dict('records')) to cells."""
    if not records:
        return []
    cells: List[TableCell] = []
    keys = list(records[0].keys())[:max_cols]

    for col_id, k in enumerate(keys, start=1):
        cells.append(TableCell(
            str(k), row_id=1, col_id=col_id,
            cell_type=CellType.HEADER,
        ))

    for row_offset, rec in enumerate(records[:max_rows], start=2):
        for col_id, k in enumerate(keys, start=1):
            val = str(rec.get(k, ''))
            cells.append(TableCell(
                val, row_id=row_offset, col_id=col_id,
                cell_type=detect_cell_type(val),
            ))

    return cells


# ── Serialization ─────────────────────────────────────────────────────────────

# WordMeta: (row_id, col_id, cell_type_int)
WordMeta = Tuple[int, int, int]


def table_cells_to_text(
    cells: List[TableCell],
) -> Tuple[str, List[WordMeta]]:
    """
    Flatten table cells into a single space-separated string with structure
    tokens, plus a parallel per-token metadata list.

    Special tokens emitted:
      [TAB]  — start of header row
      [ROW]  — start of any data row
      [CEL]  — start of each individual cell

    Returns:
      text         — string ready for morpho_stub.parse_text_stub()
      word_meta    — list of (row_id, col_id, cell_type) with one entry per
                     space-separated token in `text` (including special tokens,
                     which get row_id=0, col_id=0, cell_type=PAD)
    """
    parts: List[str] = []
    word_meta: List[WordMeta] = []

    PAD = (0, 0, int(CellType.PAD))
    cur_row = -1

    for cell in cells:
        if cell.row_id != cur_row:
            sep = '[TAB]' if cell.row_id == 1 else '[ROW]'
            parts.append(sep)
            word_meta.append(PAD)
            cur_row = cell.row_id

        parts.append('[CEL]')
        word_meta.append((cell.row_id, cell.col_id, int(cell.cell_type)))

        content_words = cell.content.split() if cell.content else ['[EMPTY]']
        if not content_words:
            content_words = ['[EMPTY]']

        for w in content_words:
            parts.append(w)
            word_meta.append((cell.row_id, cell.col_id, int(cell.cell_type)))

    return ' '.join(parts), word_meta
