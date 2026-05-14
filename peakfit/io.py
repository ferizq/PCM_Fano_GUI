"""Data IO and parameter config loading.

This module provides robust, locale-independent numeric parsing helpers
so the GUI and single-file executable behave the same regardless of the
host system locale. In particular we normalize decimal separators so
that a comma is interpreted as a decimal point and common thousands
separators (spaces, NBSP, apostrophes) are removed.
"""
from typing import Tuple, Dict
import re
import numpy as np
import json
from io import StringIO


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column (x y) data file.

    Handles files using comma as decimal separator (e.g. '399,8611') as well as
    standard dot decimal. Whitespace or tab delimited.

    Returns arrays (x, y).
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Normalize numeric tokens so the file is parsed the same regardless of
    # the OS locale. Strategy:
    # - remove common grouping separators (spaces, NBSP, apostrophes)
    # - if a token contains both '.' and ',' assume '.' is thousands and
    #   ',' is the decimal separator (e.g. '1.234,56') -> '1234.56'
    # - otherwise convert any ',' to '.' (decimal comma -> point)
    # - if a token contains multiple '.' and no comma, treat dots as
    #   grouping separators and remove them ('1.234.567' -> '1234567')

    # regex matching numbers with optional grouping separators and decimals
    num_re = re.compile(r"[-+]? (?:\d{1,3}(?:[\.\s\u00A0'’]\d{3})+|\d+)(?:[\.,]\d+)?(?:[eE][-+]?\d+)?", re.VERBOSE)

    def _normalize_num_token(tok: str) -> str:
        t = tok
        # strip surrounding whitespace
        t = t.strip()
        if t == '':
            return t
        # remove non-digit grouping characters (spaces, NBSP, apostrophes)
        t = t.replace('\u00A0', '').replace(' ', '').replace("'", '').replace('’', '')
        # if both dot and comma present, assume dot is thousands and comma decimal
        if '.' in t and ',' in t:
            t = t.replace('.', '').replace(',', '.')
            return t
        # convert comma decimal to point
        if ',' in t:
            t = t.replace(',', '.')
        # if multiple dots remain, decide whether they're grouping or decimal
        if t.count('.') > 1:
            parts = t.split('.')
            # if the last group has length 3 it's likely a thousands grouping
            if all(p.isdigit() for p in parts):
                if len(parts[-1]) == 3:
                    # remove all dots
                    t = ''.join(parts)
                else:
                    # treat last as fractional part
                    t = ''.join(parts[:-1]) + '.' + parts[-1]
        return t

    processed_lines = []
    for line in raw.splitlines():
        if line.strip() == '':
            processed_lines.append('')
            continue
        if '#' in line:
            code, comment = line.split('#', 1)
            comment = '#' + comment
        else:
            code = line
            comment = ''
        # find numeric tokens in the code portion preserving order
        nums = num_re.findall(code)
        if not nums:
            # no numeric tokens found, keep original (but preserve comment)
            nl = code.strip()
            if comment:
                nl = (nl + ' ' + comment) if nl else comment
            processed_lines.append(nl)
            continue
        normed = [_normalize_num_token(n) for n in nums]
        new_line = ' '.join(normed)
        if comment:
            new_line = new_line + ' ' + comment
        processed_lines.append(new_line)

    processed = '\n'.join(processed_lines)

    data = np.loadtxt(StringIO(processed))
    if data.ndim == 1:
        if data.size < 2:
            raise ValueError("Data file must contain at least two numbers per row")
        x = np.array([data[0]])
        y = np.array([data[1]])
    else:
        x = data[:, 0]
        y = data[:, 1]
    return x.astype(float), y.astype(float)


def load_param_config(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding='utf-8') as f:
        cfg = json.load(f)

    # Canonicalize legacy parameter names that start with a leading 'i'.
    # If the JSON contains both the legacy key (e.g. 'iC') and the canonical
    # key (e.g. 'C'), prefer the canonical key and ignore the legacy one.
    out = {}
    keys = set(cfg.keys())
    for k, v in cfg.items():
        if isinstance(k, str) and k.startswith('i') and len(k) > 1:
            nk = k[1:]
            if nk in keys:
                # canonical key present; skip legacy key
                continue
            out[nk] = v
        else:
            out[k] = v
    return out


def normalize_number_string(s: str) -> str:
    """Return a locale-independent numeric string using '.' as decimal separator.

    This is useful when reading user-entered values from the GUI (which may
    use comma as decimal separator on some systems) before casting to float.
    """
    if s is None:
        return ''
    t = str(s).strip()
    if t == '':
        return t
    t = t.replace('\u00A0', '').replace(' ', '').replace("'", '').replace('’', '')
    if '.' in t and ',' in t:
        t = t.replace('.', '').replace(',', '.')
        return t
    if ',' in t:
        t = t.replace(',', '.')
    if t.count('.') > 1:
        parts = t.split('.')
        if all(p.isdigit() for p in parts):
            if len(parts[-1]) == 3:
                t = ''.join(parts)
            else:
                t = ''.join(parts[:-1]) + '.' + parts[-1]
    return t
