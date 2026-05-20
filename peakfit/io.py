"""Data IO and parameter config loading.

Provides locale-independent numeric parsing helpers used by the GUI
and command-line helpers. Ensures numbers using comma decimals or
grouping separators are normalized to a dot-decimal form before
conversion.
"""
from typing import Tuple, Dict, Any
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

    # Numeric token regexp (matches integers, grouped thousands, decimals, exponents)
    num_re = re.compile(r"[-+]?(?:\d{1,3}(?:[\.\s\u00A0'’]\d{3})+|\d+)(?:[\.,]\d+)?(?:[eE][-+]?\d+)?")

    def _normalize_num_token(tok: str) -> str:
        t = str(tok).strip()
        if t == '':
            return t
        # remove grouping characters (space, NBSP, apostrophes)
        t = t.replace('\u00A0', '').replace(' ', '').replace("'", '').replace('’', '')
        if '.' in t and ',' in t:
            # e.g. 1.234,56 -> 1234.56
            t = t.replace('.', '').replace(',', '.')
            return t
        if ',' in t:
            # decimal comma -> point
            t = t.replace(',', '.')
        # multiple dots: decide grouping vs decimal heuristically
        if t.count('.') > 1:
            parts = t.split('.')
            if all(p.isdigit() for p in parts):
                # if final group has 3 digits, treat dots as thousands separators
                if len(parts[-1]) == 3:
                    t = ''.join(parts)
                else:
                    # otherwise treat last part as fractional
                    t = ''.join(parts[:-1]) + '.' + parts[-1]
        return t

    # Build (x, y) lists by extracting the first two numeric tokens
    # on each line. This tolerates files with varying column counts
    # (extra metadata columns, missing trailing columns, etc.).
    x_vals = []
    y_vals = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        # strip inline comments
        code = line.split('#', 1)[0].strip()
        if not code:
            continue
        # tokenize on whitespace (tabs/spaces). Avoid splitting on comma
        # so locale decimal commas remain attached to the numeric token.
        toks = re.split(r"\s+", code)
        toks = [t for t in toks if t != '']
        if not toks:
            continue
        # normalize tokens and keep those that contain digits
        normed = []
        for t in toks:
            if re.search(r"\d", t):
                nt = _normalize_num_token(t)
                if nt != '':
                    normed.append(nt)
        if len(normed) < 2:
            # not enough numeric tokens on this line, skip
            continue
        try:
            xv = float(normed[0])
            yv = float(normed[1])
        except Exception:
            # skip lines that still fail to convert
            continue
        x_vals.append(xv)
        y_vals.append(yv)

    if not x_vals:
        raise ValueError(f"No numeric (x y) data could be parsed from {path!r}")

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    return x, y


def load_param_config(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding='utf-8') as f:
        cfg = json.load(f)

    # Normalize keys and numeric subfields for locale independence.
    # Strategy:
    # - canonical name: strip leading 'i' if present, then remove underscores
    # - if the canonical form is present explicitly in the JSON, prefer it
    # - normalize string numeric fields for 'value', 'min', 'max'
    def _normalize_value_field(v):
        try:
            if isinstance(v, str):
                ns = normalize_number_string(v)
                try:
                    return float(ns)
                except Exception:
                    return v
            return v
        except Exception:
            return v

    processed = {}
    cfg_keys = set(cfg.keys())
    for k, v in cfg.items():
        # derive canonical key name
        if isinstance(k, str) and k.startswith('i') and len(k) > 1:
            base = k[1:]
        else:
            base = k
        canonical = base.replace('_', '')
        # prefer an explicitly provided canonical key in the JSON
        if canonical in cfg_keys and canonical != k:
            continue
        # normalize numeric string fields inside the parameter dict
        if isinstance(v, dict):
            new_v = {}
            for fk, fv in v.items():
                if fk in ('value', 'min', 'max'):
                    new_v[fk] = _normalize_value_field(fv)
                else:
                    new_v[fk] = fv
            processed[canonical] = new_v
        else:
            processed[canonical] = v
    return processed


def normalize_number_string(s: Any) -> str:
    """Return a locale-independent numeric string using '.' as decimal separator.

    Accepts inputs like '1.234,56', "1 234,56", "1'234,56" and returns
    a string with a dot decimal (e.g. '1234.56').
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
