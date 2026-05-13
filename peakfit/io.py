"""Data IO and parameter config loading."""
from typing import Tuple, Dict
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

    # Replace comma decimal separators with dot so numpy can parse them.
    # This is a pragmatic approach for locale-formatted data like the provided file.
    normalized = raw.replace(',', '.')

    data = np.loadtxt(StringIO(normalized))
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
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg
