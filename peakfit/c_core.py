"""Optional ctypes bridge for the native C core integrator."""
from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


_DPTR = ctypes.POINTER(ctypes.c_double)


def _kernel_code_from_name_py(kernel: str) -> int:
    if kernel is None:
        return 0
    kl = str(kernel).strip().lower()
    if kl in ('pcm_fano_gauss', 'exp', 'gauss', 'gaussian_exp'):
        return 1
    return 0


def _candidate_library_paths():
    seen = set()

    env_path = os.environ.get('PEAKFIT_CORE_LIB')
    if env_path:
        p = Path(env_path)
        if p not in seen:
            seen.add(p)
            yield p

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    lib_names = ('peakfit_core.dll', 'libpeakfit_core.so', 'libpeakfit_core.dylib')
    base_paths = [pkg_dir, repo_root / 'build', repo_root / 'dist']

    if hasattr(sys, '_MEIPASS'):
        base = Path(getattr(sys, '_MEIPASS'))
        base_paths.extend([base, base / 'build'])

    for base in base_paths:
        for lib_name in lib_names:
            candidate = base / lib_name
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _bind_library(lib: ctypes.CDLL) -> None:
    if getattr(lib, '_pf_bound', False):
        return

    lib.pf_kernel_code_from_name.argtypes = [ctypes.c_char_p]
    lib.pf_kernel_code_from_name.restype = ctypes.c_int

    lib.pf_grid_integral.argtypes = [
        _DPTR,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        _DPTR,
    ]
    lib.pf_grid_integral.restype = ctypes.c_int

    if hasattr(lib, 'pf_core_version'):
        lib.pf_core_version.argtypes = []
        lib.pf_core_version.restype = ctypes.c_char_p

    lib._pf_bound = True


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL | None:
    for path in _candidate_library_paths():
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            _bind_library(lib)
            return lib
        except Exception:
            continue
    return None


def c_core_available() -> bool:
    return _load_library() is not None


def c_core_version() -> str:
    lib = _load_library()
    if lib is None or not hasattr(lib, 'pf_core_version'):
        return 'unavailable'
    try:
        raw = lib.pf_core_version()
    except Exception:
        return 'unknown'
    if raw is None:
        return 'unknown'
    if isinstance(raw, bytes):
        return raw.decode('utf-8', errors='replace')
    return str(raw)


def kernel_code_from_name(kernel: str) -> int:
    lib = _load_library()
    if lib is None:
        return _kernel_code_from_name_py(kernel)
    try:
        return int(lib.pf_kernel_code_from_name(str(kernel).encode('utf-8')))
    except Exception:
        return _kernel_code_from_name_py(kernel)


def c_grid_integral(iw_arr,
                    ik_min: float,
                    ik_max: float,
                    grid_size: int,
                    kernel_code: int,
                    C: float,
                    D: float,
                    b: float,
                    g0: float,
                    q: float,
                    a: float,
                    L: float,
                    alpha: float) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError('peakfit_core native library is not available')

    iw = np.ascontiguousarray(np.asarray(iw_arr, dtype=np.float64))
    out = np.empty_like(iw)

    rc = lib.pf_grid_integral(
        iw.ctypes.data_as(_DPTR),
        ctypes.c_int(int(iw.size)),
        ctypes.c_double(float(ik_min)),
        ctypes.c_double(float(ik_max)),
        ctypes.c_int(int(grid_size)),
        ctypes.c_int(int(kernel_code)),
        ctypes.c_double(float(C)),
        ctypes.c_double(float(D)),
        ctypes.c_double(float(b)),
        ctypes.c_double(float(g0)),
        ctypes.c_double(float(q)),
        ctypes.c_double(float(a)),
        ctypes.c_double(float(L)),
        ctypes.c_double(float(alpha)),
        out.ctypes.data_as(_DPTR),
    )
    if int(rc) != 0:
        raise RuntimeError(f'pf_grid_integral failed with code {int(rc)}')

    return out
