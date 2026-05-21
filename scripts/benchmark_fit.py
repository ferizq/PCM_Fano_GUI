#!/usr/bin/env python3
"""Benchmark helpers for model evaluation and fitting runtime.

Usage example:
    e:/VS_CODE/PCM_Fano_GUI/.venv/Scripts/python.exe scripts/benchmark_fit.py --points 2000 --repeats 3
"""
import argparse
import json
import os
import sys
import time
from statistics import mean, stdev

import numpy as np

# Ensure package import works when script is executed from repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from peakfit.fitting import fit_with_scipy
from peakfit.model import model


def _base_params():
    return {
        'C': 171400.0,
        'D': 100000.0,
        'b': 0.0,
        'g0': 4.5,
        'q': 2.0,
        'a': 0.5431,
        'L': 10.0,
        'N': 0.85,
        'y0': 0.12,
    }


def _fit_config_from_params(params):
    return {
        'C': {'value': params['C'], 'vary': False},
        'D': {'value': params['D'], 'vary': False},
        'b': {'value': params['b'], 'vary': False},
        'g0': {'value': params['g0'], 'vary': False},
        'q': {'value': params['q'], 'vary': False},
        'a': {'value': params['a'], 'vary': False},
        'L': {'value': params['L'], 'vary': False},
        'N': {'value': 0.3, 'vary': True, 'min': 0.01, 'max': 5.0},
        'y0': {'value': -0.2, 'vary': True, 'min': -5.0, 'max': 5.0},
    }


def _time_model(points, grid_size, repeats, kernel, accelerator):
    iw = np.linspace(360.0, 620.0, points)
    params = _base_params()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model(iw, params, integrator='grid', grid_size=grid_size, kernel=kernel, accelerator=accelerator)
        times.append(time.perf_counter() - t0)
    return times


def _time_fit(points, grid_size, repeats, kernel, accelerator):
    iw = np.linspace(360.0, 620.0, points)
    params = _base_params()
    y = model(iw, params, integrator='grid', grid_size=grid_size, kernel=kernel, accelerator=accelerator)
    cfg = _fit_config_from_params(params)

    times = []
    nfev_list = []
    r2_list = []
    converged_list = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        fitted, errs, popt, pcov, y_model, r2, converged, message, nfev = fit_with_scipy(
            iw,
            y,
            cfg,
            integrator_opts={
                'integrator': 'grid',
                'grid_size': grid_size,
                'autoscale': True,
                'kernel': kernel,
                'accelerator': accelerator,
            },
            curvefit_opts={'maxfev': 1200},
        )
        times.append(time.perf_counter() - t0)
        nfev_list.append(int(nfev))
        r2_list.append(float(r2) if np.isfinite(r2) else float('nan'))
        converged_list.append(bool(converged))

    return times, nfev_list, r2_list, converged_list


def _summary(vals):
    if not vals:
        return {'mean': None, 'stdev': None, 'min': None, 'max': None}
    if len(vals) == 1:
        return {'mean': vals[0], 'stdev': 0.0, 'min': vals[0], 'max': vals[0]}
    return {
        'mean': mean(vals),
        'stdev': stdev(vals),
        'min': min(vals),
        'max': max(vals),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark model and fitting runtime.')
    parser.add_argument('--points', type=int, default=2000, help='Number of data points.')
    parser.add_argument('--grid-size', type=int, default=600, help='Grid size for integrator.')
    parser.add_argument('--repeats', type=int, default=3, help='Repeat count per benchmark.')
    parser.add_argument('--kernel', choices=['PCM_Fano_Bessel', 'PCM_Fano_Gauss'], default='PCM_Fano_Bessel')
    parser.add_argument('--accelerator', choices=['auto', 'c', 'numpy', 'numba'], default='auto')
    parser.add_argument('--output-json', default='', help='Optional path to write JSON results.')
    args = parser.parse_args()

    model_times = _time_model(args.points, args.grid_size, args.repeats, args.kernel, args.accelerator)
    fit_times, nfev_list, r2_list, converged_list = _time_fit(args.points, args.grid_size, args.repeats, args.kernel, args.accelerator)

    report = {
        'points': args.points,
        'grid_size': args.grid_size,
        'repeats': args.repeats,
        'kernel': args.kernel,
        'accelerator': args.accelerator,
        'model_runtime_seconds': _summary(model_times),
        'fit_runtime_seconds': _summary(fit_times),
        'fit_nfev': _summary(nfev_list),
        'fit_r2': _summary(r2_list),
        'fit_converged_count': int(sum(converged_list)),
    }

    print(json.dumps(report, indent=2))

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
