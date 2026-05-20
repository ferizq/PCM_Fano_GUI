import os
import sys
import numpy as np

# ensure package root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from peakfit.model import kernel_integrand, model

def compute_integral(iw, params, kernel, ik_min=0.0, ik_max=1.0, grid_size=2000):
    ik = np.linspace(ik_min, ik_max, grid_size)
    ik_mesh = ik[:, None]
    iw_mesh = np.atleast_1d(iw)[None, :]
    vals = kernel_integrand(ik_mesh, iw_mesh, params, kernel=kernel)
    dx = np.diff(ik)
    integral = np.sum((vals[1:, :] + vals[:-1, :]) * (dx[:, None]) / 2.0, axis=0)
    return integral


def test_scaling():
    iw = np.array([0.1])
    params = {
        'C': 171400.0,
        'D': 100000.0,
        'b': 0.0,
        'g0': 4.5,
        'q': 2.0,
        'a': 0.5431,
        'L': 10.0,
        'N': 1.0,
        'y0': 0.0,
    }

    # Bessel
    integral_b = compute_integral(iw, params, kernel='PCM_Fano_Bessel', grid_size=2000)
    y_b = model(iw, params, integrator='grid', grid_size=2000, kernel='PCM_Fano_Bessel')
    s_obs_b = float(y_b[0] / integral_b[0])
    s_exp_b = 1.0 / ((params['q'] ** 2 + 1.0) * (params['L'] ** 3))
    rel_err_b = abs(s_obs_b - s_exp_b) / (abs(s_exp_b) + 1e-24)

    # Gauss (use same params + alpha if desired)
    params_g = params.copy()
    params_g['alpha'] = 1.0
    integral_g = compute_integral(iw, params_g, kernel='PCM_Fano_Gauss', grid_size=2000)
    y_g = model(iw, params_g, integrator='grid', grid_size=2000, kernel='PCM_Fano_Gauss')
    s_obs_g = float(y_g[0] / integral_g[0])
    s_exp_g = (params_g['L'] ** 3) / (params_g['q'] ** 2 + 1.0) * (params_g.get('alpha', 1.0) ** (-4.0 / 3.0))
    rel_err_g = abs(s_obs_g - s_exp_g) / (abs(s_exp_g) + 1e-24)

    assert np.isfinite(rel_err_b)
    assert np.isfinite(rel_err_g)
    assert rel_err_b < 5e-3
    assert rel_err_g < 5e-3

if __name__ == '__main__':
    test_scaling()
