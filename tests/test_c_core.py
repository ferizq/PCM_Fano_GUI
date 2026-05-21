import numpy as np
import pytest

from peakfit.c_core import c_core_available, c_grid_integral, kernel_code_from_name
from peakfit.model import kernel_integrand


def _python_grid_integral(iw, params, kernel, ik_min=0.0, ik_max=1.0, grid_size=400):
    ik = np.linspace(ik_min, ik_max, int(grid_size))
    vals = kernel_integrand(ik[:, None], np.atleast_1d(iw)[None, :], params, kernel=kernel)
    dx = np.diff(ik)
    return np.sum((vals[1:, :] + vals[:-1, :]) * (dx[:, None]) / 2.0, axis=0)


@pytest.mark.skipif(not c_core_available(), reason='native C core DLL not found')
@pytest.mark.parametrize('kernel', ['PCM_Fano_Bessel', 'PCM_Fano_Gauss'])
def test_c_grid_integral_matches_python_grid(kernel):
    iw = np.linspace(360.0, 620.0, 80)
    params = {
        'C': 171400.0,
        'D': 100000.0,
        'b': 0.0,
        'g0': 4.5,
        'q': 2.0,
        'a': 0.5431,
        'L': 10.0,
        'alpha': 1.0,
    }

    c_out = c_grid_integral(
        iw,
        0.0,
        1.0,
        400,
        kernel_code_from_name(kernel),
        params['C'],
        params['D'],
        params['b'],
        params['g0'],
        params['q'],
        params['a'],
        params['L'],
        params['alpha'],
    )
    py_out = _python_grid_integral(iw, params, kernel, ik_min=0.0, ik_max=1.0, grid_size=400)

    assert c_out.shape == py_out.shape
    assert np.all(np.isfinite(c_out))
    assert np.allclose(c_out, py_out, rtol=1e-8, atol=1e-10)
