import numpy as np

from peakfit.fitting import _build_free_params, fit_with_scipy
from peakfit.model import model


def test_model_basic():
    iw = np.linspace(0.01, 1.0, 30)
    params = {
        'iC': 1.0,
        'iD': 0.1,
        'ib': 0.0,
        'ig0': 0.5,
        'iq': 0.0,
        'ia': 1.0,
        'iL': 1.0,
        'N': 1.0,
        'y0': 0.0,
    }
    y = model(iw, params, integrator='grid', grid_size=200)
    assert y.shape == iw.shape
    assert np.all(np.isfinite(y))


def test_model_accelerator_modes_supported():
    iw = np.linspace(0.01, 1.0, 64)
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
    y_auto = model(iw, params, integrator='grid', grid_size=200, kernel='PCM_Fano_Bessel', accelerator='auto')
    y_numpy = model(iw, params, integrator='grid', grid_size=200, kernel='PCM_Fano_Bessel', accelerator='numpy')

    assert y_auto.shape == iw.shape
    assert y_numpy.shape == iw.shape
    assert np.all(np.isfinite(y_auto))
    assert np.all(np.isfinite(y_numpy))
    assert np.allclose(y_auto, y_numpy, rtol=5e-3, atol=1e-8)


def test_model_c_accelerator_supported_or_fallback():
    iw = np.linspace(0.01, 1.0, 64)
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
    y_c = model(iw, params, integrator='grid', grid_size=200, kernel='PCM_Fano_Bessel', accelerator='c')
    y_numpy = model(iw, params, integrator='grid', grid_size=200, kernel='PCM_Fano_Bessel', accelerator='numpy')

    assert y_c.shape == iw.shape
    assert np.all(np.isfinite(y_c))
    assert np.allclose(y_c, y_numpy, rtol=5e-3, atol=1e-8)


def test_build_free_params_coerces_vary_flag():
    cfg = {
        'a': {'value': 1.0, 'vary': 'true'},
        'b': {'value': 2.0, 'vary': 'False'},
        'c': {'value': 3.0, 'vary': 1},
        'd': {'value': 4.0, 'vary': 0},
    }
    free_names, p0, bounds = _build_free_params(cfg)

    assert free_names == ['a', 'c']
    assert np.allclose(p0, np.array([1.0, 3.0]))
    assert not np.all(np.isfinite(bounds[0]))
    assert not np.all(np.isfinite(bounds[1]))


def test_fit_with_scipy_two_free_params_regression():
    iw = np.linspace(360.0, 620.0, 120)
    true_params = {
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
    y = model(iw, true_params, integrator='grid', grid_size=240, kernel='PCM_Fano_Bessel')

    param_config = {
        'C': {'value': true_params['C'], 'vary': False},
        'D': {'value': true_params['D'], 'vary': False},
        'b': {'value': true_params['b'], 'vary': False},
        'g0': {'value': true_params['g0'], 'vary': False},
        'q': {'value': true_params['q'], 'vary': False},
        'a': {'value': true_params['a'], 'vary': False},
        'L': {'value': true_params['L'], 'vary': False},
        'N': {'value': 0.2, 'vary': True, 'min': 0.01, 'max': 5.0},
        'y0': {'value': -0.5, 'vary': True, 'min': -5.0, 'max': 5.0},
    }

    fitted, errs, popt, pcov, y_model, r2, converged, message, nfev = fit_with_scipy(
        iw,
        y,
        param_config,
        integrator_opts={
            'integrator': 'grid',
            'grid_size': 240,
            'autoscale': True,
            'kernel': 'PCM_Fano_Bessel',
            'accelerator': 'numpy',
        },
        curvefit_opts={'maxfev': 800}
    )

    assert nfev > 0
    assert popt is not None
    assert y_model.shape == iw.shape
    assert np.isfinite(r2)
    assert r2 > 0.99
    assert abs(float(fitted['N']) - true_params['N']) < 1e-3
    assert abs(float(fitted['y0']) - true_params['y0']) < 1e-3
