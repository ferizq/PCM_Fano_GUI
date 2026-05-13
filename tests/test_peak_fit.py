def test_model_basic():
    import numpy as np
    from peakfit.model import model

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
