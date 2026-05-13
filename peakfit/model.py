"""Model and integrand for peak fitting defined via a numerical integral.

The integrand is implemented exactly as specified by the user:

    w0 = sqrt(iC + iD*cos(ik*pi/2)) - ib
    eps = 2*(iw - w0)/ig0
    den = (sin(x) - x*cos(x))**2 / ik**4   with x = (ik*pi/ia)*iL
    num = (eps + iq)**2 / (1 + eps**2)
    integrand = den * num

Two integration methods are available: 'grid' (fast, approximate)
and 'quad' (uses scipy.integrate.quad for higher accuracy).
"""
from typing import Any, Dict, Optional
import numpy as np


def kernel_integrand(ik, iw, params: Dict[str, Any]):
    """Compute the integrand value(s) for given ik and iw.

    Parameters
    - ik: scalar or array-like (integration variable)
    - iw: scalar or array-like (independent variable from data)
    - params: dict with keys iC,iD,ib,ig0,iq,ia,iL

    Returns
    - array-like with shape broadcasted from ik and iw
    """
    ik = np.asarray(ik, dtype=float)
    iw = np.asarray(iw, dtype=float)

    iC = float(params.get("iC", 171400.0))
    iD = float(params.get("iD", 100000.0))
    ib = float(params.get("ib", 0.0))
    ig0 = float(params.get("ig0", 4.5))
    iq = float(params.get("iq", -1.0e12))
    ia = float(params.get("ia", 0.5431))
    iL = float(params.get("iL", 10.0))

    # w0 = sqrt(iC + iD*cos(ik * pi / 2)) - ib
    # ensure non-negative inside sqrt where physically appropriate
    sqrt_arg = iC + iD * np.cos(ik * np.pi / 2.0)
    # guard small negative rounding by clipping to zero
    sqrt_arg = np.clip(sqrt_arg, a_min=0.0, a_max=None)
    w0 = np.sqrt(sqrt_arg) - ib

    eps = 2.0 * (iw - w0) / ig0
    num = (eps + iq) ** 2 / (1.0 + eps ** 2)

    # x = (ik * pi / ia) * iL
    x = (ik * np.pi / ia) * iL

    numerator = np.sin(x) - x * np.cos(x)
    # safe handling for ik -> 0: analytic limit gives den ~ c**6 * ik**2 / 9
    c = (iL * np.pi / ia)

    # Avoid evaluating the division for ik == 0 to prevent invalid-value warnings.
    den = np.empty_like(numerator, dtype=float)
    small_mask = np.abs(ik) <= 1e-12
    large_mask = ~small_mask
    if np.any(large_mask):
        den[large_mask] = (numerator[large_mask] ** 2) / (ik[large_mask] ** 4)
    if np.any(small_mask):
        den[small_mask] = (c ** 6) * (ik[small_mask] ** 2) / 9.0

    return den * num


def model(iw, params: Dict[str, Any], integrator: str = "grid", grid_size: int = 400,
          ik_min: float = 0.0, ik_max: float = 1.0, quad_opts: Optional[Dict] = None):
    """Compute the model y-values for input `iw` using the integral-defined kernel.

    Returns y = N * integral_{ik=ik_min..ik_max} kernel_integrand(ik, iw, params) d(ik) + y0
    """
    iw_arr = np.atleast_1d(iw).astype(float)

    if integrator == "grid":
        ik = np.linspace(ik_min, ik_max, int(grid_size))
        ik_mesh = ik[:, None]
        iw_mesh = iw_arr[None, :]
        vals = kernel_integrand(ik_mesh, iw_mesh, params)
        # Manual trapezoidal integration along the ik axis to avoid depending on np.trapz
        dx = np.diff(ik)
        integral = np.sum((vals[1:, :] + vals[:-1, :]) * (dx[:, None]) / 2.0, axis=0)
    elif integrator == "quad":
        try:
            from scipy.integrate import quad
        except Exception:
            raise ImportError("scipy is required for the 'quad' integrator. Install scipy or use integrator='grid'.")
        quad_opts = quad_opts or {}
        integral_list = []
        for wi in iw_arr:
            f = lambda k: float(kernel_integrand(np.float64(k), np.float64(wi), params))
            res, err = quad(f, ik_min, ik_max, **quad_opts)
            integral_list.append(res)
        integral = np.array(integral_list)
    else:
        raise ValueError(f"Unknown integrator '{integrator}'. Use 'grid' or 'quad'.")

    N = float(params.get("N", 1.0))
    y0 = float(params.get("y0", 0.0))
    # Reduce linear dependency by scaling N with (iq^2 + 1) and iL^3 as requested.
    iq = float(params.get('iq', 0.0))
    iL = float(params.get('iL', 1.0))
    denom = (iq ** 2 + 1.0) * (iL ** 3)
    if denom == 0.0:
        denom = 1e-24
    return (N / denom) * integral + y0
