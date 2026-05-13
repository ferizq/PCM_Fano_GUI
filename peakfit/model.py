"""Model and integrand for peak fitting defined via a numerical integral.

This module provides two selectable kernel implementations:
- PCM_Fano_Bessel (original Bessel/sinc-like integrand)
- PCM_Fano_Gauss (Gaussian-like envelope)

The integrand follows the user-specified form:

    w0 = sqrt(iC + iD*cos(ik*pi/2)) - ib
    eps = 2*(iw - w0)/ig0
    den = kernel-dependent denominator (see implementations)
    num = (eps + iq)**2 / (1 + eps**2)
    integrand = den * num

Two integration methods are available: 'grid' (fast, approximate)
and 'quad' (uses scipy.integrate.quad for higher accuracy).
"""
from typing import Any, Dict, Optional
import numpy as np

def _get(params: Dict[str, Any], name: str, default: Any):
    """Retrieve a parameter value, accepting legacy 'i'-prefixed names.

    Prefers the canonical name (e.g. 'a') and falls back to 'ia' if present.
    Returns the value coerced to float or the provided default.
    """
    if params is None:
        return float(default)
    if name in params:
        return float(params.get(name))
    legacy = 'i' + name
    if legacy in params:
        return float(params.get(legacy))
    return float(default)


def kernel_integrand(ik, iw, params: Dict[str, Any], kernel: str = 'PCM_Fano_Bessel'):
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

    C = _get(params, 'C', 171400.0)
    D = _get(params, 'D', 100000.0)
    b = _get(params, 'b', 0.0)
    g0 = _get(params, 'g0', 4.5)
    q = _get(params, 'q', -1.0e12)
    a = _get(params, 'a', 0.5431)
    L = _get(params, 'L', 10.0)

    # w0 = sqrt(C + D*cos(ik * pi / 2)) - b
    # ensure non-negative inside sqrt where physically appropriate
    sqrt_arg = C + D * np.cos(ik * np.pi / 2.0)
    # guard small negative rounding by clipping to zero
    sqrt_arg = np.clip(sqrt_arg, a_min=0.0, a_max=None)
    w0 = np.sqrt(sqrt_arg) - b

    eps = 2.0 * (iw - w0) / g0
    num = (eps + q) ** 2 / (1.0 + eps ** 2)

    # support multiple kernel shapes selectable by the caller via `kernel`.
    # Accept the new canonical names and legacy aliases for compatibility.
    if kernel is None:
        kernel = 'PCM_Fano_Bessel'
    kernel = str(kernel).strip().lower()

    if kernel in ('pcm_fano_bessel', 'sinc', 'bessel'):
        # x = (ik * pi / a) * L
        x = (ik * np.pi / a) * L
        numerator = np.sin(x) - x * np.cos(x)
        # safe handling for ik -> 0: analytic limit gives den ~ c**6 * ik**2 / 9
        c = (L * np.pi / a)

        # Avoid evaluating the division for ik == 0 to prevent invalid-value warnings.
        den = np.empty_like(numerator, dtype=float)
        small_mask = np.abs(ik) <= 1e-12
        large_mask = ~small_mask
        if np.any(large_mask):
            den[large_mask] = (numerator[large_mask] ** 2) / (ik[large_mask] ** 4)
        if np.any(small_mask):
            den[small_mask] = (c ** 6) * (ik[small_mask] ** 2) / 9.0
    elif kernel in ('pcm_fano_gauss', 'exp', 'gauss', 'gaussian_exp'):
        # new kernel: a Gaussian-like envelope times ik^2
        # den = ik^2 * exp((-2*pi^2 * ik^2 * L^2)/(alpha*a^2))
        alpha = _get(params, 'alpha', 1.0)
        # guard against zero a
        a_safe = a if a != 0.0 else 1e-12
        exponent = (-2.0 * (np.pi ** 2) * (ik ** 2) * (L ** 2)) / (alpha * (a_safe ** 2))
        den = (ik ** 2) * np.exp(exponent)
    else:
        # unknown kernel: fall back to the PCM_Fano_Bessel (sinc-like) implementation
        x = (ik * np.pi / a) * L
        numerator = np.sin(x) - x * np.cos(x)
        c = (L * np.pi / a)
        den = np.empty_like(numerator, dtype=float)
        small_mask = np.abs(ik) <= 1e-12
        large_mask = ~small_mask
        if np.any(large_mask):
            den[large_mask] = (numerator[large_mask] ** 2) / (ik[large_mask] ** 4)
        if np.any(small_mask):
            den[small_mask] = (c ** 6) * (ik[small_mask] ** 2) / 9.0

    return den * num


def model(iw, params: Dict[str, Any], integrator: str = "grid", grid_size: int = 4000,
          ik_min: float = 0.0, ik_max: float = 1.0, quad_opts: Optional[Dict] = None,
          kernel: str = 'PCM_Fano_Bessel'):
    """Compute the model y-values for input `iw` using the integral-defined kernel.

    Returns y = N * integral_{ik=ik_min..ik_max} kernel_integrand(ik, iw, params) d(ik) + y0
    """
    iw_arr = np.atleast_1d(iw).astype(float)

    if integrator == "grid":
        ik = np.linspace(ik_min, ik_max, int(grid_size))
        ik_mesh = ik[:, None]
        iw_mesh = iw_arr[None, :]
        vals = kernel_integrand(ik_mesh, iw_mesh, params, kernel=kernel)
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
            f = lambda k: float(kernel_integrand(np.float64(k), np.float64(wi), params, kernel=kernel))
            res, err = quad(f, ik_min, ik_max, **quad_opts)
            integral_list.append(res)
        integral = np.array(integral_list)
    else:
        raise ValueError(f"Unknown integrator '{integrator}'. Use 'grid' or 'quad'.")

    N = float(_get(params, 'N', 1.0))
    y0 = float(_get(params, 'y0', 0.0))
    # Kernel-specific amplitude scaling:
    # - PCM_Fano_Bessel: Y = y0 + N/((q^2 + 1)*L^3) * integral
    # - PCM_Fano_Gauss:  Y = y0 + N*L^3/(q^2 + 1) * integral
    q_for_denom = float(_get(params, 'q', 0.0))
    L_for_denom = float(_get(params, 'L', 1.0))
    ksel = str(kernel).strip().lower() if kernel is not None else 'pcm_fano_bessel'
    if ksel in ('pcm_fano_bessel', 'sinc', 'bessel'):
        denom = (q_for_denom ** 2 + 1.0) * (L_for_denom ** 3)
        if denom == 0.0:
            denom = 1e-24
        return y0 + (N / denom) * integral
    elif ksel in ('pcm_fano_gauss', 'exp', 'gauss', 'gaussian_exp'):
        denom = (q_for_denom ** 2 + 1.0)
        if denom == 0.0:
            denom = 1e-24
        # apply alpha scaling factor: multiply by alpha^(-4/3)
        alpha_for_scale = float(_get(params, 'alpha', 1.0))
        # guard against non-positive alpha to avoid division by zero
        if alpha_for_scale <= 0.0:
            alpha_for_scale = 1e-24
        alpha_factor = alpha_for_scale ** (-4.0 / 3.0)
        return y0 + (N * (L_for_denom ** 3) / denom) * alpha_factor * integral
    else:
        # Fallback to previous conservative scaling
        denom = (q_for_denom ** 2 + 1.0) * (L_for_denom ** 3)
        if denom == 0.0:
            denom = 1e-24
        return (N / denom) * integral + y0
