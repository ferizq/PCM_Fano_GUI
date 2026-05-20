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
import re

def _get(params: Dict[str, Any], name: str, default: Any):
    """Retrieve a parameter value, accepting legacy 'i'-prefixed names.

    Prefers the canonical name (e.g. 'a') and falls back to 'ia' if present.
    Returns the value coerced to float or the provided default.
    """
    if params is None:
        return float(default)
    # direct match
    if name in params:
        return float(params.get(name))
    # try a canonical form with underscores removed (some loaders strip underscores)
    alt = name.replace('_', '')
    if alt in params:
        return float(params.get(alt))
    # legacy 'i' prefixed forms
    legacy = 'i' + name
    if legacy in params:
        return float(params.get(legacy))
    legacy_alt = 'i' + alt
    if legacy_alt in params:
        return float(params.get(legacy_alt))
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
    # model_components now returns (base, gauss_total, gauss_components, lorentz_total, lorentz_components)
    base, gauss_total, _, lorentz_total, _ = model_components(iw, params, integrator=integrator, grid_size=grid_size, ik_min=ik_min, ik_max=ik_max, quad_opts=quad_opts, kernel=kernel)
    return base + gauss_total + lorentz_total


def model_components(iw, params: Dict[str, Any], integrator: str = "grid", grid_size: int = 4000,
                     ik_min: float = 0.0, ik_max: float = 1.0, quad_opts: Optional[Dict] = None,
                     kernel: str = 'PCM_Fano_Bessel'):
    """Compute the model components separately.

    Returns a tuple ``(base, gauss_total, gauss_components, lorentz_total, lorentz_components)`` where
    ``base`` is the integral-derived contribution (including ``y0`` and ``N`` scaling),
    ``gauss_total`` is the sum of any additive Gaussian components and
    ``gauss_components`` is a list with each individual Gaussian array (may be
    empty when no Gaussians are provided). Similarly, ``lorentz_total`` is the
    sum of any additive Lorentzian components and ``lorentz_components`` is a
    list with each individual Lorentzian array.
    """
    iw_arr = np.atleast_1d(iw).astype(float)

    # Compute the integral over ik using the requested integrator
    if integrator == "grid":
        ik = np.linspace(ik_min, ik_max, int(grid_size))
        ik_mesh = ik[:, None]
        iw_mesh = iw_arr[None, :]
        vals = kernel_integrand(ik_mesh, iw_mesh, params, kernel=kernel)
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

    # Base parameters
    N = float(_get(params, 'N', 1.0))
    y0 = float(_get(params, 'y0', 0.0))
    q_for_denom = float(_get(params, 'q', 0.0))
    L_for_denom = float(_get(params, 'L', 1.0))

    # Kernel-specific amplitude scaling
    ksel = str(kernel).strip().lower() if kernel is not None else 'pcm_fano_bessel'
    if ksel in ('pcm_fano_bessel', 'sinc', 'bessel'):
        denom = (q_for_denom ** 2 + 1.0) * (L_for_denom ** 3)
        if denom == 0.0:
            denom = 1e-24
        base = y0 + (N / denom) * integral
    elif ksel in ('pcm_fano_gauss', 'exp', 'gauss', 'gaussian_exp'):
        denom = (q_for_denom ** 2 + 1.0)
        if denom == 0.0:
            denom = 1e-24
        alpha_for_scale = float(_get(params, 'alpha', 1.0))
        if alpha_for_scale <= 0.0:
            alpha_for_scale = 1e-24
        alpha_factor = alpha_for_scale ** (-4.0 / 3.0)
        base = y0 + (N * (L_for_denom ** 3) / denom) * alpha_factor * integral
    else:
        denom = (q_for_denom ** 2 + 1.0) * (L_for_denom ** 3)
        if denom == 0.0:
            denom = 1e-24
        base = (N / denom) * integral + y0

    # Support multiple additive Gaussian and Lorentzian peaks. Detect parameter
    # names like Ng, Ng1, Ng2, x0g, x0g1, gg, gg1 for Gaussians and
    # Nl, Nl1, x0l, x0l1, gl, gl1 for Lorentzians.
    gauss_components = []
    gauss_total = np.zeros_like(iw_arr, dtype=float)
    lorentz_components = []
    lorentz_total = np.zeros_like(iw_arr, dtype=float)
    try:
        # Build a normalized key map for robust detection (strip leading 'i' and underscores)
        norm_map = {}
        if isinstance(params, dict):
            for k in params.keys():
                ks = str(k).lower()
                if ks.startswith('i') and len(ks) > 1:
                    ks = ks[1:]
                kn = ks.replace('_', '')
                if kn not in norm_map:
                    norm_map[kn] = k

        # Gather numeric suffix indices from both gaussian and lorentzian keys
        idxs = set()
        for kn in norm_map.keys():
            for pattern in (r'^ng(\d+)$', r'^x0(\d+)$', r'^x0g(\d+)$', r'^gg(\d+)$', r'^nl(\d+)$', r'^x0l(\d+)$', r'^gl(\d+)$'):
                m = re.match(pattern, kn)
                if m:
                    idxs.add(int(m.group(1)))

        # If no numbered components found, allow legacy unsuffixed names (single components)
        if not idxs and any(k in norm_map for k in ('ng', 'x0', 'x0g', 'gg', 'nl', 'x0l', 'gl')):
            idxs.add(1)

        # For each detected index, assemble gaussian and lorentzian components as available
        for idx in sorted(idxs):
            def _get_raw_value(candidates, fallback):
                raw = None
                for cand in candidates:
                    if cand in norm_map:
                        raw = params[norm_map[cand]]
                        break
                if raw is None:
                    try:
                        return float(_get(params, fallback + (str(idx) if idx > 1 else ''), 0.0))
                    except Exception:
                        return 0.0
                try:
                    if isinstance(raw, dict) and 'value' in raw:
                        return float(raw.get('value', 0.0))
                    return float(raw)
                except Exception:
                    try:
                        return float(_get(params, fallback + (str(idx) if idx > 1 else ''), 0.0))
                    except Exception:
                        return 0.0

            # Gaussian params
            Ng = _get_raw_value([f'ng{idx}'], 'ng')
            x0g = _get_raw_value([f'x0g{idx}', f'x0{idx}'], 'x0')
            gg = _get_raw_value([f'gg{idx}'], 'gg')
            try:
                if gg == 0.0:
                    gg = 1e-24
                comp_g = Ng * np.exp(-4.0 * np.log(2.0) * ((iw_arr - x0g) ** 2) / (gg * gg))
            except Exception:
                comp_g = np.zeros_like(iw_arr, dtype=float)
            gauss_components.append(comp_g)
            gauss_total = gauss_total + comp_g

            # Lorentzian params
            Nl = _get_raw_value([f'nl{idx}'], 'nl')
            x0l = _get_raw_value([f'x0l{idx}'], 'x0l')
            gl = _get_raw_value([f'gl{idx}'], 'gl')
            try:
                if gl == 0.0:
                    gl = 1e-24
                comp_l = Nl / ((2.0 * (iw_arr - x0l) / gl) ** 2 + 1.0)
            except Exception:
                comp_l = np.zeros_like(iw_arr, dtype=float)
            lorentz_components.append(comp_l)
            lorentz_total = lorentz_total + comp_l
    except Exception:
        # on failure fall back to legacy single-component behavior (gaussian)
        try:
            Ng = float(_get(params, 'N_g', 0.0))
            x0g = float(_get(params, 'x0', 0.0))
            gg = float(_get(params, 'gg', 0.0))
            if gg == 0.0:
                gg = 1e-24
            gauss_total = Ng * np.exp(-4.0 * np.log(2.0) * ((iw_arr - x0g) ** 2) / (gg * gg))
            gauss_components = [gauss_total]
        except Exception:
            gauss_total = np.zeros_like(iw_arr, dtype=float)
            gauss_components = []

    return base, gauss_total, gauss_components, lorentz_total, lorentz_components
