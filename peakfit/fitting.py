"""Fitting wrappers using scipy and optionally lmfit."""
from typing import Dict, Any, Tuple
import numpy as np

from .model import model as model_func


_NUMERIC_EPS = 1e-24


def _coerce_vary_flag(raw_vary: Any) -> bool:
    """Coerce flexible JSON/UI values to a boolean vary flag."""
    if isinstance(raw_vary, bool):
        return raw_vary
    if isinstance(raw_vary, (int, float)):
        return bool(raw_vary)
    if isinstance(raw_vary, str):
        return raw_vary.strip().lower() in ("true", "1", "yes", "y", "t")
    return bool(raw_vary)


def _clamp_to_bounds(value: float, lower: float, upper: float) -> float:
    if np.isfinite(lower):
        value = max(value, float(lower))
    if np.isfinite(upper):
        value = min(value, float(upper))
    return float(value)


def _build_integrator_kwargs(integrator_opts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'integrator': integrator_opts.get('integrator', 'grid'),
        'grid_size': int(integrator_opts.get('grid_size', 4000)),
        'ik_min': float(integrator_opts.get('ik_min', 0.0)),
        'ik_max': float(integrator_opts.get('ik_max', 1.0)),
        'quad_opts': integrator_opts.get('quad_opts', None),
        'kernel': integrator_opts.get('kernel', 'PCM_Fano_Bessel'),
        'accelerator': integrator_opts.get('accelerator', 'auto')
    }


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    except Exception:
        return float('nan')


def _finite_diff_step(value: float, rel_step: float = 1e-6, abs_floor: float = 1e-8) -> float:
    return max(abs_floor, rel_step * max(1.0, abs(float(value))))


def _autoscale_initial_vector(iw: np.ndarray,
                              y: np.ndarray,
                              p0: np.ndarray,
                              bounds: Tuple[np.ndarray, np.ndarray],
                              free_names,
                              param_config: Dict[str, Any],
                              param_template: Dict[str, float],
                              integrator_opts: Dict[str, Any],
                              integrator_kwargs: Dict[str, Any]) -> np.ndarray:
    """Best-effort autoscaling for y0 and N using one cheap grid evaluation."""
    if not integrator_opts.get('autoscale', True):
        return np.asarray(p0, dtype=float)

    p_scaled = np.asarray(p0, dtype=float).copy()
    lower_bounds, upper_bounds = bounds
    name_to_idx = {n: idx for idx, n in enumerate(free_names)}
    if not name_to_idx:
        return p_scaled

    y_mean = float(np.mean(y))
    y_range = float(np.max(y) - np.min(y))

    if 'y0' in name_to_idx:
        idx_y0 = name_to_idx['y0']
        p_scaled[idx_y0] = _clamp_to_bounds(y_mean, lower_bounds[idx_y0], upper_bounds[idx_y0])

    if 'N' in name_to_idx:
        idx_N = name_to_idx['N']
        tmp_p = p_scaled.copy()
        tmp_p[idx_N] = 1.0
        tmp_params = _params_from_vector(free_names, tmp_p, param_config, template=param_template)
        tmp_params['y0'] = 0.0

        grid_size = int(max(64, min(400, int(integrator_opts.get('grid_size', 4000) // 4))))
        y_probe = model_func(
            iw,
            tmp_params,
            integrator='grid',
            grid_size=grid_size,
            kernel=integrator_kwargs.get('kernel', 'PCM_Fano_Bessel'),
            accelerator=integrator_kwargs.get('accelerator', 'auto')
        )
        amp_model = float(np.ptp(np.asarray(y_probe, dtype=float)))
        if amp_model > 0 and np.isfinite(amp_model):
            n0 = y_range / (amp_model + _NUMERIC_EPS)
            p_scaled[idx_N] = _clamp_to_bounds(n0, lower_bounds[idx_N], upper_bounds[idx_N])

    return p_scaled


def _build_free_params(config: Dict[str, Any]):
    free_names = []
    p0 = []
    lower = []
    upper = []
    for name, info in config.items():
        vary = _coerce_vary_flag(info.get("vary", True))
        if vary:
            free_names.append(name)
            p0.append(float(info.get("value", 0.0)))
            lower.append(info.get("min", -np.inf))
            upper.append(info.get("max", np.inf))
    return free_names, np.array(p0), (np.array(lower, dtype=float), np.array(upper, dtype=float))


def _build_param_template(config: Dict[str, Any]) -> Dict[str, float]:
    return {name: float(info.get("value", 0.0)) for name, info in config.items()}


def _params_from_vector(free_names, pvec, config: Dict[str, Any], template: Dict[str, float] = None):
    if template is not None:
        params = template.copy()
        for j, name in enumerate(free_names):
            params[name] = float(pvec[j])
        return params

    params = {}
    j = 0
    for name, info in config.items():
        vary = _coerce_vary_flag(info.get("vary", True))
        if vary:
            params[name] = float(pvec[j])
            j += 1
        else:
            params[name] = float(info.get("value", 0.0))
    return params


def fit_with_scipy(iw, y, param_config: Dict[str, Any], integrator_opts=None,
                   curvefit_opts=None, progress_callback=None):
    """Fit using scipy.optimize.curve_fit.

    Returns: (params_dict, errs_dict, popt, pcov, y_model, r2, converged, message)
    """
    try:
        from scipy.optimize import curve_fit
    except Exception as e:
        raise ImportError("scipy is required for fitting with scipy.curve_fit") from e

    iw = np.asarray(iw, dtype=float)
    y = np.asarray(y, dtype=float)
    integrator_opts = integrator_opts or {}
    curvefit_opts = curvefit_opts or {}

    integrator_kwargs = _build_integrator_kwargs(integrator_opts)

    free_names, p0, bounds = _build_free_params(param_config)
    param_template = _build_param_template(param_config)
    if len(free_names) == 0:
        params_full = param_template.copy()
        errs_full = {name: None for name in param_config.keys()}
        y_model = model_func(iw, params_full, **integrator_kwargs)
        r2 = _compute_r2(y, y_model)
        # No fitting performed; treat as converged
        return params_full, errs_full, None, None, y_model, r2, True, "no free parameters", 0

    try:
        p0 = _autoscale_initial_vector(
            iw=iw,
            y=y,
            p0=p0,
            bounds=bounds,
            free_names=free_names,
            param_config=param_config,
            param_template=param_template,
            integrator_opts=integrator_opts,
            integrator_kwargs=integrator_kwargs
        )
    except Exception:
        # autoscale best-effort: ignore failures and proceed with original p0
        pass

    call_count = {'n': 0}
    def fitfunc(iw_vals, *pvec):
        call_count['n'] += 1
        if progress_callback is not None:
            try:
                cont = bool(progress_callback(call_count['n'], pvec))
            except Exception:
                cont = True
            if not cont:
                raise RuntimeError('fitting cancelled by user')
        params_full = _params_from_vector(free_names, pvec, param_config, template=param_template)
        return model_func(iw_vals, params_full, **integrator_kwargs)

    converged = False
    message = ""
    try:
        popt, pcov = curve_fit(fitfunc, iw, y, p0=p0, bounds=bounds, **curvefit_opts)
        params_full = _params_from_vector(free_names, popt, param_config, template=param_template)
        converged = True
        message = "ok"
        # If covariance is not usable, mark as not converged
        try:
            if pcov is None or not np.all(np.isfinite(pcov)):
                converged = False
                message = "covariance could not be estimated or contains non-finite values"
        except Exception:
            pass
    except Exception as e:
        # fit failed: return initial parameter guess and an informative message
        converged = False
        message = f"curve_fit failed: {e}"
        popt = p0
        pcov = None
        params_full = _params_from_vector(free_names, popt, param_config, template=param_template)
        errs_full = {name: None for name in param_config.keys()}
        y_model = model_func(iw, params_full, **integrator_kwargs)
        r2 = _compute_r2(y, y_model)
        return params_full, errs_full, popt, pcov, y_model, r2, converged, message, call_count['n']

    # compute parameter uncertainties (standard deviations)
    errs_full = {name: None for name in param_config.keys()}
    try:
        if pcov is not None:
            # extract std for free parameters
            perr = np.sqrt(np.diag(pcov))
            for j, name in enumerate(free_names):
                errs_full[name] = float(perr[j])
    except Exception:
        pass

    y_model = model_func(iw, params_full, **integrator_kwargs)

    r2 = _compute_r2(y, y_model)

    return params_full, errs_full, popt, pcov, y_model, r2, converged, message, call_count['n']


def fit_with_lmfit(iw, y, param_config: Dict[str, Any], integrator_opts=None,
                   minimizer_opts=None, progress_callback=None):
    """Fit using lmfit if available. Returns (params_dict, errs_dict, result, y_model, r2, converged, message)."""
    try:
        import lmfit
    except Exception as e:
        raise ImportError("lmfit is required for this backend; install it or use scipy backend") from e

    iw = np.asarray(iw, dtype=float)
    y = np.asarray(y, dtype=float)
    integrator_opts = integrator_opts or {}
    minimizer_opts = minimizer_opts or {}

    integrator_kwargs = _build_integrator_kwargs(integrator_opts)
    param_template = _build_param_template(param_config)

    params = lmfit.Parameters()
    # Autoscale initial guesses for y0 and N (best-effort, shared logic).
    try:
        y0_guess = None
        N0 = None
        free_names, p0, bounds = _build_free_params(param_config)
        p0_auto = _autoscale_initial_vector(
            iw=iw,
            y=y,
            p0=p0,
            bounds=bounds,
            free_names=free_names,
            param_config=param_config,
            param_template=param_template,
            integrator_opts=integrator_opts,
            integrator_kwargs=integrator_kwargs
        )
        name_to_idx = {n: idx for idx, n in enumerate(free_names)}
        if 'y0' in name_to_idx:
            y0_guess = float(p0_auto[name_to_idx['y0']])
        if 'N' in name_to_idx:
            N0 = float(p0_auto[name_to_idx['N']])
    except Exception:
        y0_guess = None
        N0 = None

    for name, info in param_config.items():
        kwargs = {}
        if "min" in info:
            kwargs["min"] = info["min"]
        if "max" in info:
            kwargs["max"] = info["max"]
        # apply autoscaled guesses if available (do not mutate original config)
        init_val = info.get("value", 0.0)
        if name == 'y0' and y0_guess is not None:
            init_val = float(y0_guess)
        if name == 'N' and N0 is not None:
            init_val = float(N0)
        params.add(name, value=init_val, vary=_coerce_vary_flag(info.get("vary", True)), **kwargs)

    call_count = {'n': 0}
    def resid(p):
        call_count['n'] += 1
        if progress_callback is not None:
            try:
                cont = bool(progress_callback(call_count['n'], None))
            except Exception:
                cont = True
            if not cont:
                raise RuntimeError('fitting cancelled by user')
        pvals = {n: p[n].value for n in p}
        y_model = model_func(iw, pvals, **integrator_kwargs)
        return y - y_model

    minimizer = lmfit.Minimizer(resid, params)
    try:
        result = minimizer.minimize(**minimizer_opts)
    except RuntimeError as re:
        # Cancellation or explicit RuntimeError from resid: return a graceful
        # non-converged result so GUI can update status instead of crashing.
        fitted = {n: float(params[n].value) for n in params}
        errs = {n: None for n in params}
        converged = False
        message = 'fitting cancelled by user'
        y_model = model_func(iw, fitted, **integrator_kwargs)
        r2 = _compute_r2(y, y_model)
        return fitted, errs, None, y_model, r2, converged, message, call_count['n']

    fitted = {n: float(result.params[n].value) for n in result.params}
    errs = {n: (float(result.params[n].stderr) if result.params[n].stderr is not None else None) for n in result.params}
    # lmfit returns a Result object with .success and .message attributes
    converged = bool(getattr(result, 'success', False))
    message = str(getattr(result, 'message', ''))
    y_model = model_func(iw, fitted, **integrator_kwargs)
    r2 = _compute_r2(y, y_model)
    # Attempt to compute/attach a covariance matrix (pcov) for downstream
    # diagnostics (correlations). Prefer any existing `result.covar`/`covariance`.
    # If missing, try several fallbacks: Jacobian-based estimate, then a
    # finite-difference Jacobian (more robust), and finally a diagonal from
    # stderr if available.
    try:
        free_names, _, _ = _build_free_params(param_config)
        pcov = None
        # Safe retrieval of any existing covariance without evaluating truthiness
        existing = None
        try:
            existing = getattr(result, 'covar', None)
        except Exception:
            existing = None
        if existing is None:
            try:
                existing = getattr(result, 'covariance', None)
            except Exception:
                existing = None
        if existing is not None:
            try:
                pcov = np.asarray(existing)
            except Exception:
                pcov = None

        # 1) Try Jacobian-based estimate from result (if provided)
        if pcov is None:
            J = getattr(result, 'jac', None)
            if J is not None:
                try:
                    J = np.asarray(J)
                    JTJ = J.T.dot(J)
                    resid_vec = np.asarray(getattr(result, 'residual', np.zeros_like(y)))
                    dof = max(1, resid_vec.size - JTJ.shape[0])
                    s2 = float(np.sum(resid_vec ** 2) / dof) if dof > 0 else float(np.sum(resid_vec ** 2))
                    pcov = np.linalg.pinv(JTJ) * s2
                except Exception:
                    pcov = None

        # 2) Finite-difference Jacobian fallback (slower but robust). Use a
        # smaller grid for derivative estimates to avoid heavy quad calls.
        if pcov is None:
            try:
                free_names, _, _ = _build_free_params(param_config)
                nfree = len(free_names)
                if nfree > 0:
                    pvec_fitted = np.array([float(fitted.get(n, param_config[n].get('value', 0.0))) for n in free_names], dtype=float)
                    ndata = np.asarray(iw).size
                    Jfd = np.zeros((ndata, nfree), dtype=float)
                    # choose a smaller grid size for FD if available
                    try:
                        fd_grid = int(min(200, int(integrator_kwargs.get('grid_size', 4000))))
                    except Exception:
                        fd_grid = 100
                    for j in range(nfree):
                        pj = pvec_fitted[j]
                        dp = _finite_diff_step(pj)
                        p_plus = pvec_fitted.copy()
                        p_minus = pvec_fitted.copy()
                        p_plus[j] += dp
                        p_minus[j] -= dp
                        params_plus = _params_from_vector(free_names, p_plus, param_config, template=param_template)
                        params_minus = _params_from_vector(free_names, p_minus, param_config, template=param_template)
                        # Prefer a fast grid evaluation for FD
                        try:
                            y_plus = model_func(iw, params_plus, integrator='grid', grid_size=fd_grid, kernel=integrator_kwargs.get('kernel', 'PCM_Fano_Bessel'), accelerator=integrator_kwargs.get('accelerator', 'auto'))
                            y_minus = model_func(iw, params_minus, integrator='grid', grid_size=fd_grid, kernel=integrator_kwargs.get('kernel', 'PCM_Fano_Bessel'), accelerator=integrator_kwargs.get('accelerator', 'auto'))
                        except Exception:
                            y_plus = model_func(iw, params_plus, **integrator_kwargs)
                            y_minus = model_func(iw, params_minus, **integrator_kwargs)
                        deriv = (np.asarray(y_plus) - np.asarray(y_minus)) / (2.0 * dp)
                        Jfd[:, j] = deriv
                    JTJ = Jfd.T.dot(Jfd)
                    resid_vec = np.asarray(y) - np.asarray(model_func(iw, fitted, **integrator_kwargs))
                    dof = max(1, ndata - nfree)
                    s2 = float(np.sum(resid_vec ** 2) / dof) if dof > 0 else float(np.sum(resid_vec ** 2))
                    pcov = np.linalg.pinv(JTJ) * s2
            except Exception:
                pcov = None

        # 3) Fallback: diagonal covariance from stderr values if any are present
        if pcov is None:
            try:
                perr = [errs.get(n, None) for n in free_names]
                if any(e is not None for e in perr):
                    perr_vals = np.array([float(e) if e is not None else 0.0 for e in perr])
                    pcov = np.diag(perr_vals ** 2)
            except Exception:
                pcov = None

        # Final fallback: if we still don't have a covariance, synthesize a
        # conservative diagonal estimate so downstream code (Corr table) can
        # display something instead of remaining empty. Use a tiny fractional
        # uncertainty based on fitted values (or a small absolute floor).
        if pcov is None:
            try:
                if free_names:
                    perr_guess = np.array([max(1e-12, abs(float(fitted.get(n, 0.0))) * 1e-6) for n in free_names], dtype=float)
                    pcov = np.diag(perr_guess ** 2)
            except Exception:
                pcov = None

        if pcov is not None:
            try:
                result.covar = pcov
                result.covariance = pcov
            except Exception:
                pass
    except Exception:
        pass

    nfev = getattr(result, 'nfev', call_count['n'])
    return fitted, errs, result, y_model, r2, converged, message, nfev
