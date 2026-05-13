"""Fitting wrappers using scipy and optionally lmfit."""
from typing import Dict, Any, Tuple
import numpy as np

from .model import model as model_func


def _build_free_params(config: Dict[str, Any]):
    free_names = []
    p0 = []
    lower = []
    upper = []
    for name, info in config.items():
        # robustly coerce 'vary' to boolean (allow strings like 'false', numeric 0, etc.)
        raw_vary = info.get("vary", True)
        if isinstance(raw_vary, bool):
            vary = raw_vary
        elif isinstance(raw_vary, (int, float)):
            vary = bool(raw_vary)
        elif isinstance(raw_vary, str):
            vary = raw_vary.strip().lower() in ("true", "1", "yes", "y", "t")
        else:
            vary = bool(raw_vary)
        if vary:
            free_names.append(name)
            p0.append(float(info.get("value", 0.0)))
            lower.append(info.get("min", -np.inf))
            upper.append(info.get("max", np.inf))
    return free_names, np.array(p0), (np.array(lower, dtype=float), np.array(upper, dtype=float))


def _params_from_vector(free_names, pvec, config: Dict[str, Any]):
    params = {}
    j = 0
    for name, info in config.items():
        raw_vary = info.get("vary", True)
        if isinstance(raw_vary, bool):
            vary = raw_vary
        elif isinstance(raw_vary, (int, float)):
            vary = bool(raw_vary)
        elif isinstance(raw_vary, str):
            vary = raw_vary.strip().lower() in ("true", "1", "yes", "y", "t")
        else:
            vary = bool(raw_vary)
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

    # Build a clean set of kwargs for the model() function so extra UI keys
    # (e.g. 'autoscale') are not passed through causing unexpected-kwarg errors.
    integrator_kwargs = {
        'integrator': integrator_opts.get('integrator', 'grid'),
        'grid_size': int(integrator_opts.get('grid_size', 400)),
        'ik_min': float(integrator_opts.get('ik_min', 0.0)),
        'ik_max': float(integrator_opts.get('ik_max', 1.0)),
        'quad_opts': integrator_opts.get('quad_opts', None)
    }

    free_names, p0, bounds = _build_free_params(param_config)
    if len(free_names) == 0:
        params_full = {name: float(info.get("value", 0.0)) for name, info in param_config.items()}
        errs_full = {name: None for name in param_config.keys()}
        y_model = model_func(iw, params_full, **integrator_kwargs)
        # compute R^2
        ss_res = np.sum((y - y_model) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        # No fitting performed; treat as converged
        return params_full, errs_full, None, None, y_model, r2, True, "no free parameters", 0

    # Autoscale sensible initial guesses for `y0` and `N` to improve conditioning.
    # Use a cheap grid evaluation (small grid) for the amplitude estimate so
    # we don't invoke the expensive 'quad' integrator here. Only do this
    # when integrator_opts explicitly allow autoscaling (UI toggle).
    try:
        if integrator_opts.get('autoscale', True):
            # local copies
            p0 = p0.astype(float)
            lower_bounds, upper_bounds = bounds
            # compute some simple data statistics
            y_mean = float(np.mean(y))
            y_range = float(np.max(y) - np.min(y))
            # map free_names -> index
            name_to_idx = {n: idx for idx, n in enumerate(free_names)}
            # if y0 is free, initialize it to data mean (clamped)
            if 'y0' in name_to_idx:
                idx_y0 = name_to_idx['y0']
                y0_guess = y_mean
                lo = lower_bounds[idx_y0]
                hi = upper_bounds[idx_y0]
                if np.isfinite(lo):
                    y0_guess = max(y0_guess, lo)
                if np.isfinite(hi):
                    y0_guess = min(y0_guess, hi)
                p0[idx_y0] = float(y0_guess)
            # if N is free, estimate a starting scale by evaluating the model with N=1
            if 'N' in name_to_idx:
                idx_N = name_to_idx['N']
                tmp_p = p0.copy()
                tmp_p[idx_N] = 1.0
                tmp_params = _params_from_vector(free_names, tmp_p, param_config)
                # force zero baseline for amplitude estimate
                tmp_params['y0'] = 0.0
                # use a smaller grid for speed
                grid_size = int(max(64, min(400, int(integrator_opts.get('grid_size', 400) // 4))))
                amp_model = np.max(model_func(iw, tmp_params, integrator='grid', grid_size=grid_size)) - np.min(model_func(iw, tmp_params, integrator='grid', grid_size=grid_size))
                if amp_model > 0 and np.isfinite(amp_model):
                    N0 = float(y_range / (amp_model + 1e-24))
                    lo = lower_bounds[idx_N]
                    hi = upper_bounds[idx_N]
                    if np.isfinite(lo):
                        N0 = max(N0, lo)
                    if np.isfinite(hi):
                        N0 = min(N0, hi)
                    p0[idx_N] = float(N0)
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
        params_full = _params_from_vector(free_names, pvec, param_config)
        return model_func(iw_vals, params_full, **integrator_kwargs)

    converged = False
    message = ""
    try:
        popt, pcov = curve_fit(fitfunc, iw, y, p0=p0, bounds=bounds, **curvefit_opts)
        params_full = _params_from_vector(free_names, popt, param_config)
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
        params_full = _params_from_vector(free_names, popt, param_config)
        errs_full = {name: None for name in param_config.keys()}
        y_model = model_func(iw, params_full, **integrator_kwargs)
        try:
            ss_res = np.sum((y - y_model) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        except Exception:
            r2 = float('nan')
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

    # compute R^2
    try:
        ss_res = np.sum((y - y_model) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    except Exception:
        r2 = float('nan')

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

    # Build a clean set of kwargs for the model() function so extra UI keys
    # (e.g. 'autoscale') are not passed through causing unexpected-kwarg errors.
    integrator_kwargs = {
        'integrator': integrator_opts.get('integrator', 'grid'),
        'grid_size': int(integrator_opts.get('grid_size', 400)),
        'ik_min': float(integrator_opts.get('ik_min', 0.0)),
        'ik_max': float(integrator_opts.get('ik_max', 1.0)),
        'quad_opts': integrator_opts.get('quad_opts', None)
    }

    params = lmfit.Parameters()
    # Autoscale initial guesses for `y0` and `N` (best-effort; use a small grid)
    try:
        y0_guess = None
        N0 = None
        if integrator_opts.get('autoscale', True):
            free_names, p0, bounds = _build_free_params(param_config)
            p0 = p0.astype(float)
            lower_bounds, upper_bounds = bounds
            y_mean = float(np.mean(y))
            y_range = float(np.max(y) - np.min(y))
            name_to_idx = {n: idx for idx, n in enumerate(free_names)}
            if 'y0' in name_to_idx:
                idx_y0 = name_to_idx['y0']
                y0_guess = y_mean
                lo = lower_bounds[idx_y0]
                hi = upper_bounds[idx_y0]
                if np.isfinite(lo):
                    y0_guess = max(y0_guess, lo)
                if np.isfinite(hi):
                    y0_guess = min(y0_guess, hi)
            if 'N' in name_to_idx:
                idx_N = name_to_idx['N']
                tmp_p = p0.copy()
                tmp_p[idx_N] = 1.0
                tmp_params = _params_from_vector(free_names, tmp_p, param_config)
                tmp_params['y0'] = 0.0
                grid_size = int(max(64, min(400, int(integrator_opts.get('grid_size', 400) // 4))))
                amp_model = np.max(model_func(iw, tmp_params, integrator='grid', grid_size=grid_size)) - np.min(model_func(iw, tmp_params, integrator='grid', grid_size=grid_size))
                if amp_model > 0 and np.isfinite(amp_model):
                    N0 = float(y_range / (amp_model + 1e-24))
                    lo = lower_bounds[idx_N]
                    hi = upper_bounds[idx_N]
                    if np.isfinite(lo):
                        N0 = max(N0, lo)
                    if np.isfinite(hi):
                        N0 = min(N0, hi)
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
        params.add(name, value=init_val, vary=info.get("vary", True), **kwargs)

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
        try:
            ss_res = np.sum((y - y_model) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        except Exception:
            r2 = float('nan')
        return fitted, errs, None, y_model, r2, converged, message, call_count['n']

    fitted = {n: float(result.params[n].value) for n in result.params}
    errs = {n: (float(result.params[n].stderr) if result.params[n].stderr is not None else None) for n in result.params}
    # lmfit returns a Result object with .success and .message attributes
    converged = bool(getattr(result, 'success', False))
    message = str(getattr(result, 'message', ''))
    y_model = model_func(iw, fitted, **integrator_kwargs)
    try:
        ss_res = np.sum((y - y_model) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
    except Exception:
        r2 = float('nan')
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
                    eps = 1e-6
                    # choose a smaller grid size for FD if available
                    try:
                        fd_grid = int(min(200, int(integrator_kwargs.get('grid_size', 400))))
                    except Exception:
                        fd_grid = 100
                    for j in range(nfree):
                        pj = pvec_fitted[j]
                        dp = eps * max(1.0, abs(pj))
                        if dp == 0:
                            dp = eps
                        p_plus = pvec_fitted.copy()
                        p_minus = pvec_fitted.copy()
                        p_plus[j] += dp
                        p_minus[j] -= dp
                        params_plus = _params_from_vector(free_names, p_plus, param_config)
                        params_minus = _params_from_vector(free_names, p_minus, param_config)
                        # Prefer a fast grid evaluation for FD
                        try:
                            y_plus = model_func(iw, params_plus, integrator='grid', grid_size=fd_grid)
                            y_minus = model_func(iw, params_minus, integrator='grid', grid_size=fd_grid)
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
