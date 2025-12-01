import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# -------- Empirical features --------
def realized_var(returns, window=21):
    rv = returns**2
    return rv.rolling(window=window, min_periods=window).mean()

def leverage_corr(returns, rv_future):
    delta = rv_future.diff()
    valid = ~np.isnan(delta)
    if valid.sum() < 20:
        return np.nan
    return np.corrcoef(returns[valid], delta[valid])[0,1]

def acf_squared(returns, lag=1):
    x = returns**2
    x = x - x.mean()
    if len(x) < lag + 2:
        return np.nan
    num = np.sum(x[lag:] * x[:-lag])
    den = np.sum(x * x)
    return num / den if den > 0 else np.nan

def central_moments(x):
    x = pd.Series(x).dropna().values
    m = np.mean(x); c = x - m
    var = np.mean(c**2)
    skew = np.mean(c**3) / (var**1.5) if var > 0 else np.nan
    kurt = np.mean(c**4) / (var**2) if var > 0 else np.nan
    exkurt = kurt - 3 if kurt is not np.nan else np.nan
    return {'mean': m, 'var': var, 'skew': skew, 'exkurt': exkurt}

# -------- Model feature proxies under Heston (physical measure) --------
def feature_residuals(params, dt, empirical, weights=None, feller_penalty_weight=10.0):
    v0, kappa, theta, xi, rho, mu = params

    # Bounds guardrails (least_squares also enforces bounds, but we short-circuit absurd cases)
    if theta <= 0 or kappa <= 0 or xi <= 0 or abs(rho) > 1:
        return np.full(7, 1e6)

    # Theoretical proxies (tractable approximations)
    mean_ret_model = mu * dt
    var_ret_model  = theta * dt
    acf2_model     = np.exp(-kappa * dt)
    rv_var_model   = xi**2 * theta * dt
    lev_model      = rho
    # Skew/kurt proxies: diffusion Heston yields near-zero skew and small exkurt
    # Introduce simple param-driven proxies to pull xi and rho to reflect tails:
    skew_model     = -0.5 * rho * (xi / np.sqrt(max(theta, 1e-12)))  # sign via leverage, scale via vol-of-vol
    exkurt_model   = 0.5 * (xi**2 / max(theta, 1e-12))               # more vol-of-vol => heavier tails (approx)

    model = np.array([
        mean_ret_model,
        var_ret_model,
        acf2_model,
        rv_var_model,
        lev_model,
        skew_model,
        exkurt_model
    ])
    emp = np.array([
        empirical['mean_ret'],
        empirical['var_ret'],
        empirical['acf2_lag1'],
        empirical['rv_var'],
        empirical['lev_corr'],
        empirical['skew'],
        empirical['exkurt']
    ])

    # Residuals
    res = model - emp

    # Weights
    if weights is None:
        # Heavier weights on variance and tails; moderate on ACF and leverage
        weights = np.array([1.0, 3.0, 1.5, 2.0, 1.5, 3.0, 3.0])

    res = res * weights

    # Feller soft penalty: max(0, xi^2 - 2*kappa*theta)
    feller_violation = max(0.0, xi**2 - 2.0 * kappa * theta)
    # Add as an extra residual (scaled)
    penalty = feller_penalty_weight * feller_violation
    return np.append(res, penalty)

def calibrate_asset_moment_matching(daily_returns, dt=1.0/252):
    r = pd.Series(daily_returns).dropna()
    if len(r) < 300:
        raise ValueError("Need at least ~300 daily points for robust moment matching.")
    n = len(r)

    # Empirical features
    cm = central_moments(r)
    mean_ret = cm['mean']
    var_ret  = cm['var']
    skew     = cm['skew'] if not np.isnan(cm['skew']) else 0.0
    exkurt   = cm['exkurt'] if not np.isnan(cm['exkurt']) else 0.0

    acf2_lag1 = acf_squared(r, lag=1)
    rv = realized_var(r, window=21)
    rv_future = rv.shift(-1)
    lev_corr = leverage_corr(r, rv_future)
    rv_var = rv.var(ddof=1)

    empirical = {
        'mean_ret': mean_ret,
        'var_ret': var_ret,
        'acf2_lag1': acf2_lag1 if not np.isnan(acf2_lag1) else 0.2,
        'rv_var': rv_var if not np.isnan(rv_var) else max(var_ret**2, 1e-8),
        'lev_corr': lev_corr if not np.isnan(lev_corr) else -0.2,
        'skew': skew,
        'exkurt': exkurt
    }

    # Initial guesses from simple mappings
    mu0 = mean_ret / dt
    theta0 = var_ret / dt
    kappa0 = 2.0
    xi0 = 0.5
    rho0 = -0.3
    v0_0 = theta0

    x0 = np.array([v0_0, kappa0, theta0, xi0, rho0, mu0])

    # Bounds
    lb = np.array([1e-8, 1e-4, 1e-8, 1e-4, -0.99, -np.inf])
    ub = np.array([np.inf,  50.0,  np.inf,  5.0,   0.99,   np.inf])

    res = least_squares(feature_residuals, x0,
                        bounds=(lb, ub),
                        args=(dt, empirical),
                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=50000)
    v0, kappa, theta, xi, rho, mu = res.x
    return {
        'v0': v0, 'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho, 'mu': mu,
        'success': res.success, 'cost': res.cost, 'n': n
    }

# -------- Batch over assets --------
def recalibrate_all(asset_returns_dict):
    results = []
    for asset, ret in asset_returns_dict.items():
        p = calibrate_asset_moment_matching(ret)
        p['Asset'] = asset
        results.append(p)
    return pd.DataFrame(results)[['Asset','v0','kappa','theta','xi','rho','mu','cost','success','n']]