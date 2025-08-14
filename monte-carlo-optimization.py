"""
Monte Carlo portfolio optimizer (no whole-share allocation)

- Tickers come from STARTING_SHARES keys
- Cleans and aligns price data
- Optimizes weights using SLSQP with sum(w)=1 and w>=0 (no shorting by default)
- Objective: maximize mean_final - LAMBDA*CVaR_loss - GAMMA*prob_loss*investment
- Uses common random numbers for a deterministic objective (so optimizer doesn't stall)
- Compares current vs optimized distributions in one plot

Python 3.11+
pip install yfinance scipy numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
from numpy.linalg import cholesky, LinAlgError

# ----------------------------
# User settings
# ----------------------------

# Maintain tickers & starting lots here only (fractional shares OK)
STARTING_SHARES = {
    # 'TICKER': shares,
    'AMD': 0.9391,
    'AMZN': 0.5118,
    'COF': 0.3083,
    'ET': 4,
    'V': 0.2906,
}

TICKERS = list(STARTING_SHARES.keys())
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.today()

# Monte Carlo settings
SEED = 42
MC_DAYS = 365
MC_ITERS_OBJECTIVE = 20_000   # eval-time iterations (speed vs accuracy)
MC_ITERS_REPORT   = 100_000   # reporting/plot iterations

# Risk preferences (penalty-based objective)
USE_PENALTIES = True
LAMBDA = 1.0    # $ penalty per dollar of 95% CVaR loss
GAMMA  = 2.0    # penalty on (prob_loss * investment)

# Optional hard-constraint targets (only used if USE_PENALTIES=False)
TARGET_MAX_PROB_LOSS = 0.30   # <= 30% chance of loss
TARGET_MAX_CVAR_LOSS = 0.10   # <= 10% of investment as CVaR loss (95%)

# Weight bounds
ALLOW_SHORTING = False
SHORT_LIMIT = -0.10  # lower bound if shorting allowed (e.g., -10%)
MAX_W = 1.00         # cap any single name (e.g., 0.60 for 60% cap)
MIN_W = 0.00         # min weight (0.00 allows dropping a name)

# Numerical safety for covariance
RIDGE = 1e-10

# ----------------------------
# Helpers
# ----------------------------

def fetch_prices(tickers, start, end):
    # Use adjusted prices for log-returns; silence FutureWarning with explicit arg
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    prices = df['Close'] if 'Close' in df.columns else df
    prices = prices.dropna(how='all').dropna(axis=1, how='any')  # drop tickers with gaps
    return prices

def ensure_cov_psd(cov, ridge=RIDGE):
    cov = np.array(cov, dtype=float)
    try:
        cholesky(cov)
        return cov
    except LinAlgError:
        return cov + np.eye(cov.shape[0]) * ridge

def shares_to_weights(tickers, shares_vec, last_close):
    shares = np.asarray(shares_vec, dtype=float)
    prices = np.array([last_close[t] for t in tickers], dtype=float)
    values = shares * prices
    total = float(values.sum())
    if total <= 0:
        weights = np.ones(len(tickers)) / len(tickers)
        total = 1.0
    else:
        weights = values / total
    return weights, total, values

def risk_metrics(final_values, investment):
    mean_final = final_values.mean()
    VaR_95 = np.percentile(final_values, 5)
    CVaR_95 = final_values[final_values <= VaR_95].mean()
    prob_loss = (final_values < investment).mean()
    VaR_loss = max(0.0, investment - VaR_95)
    CVaR_loss = max(0.0, investment - CVaR_95)
    return dict(
        mean_final=mean_final,
        VaR_95=VaR_95,
        CVaR_95=CVaR_95,
        VaR_loss=VaR_loss,
        CVaR_loss=CVaR_loss,
        prob_loss=prob_loss,
    )

# Common-random-number MC for deterministic objective
def mc_with_common_normals(weights, mean_vec, L, Z, investment):
    # Z: (iters, days, n) standard normals
    mv = Z @ L.T + mean_vec  # broadcast mean across axes
    port_daily = mv @ np.asarray(weights)
    growth = np.exp(port_daily.sum(axis=1))
    return investment * growth

# Fresh Monte Carlo (for reporting/plots)
def monte_carlo(weights, mean_vec, cov_mat, num_days, num_iterations, investment, rng):
    draws = rng.multivariate_normal(mean_vec, cov_mat, (num_iterations, num_days))
    port_daily = draws @ np.asarray(weights)
    growth = np.exp(port_daily.sum(axis=1))
    return investment * growth

# ----------------------------
# Main
# ----------------------------

def main():
    np.set_printoptions(suppress=True, precision=6)

    # --- Data ---
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)
    tickers = list(prices.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("No tickers with complete data after cleaning. Try forward-filling or different dates.")

    returns = np.log(prices / prices.shift(1)).dropna()
    last_close = prices.iloc[-1].astype(float)

    mean_vec = returns.mean().values
    cov_mat = ensure_cov_psd(returns.cov().values, ridge=RIDGE)

    # --- Align starting shares to surviving tickers ---
    aligned_shares = np.array([STARTING_SHARES.get(t, 0) for t in tickers], dtype=float)
    weights_current, total_value_current, start_values = shares_to_weights(tickers, aligned_shares, last_close)
    investment = float(total_value_current) if total_value_current > 0 else float(last_close.mean()) * 10.0

    print(f"Using {n} tickers: {tickers}")
    print(f"End date: {END_DATE}\n")
    print("Starting lots (aligned):")
    for t, s, v, w in zip(tickers, aligned_shares, start_values, weights_current):
        print(f"  {t}: {s:g} shares  -> ${v:,.2f}  ({w:.2%})")
    print(f"Total starting value: ${total_value_current:,.2f}\n")

    # --- Common random numbers for deterministic objective ---
    rng_obj = np.random.default_rng(SEED)
    Z = rng_obj.standard_normal(size=(MC_ITERS_OBJECTIVE, MC_DAYS, n))
    try:
        L = cholesky(cov_mat)
    except LinAlgError:
        L = cholesky(cov_mat + np.eye(n) * RIDGE)

    # --- Objective & constraints ---
    lower = 0.0 if not ALLOW_SHORTING else float(SHORT_LIMIT)
    bounds = [(max(MIN_W, lower), min(MAX_W, 1.0))] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]  # sum to 1

    def objective_penalty(weights):
        w = np.clip(np.asarray(weights, dtype=float), bounds[0][0], bounds[0][1])
        s = w.sum()
        w = (w / s) if s > 0 else np.ones(n)/n
        finals = mc_with_common_normals(w, mean_vec, L, Z, investment)
        m = risk_metrics(finals, investment)
        util = m['mean_final'] - LAMBDA * m['CVaR_loss'] - GAMMA * m['prob_loss'] * investment
        return -util

    def make_prob_loss_constraint(max_prob):
        def c(w):
            w = np.clip(np.asarray(w, dtype=float), bounds[0][0], bounds[0][1])
            s = w.sum()
            w = (w / s) if s > 0 else np.ones(n)/n
            finals = mc_with_common_normals(w, mean_vec, L, Z, investment)
            return max_prob - (finals < investment).mean()
        return c

    def make_cvar_constraint(max_frac_loss):
        def c(w):
            w = np.clip(np.asarray(w, dtype=float), bounds[0][0], bounds[0][1])
            s = w.sum()
            w = (w / s) if s > 0 else np.ones(n)/n
            finals = mc_with_common_normals(w, mean_vec, L, Z, investment)
            VaR_95 = np.percentile(finals, 5)
            CVaR_95 = finals[finals <= VaR_95].mean()
            CVaR_loss = max(0.0, investment - CVaR_95)
            return (max_frac_loss * investment) - CVaR_loss
        return c

    # Initial guesses
    inits = [np.ones(n)/n]  # equal-weight
    if np.isfinite(weights_current).all() and len(weights_current) == n:
        inits.append(weights_current)
    inits += [np.random.default_rng(SEED + k).dirichlet(np.ones(n)) for k in range(1, 6)]

    # Add hard constraints if not using penalties
    if not USE_PENALTIES:
        cons = cons + [
            {'type': 'ineq', 'fun': make_prob_loss_constraint(TARGET_MAX_PROB_LOSS)},
            {'type': 'ineq', 'fun': make_cvar_constraint(TARGET_MAX_CVAR_LOSS)},
        ]

    # --- Optimize ---
    best = None
    for x0 in inits:
        res = minimize(
            objective_penalty if USE_PENALTIES else (lambda w: 0.0),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-6, 'disp': False}
        )
        # If using hard constraints only, pick the feasible with highest mean
        if not USE_PENALTIES:
            if not res.success:
                continue
            w = res.x / res.x.sum()
            finals = mc_with_common_normals(w, mean_vec, L, Z, investment)
            score = -finals.mean()
            res = res.copy()
            res.fun = score
        if (best is None) or (res.fun < best.fun):
            best = res

    if best is None or not np.isfinite(best.x).all():
        raise RuntimeError("Optimization failed to find a feasible solution.")

    opt_weights = best.x / best.x.sum()

    # --- Report current vs optimized ---
    rng_report = np.random.default_rng(SEED + 10)
    sim_current = monte_carlo(weights_current, mean_vec, cov_mat, MC_DAYS, MC_ITERS_REPORT, investment, rng_report)
    met_current = risk_metrics(sim_current, investment)

    rng_report2 = np.random.default_rng(SEED + 11)
    sim_opt = monte_carlo(opt_weights, mean_vec, cov_mat, MC_DAYS, MC_ITERS_REPORT, investment, rng_report2)
    met_opt = risk_metrics(sim_opt, investment)

    print("--- Current portfolio (from your lots) ---")
    print(f"Mean Final Value: ${met_current['mean_final']:,.2f}")
    print(f"95% VaR loss:     ${met_current['VaR_loss']:,.2f}")
    print(f"95% CVaR loss:    ${met_current['CVaR_loss']:,.2f}")
    print(f"Prob. of loss:    {met_current['prob_loss']:.2%}")

    print("\n--- Optimized portfolio ---")
    print("Weights:")
    for t, w in zip(tickers, opt_weights):
        print(f"  {t}: {w:.2%}")
    print(f"Mean Final Value: ${met_opt['mean_final']:,.2f}")
    print(f"95% VaR loss:     ${met_opt['VaR_loss']:,.2f}")
    print(f"95% CVaR loss:    ${met_opt['CVaR_loss']:,.2f}")
    print(f"Prob. of loss:    {met_opt['prob_loss']:.2%}")

    # --- Compare distributions: current vs optimized ---
    bins = np.linspace(
        min(sim_current.min(), sim_opt.min()),
        max(sim_current.max(), sim_opt.max()),
        80
    )
    VaR_95_cur = np.percentile(sim_current, 5)
    VaR_95_opt = np.percentile(sim_opt, 5)
    mean_cur = sim_current.mean()
    mean_opt = sim_opt.mean()

    plt.figure(figsize=(10, 6))
    plt.hist(sim_current, bins=bins, alpha=0.45, density=True, label='Current')
    plt.hist(sim_opt,     bins=bins, alpha=0.45, density=True, label='Optimized')
    plt.axvline(VaR_95_cur, linestyle='dashed', linewidth=2, label='VaR 95% (Current)')
    plt.axvline(VaR_95_opt, linestyle='dashed', linewidth=2, label='VaR 95% (Optimized)')
    plt.axvline(mean_cur,   linestyle='solid',  linewidth=1.5, label='Mean (Current)')
    plt.axvline(mean_opt,   linestyle='solid',  linewidth=1.5, label='Mean (Optimized)')
    plt.title('Monte Carlo Final Value Distribution: Current vs Optimized')
    plt.xlabel('Final Portfolio Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
