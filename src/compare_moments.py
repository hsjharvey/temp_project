import numpy as np
import pandas as pd

# ------------------------
# 1. Inputs
# ------------------------
assets = ["BTC", "ETH", "SOL", "BNB", "XRP"]
files = {
    "BTC": "BTC-AUD.csv",
    "ETH": "ETH-AUD.csv",
    "SOL": "SOL-AUD.csv",
    "BNB": "BNB-AUD.csv",
    "XRP": "XRP-AUD.csv",
}

# Calibrated Heston parameters (daily)
params = {
    "BTC": dict(mu=0.001669, kappa=2.385836, theta=0.000872, sigma_v=0.153398, rho=-0.135730, v0=0.000001),
    "ETH": dict(mu=0.002216, kappa=2.189822, theta=0.001606, sigma_v=0.218734, rho=-0.129252, v0=0.000002),
    "SOL": dict(mu=0.004433, kappa=1.441285, theta=0.004091, sigma_v=0.304743, rho=-0.205783, v0=0.033563),
    "BNB": dict(mu=0.002880, kappa=1.580702, theta=0.001827, sigma_v=0.482702, rho=-0.205482, v0=0.000023),
    "XRP": dict(mu=0.002606, kappa=2.414101, theta=0.002995, sigma_v=0.670663, rho=-0.325292, v0=0.000010),
}

T = 1000       # steps (days)
N = 10_000     # simulation paths


# ------------------------
# 2. Heston simulator (Euler, per asset)
# ------------------------
def simulate_heston(T, mu, kappa, theta, sigma_v, rho, v0, paths):
    dt = 1.0
    sqrt_dt = np.sqrt(dt)

    v = np.full((paths, T), v0, dtype=float)
    r = np.zeros((paths, T), dtype=float)

    for t in range(1, T):
        z1 = np.random.randn(paths)
        z2 = rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * np.random.randn(paths)

        v_prev = np.maximum(v[:, t-1], 0.0)
        v[:, t] = (
            v_prev
            + kappa * (theta - v_prev) * dt
            + sigma_v * np.sqrt(v_prev) * sqrt_dt * z2
        )
        v[:, t] = np.maximum(v[:, t], 0.0)

        r[:, t] = mu * dt + np.sqrt(v[:, t]) * sqrt_dt * z1

    return r


def central_moments(x: np.ndarray):
    """Return central moments m1..m4: mean, E[(x-μ)²], E[(x-μ)³], E[(x-μ)⁴]."""
    m1 = np.mean(x)
    c = x - m1
    m2 = np.mean(c**2)
    m3 = np.mean(c**3)
    m4 = np.mean(c**4)
    return m1, m2, m3, m4


# ------------------------
# 3. Compare actual vs simulated moments
# ------------------------
rows = {}

for asset in assets:
    # Actual returns
    df = pd.read_csv(files[asset])
    real = df["daily_return"].dropna().values.astype(float)

    # Simulate Heston
    p = params[asset]
    sim_paths = simulate_heston(T, **p, paths=N)
    sim = sim_paths.reshape(-1)   # flatten  (T * N samples)

    # Central moments
    real_m1, real_m2, real_m3, real_m4 = central_moments(real)
    sim_m1, sim_m2, sim_m3, sim_m4 = central_moments(sim)

    rows[asset] = {
        "Real Mean": real_m1,
        "Real M2 (Var)": real_m2,
        "Real M3": real_m3,
        "Real M4": real_m4,
        "Sim Mean": sim_m1,
        "Sim M2 (Var)": sim_m2,
        "Sim M3": sim_m3,
        "Sim M4": sim_m4,
    }

moment_df = pd.DataFrame.from_dict(rows, orient="index")
print(moment_df)
