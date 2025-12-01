import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === Load data from five CSV files ===
# Each CSV must have a column named "daily_return"
files = ["BTC-AUD.csv", "BNB-AUD.csv", "ETH-AUD.csv", "XRP-AUD.csv", "SOL-AUD.csv"]

# Read only the "daily_return" column from each file
dfs = [pd.read_csv(f, usecols=["daily_return"]) for f in files]

# Combine into one DataFrame with proper column names
df = pd.concat(dfs, axis=1)
df.columns = [f"Asset_{i + 1}" for i in range(len(dfs))]

# === Annualize returns and covariance (using 365 days) ===
mean_returns = df.mean() * 365
cov_matrix = df.cov() * 365
risk_free_rate = 0.04  # Annual risk-free rate


# Portfolio performance function
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe_ratio


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(len(mean_returns)))
init_guess = np.array(len(mean_returns) * [1. / len(mean_returns)])

# Optimize for maximum Sharpe ratio portfolio
opt_result = minimize(neg_sharpe_ratio, init_guess,
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_result.x
ret, vol, sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix, risk_free_rate)

print("Optimal Weights (No Short Selling):", optimal_weights)
print("Annualized Expected Return:", ret)
print("Annualized Volatility:", vol)
print("Sharpe Ratio:", sharpe)

# === Efficient Frontier ===
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
frontier_volatility = []

for tr in target_returns:
    constraints_frontier = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - tr}
    )
    result = minimize(portfolio_volatility, init_guess,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints_frontier)
    frontier_volatility.append(result.fun)

# === Simulated Portfolios ===
n_portfolios = 100000
sim_returns, sim_vols, sim_sharpes = [], [], []

for _ in range(n_portfolios):
    weights = np.random.dirichlet(np.ones(len(mean_returns)))  # ensures weights sum to 1, all >=0
    r, v, s = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    sim_returns.append(r)
    sim_vols.append(v)
    sim_sharpes.append(s)

# === Plot Efficient Frontier, CAL, and Simulated Portfolios ===
plt.figure(figsize=(10, 6))
plt.scatter(sim_vols, sim_returns, c=sim_sharpes, cmap='viridis', alpha=0.4, label='Simulated Portfolios')
plt.plot(frontier_volatility, target_returns, 'b--', linewidth=2, label='Efficient Frontier')

x = np.linspace(0, max(frontier_volatility), 100)
cal = risk_free_rate + sharpe * x
plt.plot(x, cal, 'r-', linewidth=2, label='Capital Allocation Line')

plt.title('Mean-Variance Optimal Portfolio')
plt.xlabel('Annualized Volatility (Std. Dev.)')
plt.ylabel('Annualized Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(vol, ret, c='green', marker='*', s=200, label='Sharpe Optimal Portfolio')
plt.legend()
plt.grid(True)
plt.show()
