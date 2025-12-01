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
mean_returns = df.mean() * 365
cov_matrix = df.cov() * 365

# Annualize returns (using 365 days)
returns_matrix = df.values * 365

# Risk aversion parameter (gamma)
gamma = 2


# Power utility function
def power_utility(wealth, gamma):
    if gamma == 1:
        return np.log(wealth)
    else:
        return (wealth ** (1 - gamma)) / (1 - gamma)


# Simulation of portfolios
n_portfolios = 100000
sim_returns, sim_vols, sim_utils, sim_weights = [], [], [], []

for _ in range(n_portfolios):
    # Generate random weights (no short selling, sum to 1)
    weights = np.random.dirichlet(np.ones(len(df.columns)))

    # Portfolio returns across all days
    portfolio_returns = returns_matrix @ weights
    wealth = 1 + portfolio_returns

    # Expected utility
    utility = np.mean(power_utility(wealth, gamma))

    # Store results
    sim_returns.append(np.mean(portfolio_returns))
    sim_vols.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sim_utils.append(utility)
    sim_weights.append(weights)

# Find portfolio with maximum expected utility
best_idx = np.argmax(sim_utils)
optimal_weights = sim_weights[best_idx]
best_return = sim_returns[best_idx]
best_vol = sim_vols[best_idx]
best_utility = sim_utils[best_idx]

print("Optimal Weights (Power Utility, Simulation):", optimal_weights)
print("Annualized Expected Return:", best_return)
print("Annualized Volatility:", best_vol)
print("Expected Utility:", best_utility)

# === Scatter Plot of Portfolios ===
plt.figure(figsize=(10, 6))
plt.scatter(sim_vols, sim_returns, c=sim_utils, cmap='plasma', vmin=-1, vmax=0,
            alpha=0.5, label='Simulated Portfolios')
plt.title(f'Simulated Portfolios under Power Utility (γ={gamma})')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Expected Return')
plt.colorbar(label='Expected Utility')
plt.scatter(best_vol, best_return, c='green', marker='*', s=200, label='Max utility portfolio')
plt.legend()
plt.grid(True)
plt.show()
