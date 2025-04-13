import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import time
import optuna
from scipy.interpolate import RegularGridInterpolator


# ------------------ Parameters ------------------
np.random.seed(42)
T = 1.0
X0 = 1e6
initial_wealth = 1e6
eta = 1e-4
gamma = 5e-5
r_f = 0.02  # risk-free rate
threshold = 0.5 
N = 100
dt = T / N
t = np.linspace(0, T, N + 1)
n_paths = 50
mu = 0.05
sigma = 0.2
t_mc = np.linspace(0, T, N + 1)
S0 = 150  # starting price


regime_threshold = 0.5
low_vol = 0.1
high_vol = 0.3


# ------------------ Fetch Market Data ------------------
def fetch_market_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].dropna().astype(float).values

price_path = fetch_market_data('AAPL', '2023-01-01', '2024-01-01')
N = len(price_path) - 1
dt = T / N
t = np.linspace(0, T, N + 1)

# ------------------ Cost and Wealth Functions ------------------
def compute_execution_cost(price, trading_rate, eta, gamma, dt):
    """
    Computes execution cost as the difference between ideal and actual proceeds.
    Adjusts for permanent and temporary impact.
    """
    raw_price = price[:-1]
    # Cumulative impact affects future prices (permanent), linear impact affects immediate price (temporary)
    adj_price = raw_price - eta * trading_rate - gamma * np.cumsum(trading_rate)

    # Execution proceeds
    proceeds_adj = trading_rate * adj_price * dt
    proceeds_raw = trading_rate * raw_price * dt

    # Cost = slippage = loss in proceeds due to price impact
    cost = np.sum(proceeds_raw - proceeds_adj)

    return cost

def track_wealth_with_allocation(price, trading_rate, eta, gamma, r_f, initial_wealth):
    wealth = [float(initial_wealth)]
    for i in range(len(trading_rate)):
        adjusted_price = price[i] - eta * trading_rate[i] - gamma * trading_rate[i]
        proceeds_risky = trading_rate[i] * adjusted_price
        proceeds_risk_free = wealth[-1] * r_f * dt
        new_wealth = wealth[-1] + proceeds_risky + proceeds_risk_free
        wealth.append(new_wealth.item() if isinstance(new_wealth, np.ndarray) else new_wealth)
    return np.array(wealth)

# ------------------ Strategy Simulation ------------------
def simulate_strategy(trading_schedule):
    trading_rate = -np.diff(trading_schedule)
    cost = compute_execution_cost(price_path, trading_rate, eta, gamma, dt)
    wealth = track_wealth_with_allocation(price_path, trading_rate, eta, gamma, r_f, initial_wealth)
    return trading_rate, cost, wealth

# ------------------ Naive Strategy ------------------
naive_inventory = np.linspace(X0, 0, N + 1)
naive_trading_rate, naive_cost, naive_wealth = simulate_strategy(naive_inventory)

# ------------------ Optimal Strategy ------------------
optimal_inventory = X0 * (1 - t) ** 2
optimal_trading_rate, optimal_cost, optimal_wealth = simulate_strategy(optimal_inventory)

# ------------------ HJB-style Strategy ------------------
from scipy.interpolate import RegularGridInterpolator

# Set safe, liquid market parameters
eta = 1e-7
gamma = 1e-7

def run_hjb(eta, gamma, trade_penalty_scale, terminal_penalty_scale):
    inventory_grid = np.asarray(np.linspace(0, X0, 10)).flatten()
    price_grid = np.asarray(np.linspace(min(price_path), max(price_path), 10)).flatten()

    V = np.zeros((N + 1, len(inventory_grid), len(price_grid)))
    policy = np.zeros((N, len(inventory_grid), len(price_grid)))

    for i_idx, x in enumerate(inventory_grid):
        for p_idx, p in enumerate(price_grid):
            V[-1, i_idx, p_idx] = -terminal_penalty_scale * p.item() * x.item() - 0.05 * x.item()**2

    for t_idx in range(N - 1, -1, -1):
        if t_idx % 10 == 0:
            print(f"Solving HJB for t = {t_idx}/{N}")

        V_interp = RegularGridInterpolator(
            (inventory_grid, price_grid),
            V[t_idx + 1].astype(float),
            bounds_error=False,
            fill_value=None
        )

        for i_idx, x in enumerate(inventory_grid):
            for p_idx, p in enumerate(price_grid):
                x = float(x)
                p = float(p)
                max_val = float('-inf')
                best_rate = 0
                max_allowed_rate = min(x / dt, 50000)

                for v in np.linspace(0, min(max_allowed_rate, 10000), 30):
                    x_new = max(0, x - v * dt)
                    x_new_clipped = np.clip(x_new, inventory_grid[0], inventory_grid[-1])
                    p_clipped = np.clip(p, price_grid[0], price_grid[-1])
                    future = V_interp([[x_new_clipped, p_clipped]])[0]

                    adj_price = max(p - eta * v - gamma * v, 0.01)
                    proceeds = v * adj_price
                    trade_penalty = trade_penalty_scale * v**2
                    inventory_penalty = 0.01 * x_new
                    value = proceeds * dt + future - trade_penalty - inventory_penalty

                    if value > max_val:
                        max_val = value
                        best_rate = v

                V[t_idx, i_idx, p_idx] = max_val
                policy[t_idx, i_idx, p_idx] = best_rate

    # Simulate the strategy
    hjb_inventory = [X0]
    hjb_trading_rates = []
    hjb_wealth = [initial_wealth]

    for t_idx in range(N):
        x = hjb_inventory[-1]
        p = price_path[t_idx]
        i_idx = np.argmin(np.abs(inventory_grid - x))
        p_idx = np.argmin(np.abs(price_grid - p))
        v = policy[t_idx, i_idx, p_idx]
        hjb_trading_rates.append(v)
        x_new = max(0, x - v * dt)
        adj_price = max(p - eta * v - gamma * v, 0.01)
        proceeds_risky = v * adj_price
        proceeds_risk_free = hjb_wealth[-1] * r_f * dt
        new_w = hjb_wealth[-1] + proceeds_risky + proceeds_risk_free
        hjb_inventory.append(x_new)
        hjb_wealth.append(float(new_w))

    # Return final results
    total_volume = np.sum(hjb_trading_rates) * dt
    return np.array(hjb_wealth), total_volume, hjb_trading_rates, hjb_inventory


#Optuna Objective Function

def objective(trial):
    eta = trial.suggest_float("eta", 1e-7, 1e-4, log=True)
    gamma = trial.suggest_float("gamma", 1e-7, 1e-4, log=True)
    trade_penalty = trial.suggest_float("trade_penalty", 1e-6, 1e-2, log=True)
    terminal_penalty = trial.suggest_float("terminal_penalty", 5.0, 20.0)

    hjb_wealth, volume, hjb_trading_rates, hjb_inventory = run_hjb(eta, gamma, trade_penalty, terminal_penalty)
    final_wealth = hjb_wealth[-1]
    
    # Guard against unrealistic wealth
    if final_wealth > 5e6 or final_wealth < 0.9e6:
        return 1e9  # discard extreme outcomes



    if volume < 0.3 * X0 or volume > 1.2 * X0:
        return 1e9  # No trade or overtraded

    returns = np.diff(hjb_wealth) / np.clip(hjb_wealth[:-1], 1e-6, None)
    if np.any(np.isnan(returns)) or np.all(returns == 0):
        return 1e9  # kill dead or broken runs

    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)


    # === Reference values from naive/optimal strategies ===
    naive_sharpe = 7.4
    optimal_sharpe = 4.7
    base_wealth = 1.02e6  # (final wealth of naive/optimal ~1.02M)

    # === Penalize if final inventory is too large ===
    if hjb_inventory[-1] > 0.05 * X0:
        return 1e9  # Didn't liquidate enough (kept >5% of inventory)

    # === Early stopping if strong config is found ===
    if sharpe > naive_sharpe and final_wealth > base_wealth + 10000:
        print(" Found strong HJB config. Stopping early.")
        raise optuna.exceptions.TrialPruned()


    # === Combined scoring to guide tuning ===
    score = -sharpe - 0.01 * (final_wealth / 1e6)  # Sharpe + wealth, both in similar scale

    print(f"Trial result -> Sharpe: {sharpe:.2f}, Final Wealth: ${final_wealth:,.2f}, Remaining Inventory: {hjb_inventory[-1]:,.0f}")

    return score

# ------------------ Step 3: Run the Optimization Loop ------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")

    try:
        study.optimize(objective, n_trials=1)
    except optuna.exceptions.TrialPruned:
        print(" Early exit — good HJB config found.")

    print("\n Best Parameters Found:")
    if study.best_trial:
        print("\n Best Parameters Found:")
    for k, v in study.best_params.items():
        print(f"{k}: {v:.6g}")
    else:
        print("\n No valid trials completed.")

        print(f"{k}: {v:.6g}")

    # === Step 4: Run HJB with Best Parameters ===
params = study.best_params
eta = params["eta"]
gamma = params["gamma"]
trade_penalty = params["trade_penalty"]
terminal_penalty = params["terminal_penalty"]

# Run HJB strategy with optimal parameters
hjb_wealth, volume, hjb_trading_rates, hjb_inventory = run_hjb(
    eta, gamma, trade_penalty, terminal_penalty
)
final_wealth = hjb_wealth[-1]



# Print results
print(f"\nFinal HJB Wealth: ${final_wealth:,.2f}")
print(f"Total Volume Traded: {volume:,.2f} shares")
print("HJB Trading Rates (first 10):", hjb_trading_rates[:10])
print("Final HJB Inventory:", hjb_inventory[-1])
print("Non-zero HJB trades:", any(v > 1e-3 for v in hjb_trading_rates))
print("Final HJB Wealth:", hjb_wealth[-1])

# Diagnostics
per_share_impact = eta * np.array(hjb_trading_rates) + gamma * np.cumsum(hjb_trading_rates)
print(f"Max per-share impact: {np.max(per_share_impact):,.2f}")
print(f"Total trading volume: {np.sum(np.array(hjb_trading_rates) * dt):,.0f} shares")
print(f"Average adjusted price: {np.mean(price_path[:-1] - per_share_impact):.2f}")


# Run HJB simulation first

hjb_cost = compute_execution_cost(price_path, np.array(hjb_trading_rates), eta, gamma, dt)
print("HJB Trading Rates (first 10):", hjb_trading_rates[:10])
print("Final HJB Inventory:", hjb_inventory[-1])
print("Non-zero HJB trades:", any(v > 1e-3 for v in hjb_trading_rates))
print("Final HJB Wealth:", hjb_wealth[-1])

# Diagnostics go *here*, AFTER simulation
per_share_impact = eta * np.array(hjb_trading_rates) + gamma * np.cumsum(hjb_trading_rates)
print(f"Max per-share impact: {np.max(per_share_impact):,.2f}")
print(f"Total trading volume: {np.sum(np.array(hjb_trading_rates) * dt):,.0f} shares")
print(f"Average adjusted price: {np.mean(price_path[:-1] - per_share_impact):.2f}")

# ------------------ Metrics ------------------
def sharpe_ratio(wealth):
    returns = np.diff(wealth) / wealth[:-1]
    return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

def max_drawdown(wealth):
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    return np.min(drawdown)

hjb_cost = compute_execution_cost(price_path, np.array(hjb_trading_rates), eta, gamma, dt)
print("HJB Trading Rates (first 10):", hjb_trading_rates[:10])
print("Final HJB Inventory:", hjb_inventory[-1])
print("Non-zero HJB trades:", any(v > 1e-3 for v in hjb_trading_rates))
print("Final HJB Wealth:", hjb_wealth[-1])

hjb_sharpe = sharpe_ratio(hjb_wealth)
hjb_dd = max_drawdown(hjb_wealth)

# ------------------ Output ------------------
print(f"Naive Execution Cost (Permanent Impact): ${naive_cost:,.2f}")
print(f"Optimal Execution Cost (Permanent Impact): ${optimal_cost:,.2f}")
print(f"HJB Execution Cost (Permanent Impact): ${hjb_cost:,.2f}")

print(f"Naive Sharpe Ratio: {sharpe_ratio(naive_wealth):.2f}")
print(f"Optimal Sharpe Ratio: {sharpe_ratio(optimal_wealth):.2f}")
print(f"HJB Sharpe Ratio: {hjb_sharpe:.2f}")

print(f"Naive Max Drawdown: {max_drawdown(naive_wealth):.2%}")
print(f"Optimal Max Drawdown: {max_drawdown(optimal_wealth):.2%}")
print(f"HJB Max Drawdown: {hjb_dd:.2%}")

# ------------------ Visualization ------------------
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, naive_wealth, label="Naive")
plt.plot(t, optimal_wealth, label="Optimal")
plt.plot(t, hjb_wealth, label="HJB")
plt.title("Wealth Over Time")
plt.xlabel("Time")
plt.ylabel("Wealth")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t[:-1], naive_trading_rate, label="Naive")
plt.plot(t[:-1], optimal_trading_rate, label="Optimal")
plt.plot(t[:-1], hjb_trading_rates, label="HJB")
plt.title("Trading Rate Over Time")
plt.xlabel("Time")
plt.ylabel("Shares Sold")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, price_path)
plt.title("AAPL Price Path (2023)")
plt.xlabel("Time")
plt.ylabel("Price")

plt.subplot(2, 2, 4)
plt.plot(t, hjb_inventory, label="HJB Inventory")
plt.title("HJB Inventory Over Time")
plt.xlabel("Time")
plt.ylabel("Inventory")
plt.legend()

plt.tight_layout()
plt.show(block=False)


plt.plot(t, hjb_inventory)
plt.title("HJB Inventory Check")
plt.show(block=False)
print("HJB: Total Shares Sold:", X0 - hjb_inventory[-1])
print("HJB: Final Inventory:", hjb_inventory[-1])
print("HJB: Total Proceeds:", hjb_wealth[-1] - initial_wealth)

##########################

# ------------------ GBM Path Simulation ------------------
def simulate_gbm_paths(S0, mu, sigma, T, N, n_paths):
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    for i in range(1, N + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return np.linspace(0, T, N + 1), paths

# ------------------ HJB Strategy on Simulated Paths ------------------
def simulate_hjb_paths(price_paths):
    all_wealths = []
    results = []

    for sim_price_path in price_paths:
        inventory_grid = np.linspace(0, X0, 10)
        price_grid = np.linspace(min(sim_price_path), max(sim_price_path), 10)
        V = np.zeros((N + 1, len(inventory_grid), len(price_grid)))
        policy = np.zeros((N, len(inventory_grid), len(price_grid)))
# This will be reused in each timestep to interpolate future values
# You could put this inside the time loop if your value function changes shape over time

        # Terminal condition
        V[-1] = -1000 * inventory_grid[:, None]

        for t_idx in range(N - 1, -1, -1):
            for i_idx, x in enumerate(inventory_grid):
                for p_idx, p in enumerate(price_grid):
                    horizon = (N - t_idx) * dt
                    max_rate = min(x / horizon if horizon > 0 else x, 10000)
                    best_val, best_rate = -np.inf, 0
                    for v in np.linspace(0, max_rate, 5):
                        x_new = max(0, x - v * dt)
                        V_interp = RegularGridInterpolator(
                            (inventory_grid, price_grid),
                            V[t_idx + 1],
                            bounds_error=False,
                            fill_value=None
                            )
                        for v in np.linspace(0, max_rate, 5):
                            x_new = max(0, x - v * dt)
                            x_new_clipped = np.clip(x_new, inventory_grid[0], inventory_grid[-1])
                            p_clipped = np.clip(p, price_grid[0], price_grid[-1])

                            future = V_interp([[x_new_clipped, p_clipped]])[0]

                            c = v * p - 1e-5 * v ** 2
                            value = -c * dt + future - 1000 * x_new

                        if value > best_val:
                            best_val, best_rate = value, v
                    V[t_idx, i_idx, p_idx] = best_val
                    policy[t_idx, i_idx, p_idx] = best_rate

        # Simulate using optimal policy
        inventory = [X0]
        wealth = [initial_wealth]
        hjb_trading_rates = []

        for t_idx in range(N):
            x = inventory[-1]
            p = sim_price_path[t_idx]
            i_idx = np.argmin(np.abs(inventory_grid - x))
            p_idx = np.argmin(np.abs(price_grid - p))
            v = policy[t_idx, i_idx, p_idx]
            hjb_trading_rates.append(v)
            x_new = max(0, x - v * dt)
            inventory.append(x_new)
            adj_price = p - eta * v - gamma * v
            proceeds = v * adj_price
            risk_free = wealth[-1] * r_f * dt
            wealth.append(wealth[-1] + proceeds + risk_free)

        all_wealths.append(wealth)
        returns = np.diff(wealth) / wealth[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        drawdown = np.min((wealth - np.maximum.accumulate(wealth)) / np.maximum.accumulate(wealth))
        cost = compute_execution_cost(sim_price_path, np.array(hjb_trading_rates), eta, gamma, dt)
        results.append({"final_wealth": wealth[-1], "sharpe": sharpe, "drawdown": drawdown, "cost": cost})

    return np.array(all_wealths), results

# ------------------ Run Simulation ------------------
print("\nStarting Monte Carlo Simulation of HJB under GBM Paths...")
start = time.time()

t_mc, gbm_paths = simulate_gbm_paths(S0, mu, sigma, T, N, n_paths)
hjb_wealth_paths, mc_results = simulate_hjb_paths(gbm_paths)

finals = [r["final_wealth"] for r in mc_results]
print(f"\n Monte Carlo Summary for {n_paths} GBM Paths:")
print(f"  Mean Final Wealth: ${np.mean(finals):,.2f}")
print(f"  Std Dev Final Wealth: ${np.std(finals):,.2f}")
print(f"  Max Final Wealth: ${np.max(finals):,.2f}")
print(f"  Min Final Wealth: ${np.min(finals):,.2f}")
print(f"  Sharpe (Mean): {np.mean([r['sharpe'] for r in mc_results]):.2f}")
print(f"  Max Drawdown (Mean): {np.mean([r['drawdown'] for r in mc_results]):.2%}")
print(f" Done in {time.time() - start:.2f} seconds")

# ------------------ Plot Results ------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(finals, bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Final Wealth (HJB Strategy)")
plt.xlabel("Final Wealth")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
for i in range(min(25, len(gbm_paths))):
    plt.plot(t_mc, gbm_paths[i])
plt.title("25 Sample GBM Price Paths")
plt.xlabel("Time")
plt.ylabel("Simulated Price")

plt.tight_layout()
plt.show(block=False)


# ------------------ Regime-Switching GBM Path Simulation ------------------
def simulate_regime_switching_paths(S0, mu, T, N, n_paths, low_vol, high_vol, threshold):
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    for i in range(1, N + 1):
        Z = np.random.standard_normal(n_paths)
        regime = (np.random.rand(n_paths) > threshold).astype(float)
        sigma = regime * high_vol + (1 - regime) * low_vol
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        paths[:, i] = paths[:, i - 1] * np.exp(drift + diffusion)
    return np.linspace(0, T, N + 1), paths

# ------------------ Strategy-Based Wealth Simulation ------------------
def simulate_wealth_paths(price_paths, strategy_name):
    wealth_paths = []
    for path in price_paths:
        inventory = X0
        wealth = [initial_wealth]
        for i in range(N):
            if strategy_name == "naive":
                v = inventory / (N - i)
            elif strategy_name == "optimal":
                v = inventory * (1 - t[i]) / (T - t[i] + 1e-6)
            elif strategy_name == "hjb_like":
                v = min(inventory, 10000)
            else:
                v = 0
            inventory -= v * dt
            adj_price = path[i] - eta * v - gamma * v
            proceeds = v * adj_price
            risk_free = wealth[-1] * r_f * dt
            wealth.append(wealth[-1] + proceeds + risk_free)
        wealth_paths.append(wealth)
    return np.array(wealth_paths)

# ------------------ Compute Metrics ------------------
def compute_metrics(wealth_paths):
    final_wealth = wealth_paths[:, -1]
    sharpe_ratios = []
    drawdowns = []
    for w in wealth_paths:
        ret = np.diff(w) / w[:-1]
        sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252) if np.std(ret) > 0 else 0
        peak = np.maximum.accumulate(w)
        dd = np.min((w - peak) / peak)
        sharpe_ratios.append(sharpe)
        drawdowns.append(dd)
    return final_wealth, np.mean(sharpe_ratios), np.mean(drawdowns)

# ------------------ Run Regime-Switching Simulation ------------------
low_vol = 0.1
high_vol = 0.3
regime_threshold = 0.5
t_mc, regime_paths = simulate_regime_switching_paths(S0, mu, T, N, n_paths, low_vol, high_vol, regime_threshold)

naive_wealth = simulate_wealth_paths(regime_paths, "naive")
optimal_wealth = simulate_wealth_paths(regime_paths, "optimal")
hjb_wealth = simulate_wealth_paths(regime_paths, "hjb_like")

naive_final, naive_sharpe, naive_dd = compute_metrics(naive_wealth)
optimal_final, optimal_sharpe, optimal_dd = compute_metrics(optimal_wealth)
hjb_final, hjb_sharpe, hjb_dd = compute_metrics(hjb_wealth)

# ------------------ Summary Table ------------------
summary = pd.DataFrame({
    "Strategy": ["Naive", "Optimal", "HJB-like"],
    "Mean Final Wealth": [np.mean(naive_final), np.mean(optimal_final), np.mean(hjb_final)],
    "Sharpe Ratio": [naive_sharpe, optimal_sharpe, hjb_sharpe],
    "Max Drawdown": [naive_dd, optimal_dd, hjb_dd]
})

print("\nSummary Table (Regime Switching):")
print(summary)

# ------------------ Confidence Band Plot ------------------
def plot_confidence_bands(t, wealth_paths, label, color):
    mean_wealth = np.mean(wealth_paths, axis=0)
    p5 = np.percentile(wealth_paths, 5, axis=0)
    p95 = np.percentile(wealth_paths, 95, axis=0)
    plt.plot(t, mean_wealth, label=f"{label} (Mean)", color=color)
    plt.fill_between(t, p5, p95, alpha=0.2, color=color, label=f"{label} (P5-P95)")

plt.figure(figsize=(12, 6))
plot_confidence_bands(t_mc, naive_wealth, "Naive", "blue")
plot_confidence_bands(t_mc, optimal_wealth, "Optimal", "green")
plot_confidence_bands(t_mc, hjb_wealth, "HJB-like", "red")
plt.title("Wealth Trajectories (Regime-Switching GBM)")
plt.xlabel("Time")
plt.ylabel("Wealth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

