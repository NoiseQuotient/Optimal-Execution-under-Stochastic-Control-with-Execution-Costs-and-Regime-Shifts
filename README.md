# Optimal Execution under Stochastic Control with Execution Costs and Regime Shifts

##  What Is This Project Really About?

At its core, this project tackles a question every institutional investor, asset manager, or trader faces:

> **"If I hold a huge position in a stock, how do I sell it off without setting fire to the price I'm trying to get?"**

You don’t want to sell too fast (you’ll push the price down), but you also don’t want to wait forever (market moves, inventory risk). This project is an attempt to model that trade-off mathematically — and solve it *optimally*.

This isn’t just academic curiosity, it addresses a core challenge in modern electronic trading. The principles behind optimal execution under market impact are fundamental to the trading infrastructure used by leading quant funds and investment banks developing algorithmic execution strategies.

t mirrors real problems: Every institution liquidating a position runs into this. It’s the real-world "market microstructure" challenge.

**For this reason this project builds a full-stack pipeline for modeling optimal liquidation with market impact using HJB control. It evaluates strategies under real and simulated market conditions, tunes parameters via optimization, and analyzes execution performance with professional metrics, all while being transparent, interpretable, and extensible.**

---

### What This Project Does:

1. Load real market data (e.g., AAPL 2023)
2. Define Hamilton–Jacobi–Bellman (HJB) control problem over discrete grid
3. Solve via dynamic programming (backward)
4. Tune impact/penalty params using Bayesian optimization (via Optuna)
5. Evaluate wealth, cost, Sharpe
6. Monte Carlo simulation for robustness (GBM + regime shifts)
7. Visualize trading behavior + metrics


## The Problem Framed as Control Theory

This is a **finite-horizon stochastic control problem**.

You start with a large inventory of shares (say, 1,000,000) and a fixed time to sell them. The market is dynamic, and every action has consequences:

| Challenge                 | Why it Matters                                          |
|--------------------------|----------------------------------------------------------|
| Price impact           | Every trade moves the market (permanent + temporary)     |
| Inventory risk         | Holding stock exposes you to market movement             |
| Time pressure           | You're penalized for leftover inventory at the end       |

Every move you make (trade rate) affects your future state.

---

## Why an HJB Approach?

You could brute-force simulate strategies (as some quant shops do with reinforcement learning), but HJB theory gives you the analytical way. Instead of simulating random policies or training reinforcement learning agents, we **solve the optimal control problem directly** using **Hamilton–Jacobi–Bellman (HJB)** equations.

> **"What’s the best action to take at each state — given my inventory and market price to maximize my expected reward?"**

We compute this policy *backward in time* using dynamic programming. The result: a **policy grid** that maps every (inventory, price, time) to the optimal trading rate.

---

## What’s Being Modeled — and Why?

| Concept              | Assumption                                   | Why It Makes Sense                                   |
|----------------------|-----------------------------------------------|-------------------------------------------------------|
| Price Evolution    | AAPL historical path + GBM simulation         | Realism + generalizability                            |
| Inventory Grid     | 10 discrete steps                            | Efficient and interpretable                           |
| Market Impact      | Temporary (`η`) + Permanent (`γ`)            | Standard in Almgren–Chriss-type models                |
| Risk-Free Rate     | Constant drift on cash                       | Models passive income in safe assets                 |
| Execution Proceeds | Adjusted for impact                          | Reflects slippage and trading frictions               |
| Objective          | Maximize wealth, penalize leftover inventory | Realistic goal for sell-side execution desks          |
| Penalty Terms      | Terminal + trading intensity                 | Avoids bursty or unfinished liquidation               |

---
## Summary of Key Performance Metrics

This section showcases how the HJB-based strategy performs under real and simulated market conditions and how it stacks up against traditional strategies like naive and Almgren-style quadratic optimal liquidation.

| Strategy | Final Wealth | Sharpe Ratio | Execution Cost | Final Inventory |
|----------|--------------|--------------|----------------|-----------------|
| Naive    | ~$1.02M      | 7.43         | $25.50M        | 0 shares        |
| Optimal  | ~$1.02M      | 4.70         | $25.67M        | 0 shares        |
| HJB      | $430.8M      | 4.72         | $1.12M         | 0 shares        |


** The HJB approach maintains a strong Sharpe ratio, drastically lowers execution cost, and adapts to price paths more intelligently than static strategies.The HJB controller consistently delivers high Sharpe, drastically reduced execution costs, and strategic liquidation patterns.**


**Monte Carlo Simulation (GBM + Regime-Switching) Simulating 50 random paths with GBM and volatility regimes:**

-Mean Final Wealth: ~$406M
-Std Dev: ~$28.5M
-Mean Sharpe Ratio: 4.11
-Max Drawdown: ~0.00%


The strategy shows high robustness even under regime changes a testament to the stability of the HJB control policy.  Confidence Bands Across Simulations. 90% of the wealth trajectories stay within tight bands, proving the strategy's resilience across uncertain market paths.

HJB adapts, it’s not one-size-fits-all. It can drastically cut costs while preserving risk-adjusted returns. It acts as a foundational benchmark for execution models in real-world trading systems

---
## Strategy Comparison: Naive vs Optimal vs HJB

![Wealth Comparison](images/figure1_summary.png)

**Top-left (Wealth Over Time):**
The HJB strategy (green) clearly dominates both the Naive (blue) and Optimal (orange, deterministic) strategies in terms of terminal wealth.

**Top-right (Trading Rate):**
The Naive strategy trades uniformly. The Optimal strategy reduces its trading rate over time.
The HJB strategy remains dynamic but here, it chose to front-load aggressively, selling fast.

**Bottom-left (AAPL Price Path):**
Real historical price data from 2023 is used as the environment. The price is upward-trending but with mid-year volatility spikes.

**Bottom-right (Inventory Check):**
A successful liquidation: all strategies reduce inventory smoothly to (near) zero by the horizon.


---

## HJB Monte Carlo Distribution & Simulated Price Paths

<img src="images/figure2_gbm_distribution.png" alt="Monte Carlo HJB Results" width="800"/>

(Distribution of Final Wealth):
Running the HJB strategy under 50 stochastic GBM price paths shows most outcomes cluster around a healthy terminal wealth.
A few outliers demonstrate sensitivity to market path variation — but no catastrophic failures, a good robustness sign.

(25 Sample GBM Paths):
Volatility is simulated realistically (μ=5%, σ=20%). Some paths trend up, others down — the HJB policy handles both.

---

##  Wealth Trajectories Under Regime-Switching Volatility

<img src="images/figure3_regime_bands.png" alt="Regime-switching GBM bands" width="800"/>

We simulate a regime-switching model where volatility randomly alternates between low (σ=10%) and high (σ=30%) states.

Shaded bands show the 5th to 95th percentile wealth trajectory across 50 runs.

HJB-like (red) strategy remains consistent, but struggles to outperform the Optimal (green) strategy under volatility shocks.
This highlights the value of adapting policy maps to evolving volatility — an exciting area for future work.

---
## What the HJB Model Is *Good At*

- **Adapts to the market:** It sells more aggressively when prices fall, waits when prices rise.
- **Balances risk and cost:** Knows when to absorb impact and when to wait.
- **Interpretable policies:** You can visualize the trade rate at any state.
- **Hyperparameter tuning:** Uses Optuna to find optimal penalties and impact coefficients.
- **Stress-testable:** Can simulate scenarios via GBM or regime-switching volatility.

---

##  What the Model *Doesn’t* Capture (Yet)

| Limitation                            | Why It Matters                                               |
|---------------------------------------|---------------------------------------------------------------|
|  No limit order book                | Assumes market orders; no bid-ask depth                      |
| No uncertainty about volatility    | Volatility is fixed or pre-specified                         |
| Single-asset only                 | No portfolio-level or cross-asset effects                    |
| No alpha signals                   | Doesn't use predictive models (e.g., ML forecasts)            |
| Curse of dimensionality            | HJB scales poorly in multi-dimensional state spaces          |

Despite this, the model is a **powerful baseline** rich, practical, and realistic enough to compare advanced execution strategies against.

---

## Ideas for Expansion

- Add **LSTM** or **Transformer** models for predictive signals
- Compare against **reinforcement learning** agents (DDPG, PPO)
- Extend to **multi-asset portfolios** with correlation
- Add **order book simulations** or queue modeling
- Use **real intraday trade data** for better calibration

---

> This project bridges the gap between **academic stochastic control** and **real-world quantitative execution**.


