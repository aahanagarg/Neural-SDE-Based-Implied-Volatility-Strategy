# Neural-SDE-Based-Implied-Volatility-Strategy

## Volatility via Neural SDEs: Strategy Overview

### What This Strategy Does
This strategy models and trades **implied volatility surfaces (IVS)** using **Neural Stochastic Differential Equations (Neural SDEs)** — a modern, data-driven way to model how volatility evolves over time.  

The goal is to build volatility surfaces that:
- **Fit observed option prices** accurately.  
- **Respect economic consistency** — no arbitrage, smooth, convex surfaces.  
- **Adapt dynamically** to changing market regimes.

Once trained, the model produces:
1. A **fair-value surface** for today's implied volatility.  
2. A **forecasted surface** for the next day.  

We then compare these to **actual market IV data** —  
if the market volatility is **too high**, we consider selling volatility (e.g., short calls/puts, verticals);  
if it’s **too low**, we consider buying volatility (e.g., straddles, long verticals).

### How We Get Implied Volatility and Surfaces
1. **Implied Volatility (IV)** is reverse-engineered from option prices using the **Black–Scholes formula** — it’s the volatility that makes the model price equal to the market price.  
2. For each underlying (like AAPL, SPX, or BTC), we collect option chains across **many strikes and maturities** using `yfinance` or QuantConnect.  
3. We compute **moneyness = strike / spot** and plot IV across both strike and expiry → forming the **Implied Volatility Surface (IVS)**.  
4. The Neural SDE is then trained to learn how this surface changes through time.

### Why Classic Models Fail
Traditional models assume **fixed mathematical forms** for volatility, which limit flexibility and adaptability:
- **Heston:** assumes volatility follows a mean-reverting stochastic process; captures some clustering but struggles with sudden regime shifts.  
- **SABR:** models both forward price and volatility as correlated processes; fits skews/smiles but breaks under sparse data or shocks.  
- **Stochastic Variational Inference (SVI):** a simple parametric curve that fits a single-day smile; easy to calibrate but unstable over time and not predictive.

These models:
- Depend on **manual calibration** each day.  
- Can **violate no-arbitrage conditions** when data is noisy.  
- **Don’t generalize well** when market regimes change (e.g., crisis vs. calm periods).

### Why Neural SDEs Are Better
Neural SDEs combine **deep learning flexibility** with **financial structure**:
- They **learn volatility dynamics directly from data**, not from a fixed equation.  
- Capture **nonlinear and regime-dependent behavior** naturally.  
- Allow **smooth, arbitrage-free surfaces** via regularization.  
- Produce both **today’s fair value** and **tomorrow’s forecast**, enabling trading signals.

In essence, the Neural SDE learns *how volatility moves*, not just *what it looks like*.  
It turns the volatility surface into a dynamic, data-driven process instead of a static curve fit.


#  Options Strategies


## 1. Long Straddle (ATM Call + ATM Put)

**Why this strategy is effective:**  
- Converts predicted volatility directly into payoff.
- Use when **predicted realized volatility (RV) > implied volatility (IV)**.
- Avoid when predicted RV < IV.

**Our model predicts:**  
- Expected realized volatility (RV)  
- Expected implied volatility (IV)  
- Volatility of volatility  
- Dispersion in options surface  

**References:**  
- Jones (2006) – *Option Returns and Volatility Mispricing*: Shows long straddles outperform when RV exceeds IV.  
- Driessen & Maenhout (2007) – *Volatility Timing Using Options*: ML-predicted volatility improves long straddle returns.  


## 2. Long Strangle (OTM Call + OTM Put)

**Characteristics:**  
- Cheaper than straddles  
- Lower theta decay  
- Higher convexity per dollar  
- Better for large moves (shocks)  
- Lower gamma → less rebalancing stress  

**Rationale:**  
- Neural ODE/SDE and volatility surface forecasting detect regime shifts.  
- Strangles monetize these shifts at lower cost.  

**Reference:**  
- Coval & Shumway (2001) – *The Profitability of Option Trading Strategies*: Long strangles capture jump risk and crash risk mispricing.  


## 3. Long Calendar Spread (Buy long-dated, Sell short-dated)

**When to use:**  
- Short-term IV is too low compared to long-term IV.  

**Our model computes:**  
- Term structure of IV  
- Term structure slope  
- Short-term vs long-term IV  
- Surface convexity  

**Reference:**  
- Xiang & Yan (2010) – *Volatility Term Structure Forecasting*: Predictable short-term vs long-term IV patterns.  
- Sirignano (2016) – *Forecasting Volatility with ML*: Short-horizon IV is highly predictable → ideal for calendar spreads.  


## 4. Put Backspread (Sell 1 ATM Put, Buy 2 OTM Puts)

**Purpose:**  
- Detects and profits from crash regimes and tail events.  

**Model features:**  
- Jump/volatility-of-vol indicators  
- ATM skew  
- Put skew curvature  
- Tail-risk forecasts  
- SDE dynamics predicting downward shocks  

**Benefits:**  
- High payoff in tail events  
- Benefit from rising volatility  
- Cheap to carry (small debit or slight credit)  
- Infinite downside convexity  

**Reference:**  
- Bollerslev & Todorov (2011) – *The Volatility Risk Premium and Crash Risk*: Monetizes tail volatility mispricing.  


## 5. Call Backspread (Sell 1 ATM Call, Buy 2 OTM Calls)

**Purpose:**  
- Profits from upside volatility and trend accelerations.  

**Model predicts:**  
- Volatility expansions  
- Positive skew changes  
- Trend acceleration  
- High-vol upside breakouts  

**Benefits:**  
- Positive vega  
- No directional bias needed  
- Explodes during upside volatility events (FOMC, CPI, short squeezes)  

**Reference:**  
- Xing, Zhang, Zhao (2010) – *Skewness Premium in Options Markets*: Predictable skew changes → monetizable via backspreads.  


## 6. Volatility Dispersion Strategy (Buy single-name vol, Sell index vol)

**When it works:**  
- Correlation collapses  
- Stock vol rises faster than index vol  
- Model predicts high cross-sectional vol  

**Model features:**  
- Cross-sectional volatility  
- Correlation risk  
- Surface curvature  
- Smile dispersion  

**Reference:**  
- Carr & Wu (2010) – *Volatility Dispersion Trading*: Exploits structural mispricing between single-name and index vol.  
