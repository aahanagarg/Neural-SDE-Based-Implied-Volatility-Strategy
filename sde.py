# region imports
from AlgorithmImports import *
# endregion

from AlgorithmImports import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import json

try:
    from QuantConnect.Securities import OptionRight
except ImportError:
    from QuantConnect import OptionRight

RISK_FREE = 0.03
DIVIDEND_YIELD = 0.0
NEWTON_TOL = 1e-6
NEWTON_MAX_ITERS = 50
SIGMA_BRACKET = (0.01, 2.0)
NEWTON_DAMP = 0.75

# --- Black–Scholes + IV (European) ----------------------------------


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, r, q, sigma, otype="call"):
    """
    Black–Scholes price for European calls/puts.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if otype == "call":
        return S * math.exp(-q * T) * _phi(d1) - K * math.exp(-r * T) * _phi(d2)
    else:
        return K * math.exp(-r * T) * _phi(-d2) - S * math.exp(-q * T) * _phi(-d1)


def bs_vega(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    nd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return S * math.exp(-q * T) * nd1 * sqrtT


def implied_vol(
    S,
    K,
    T,
    r,
    q,
    market_price,
    otype="call",
    tol=NEWTON_TOL,
    max_iter=NEWTON_MAX_ITERS,
    bracket=SIGMA_BRACKET,
    damp=NEWTON_DAMP,
):
    """
    Black–Scholes implied vol via damped Newton + bisection fallback.
    Treats options as European.
    """
    if (
        not np.isfinite(market_price)
        or market_price <= 0
        or T <= 0
        or S <= 0
        or K <= 0
    ):
        return float("nan")

    lo, hi = bracket
    # heuristic initial guess
    sigma = 0.25 + 0.25 * abs(math.log(K / S))
    sigma = float(np.clip(sigma, lo, hi))

    # Newton iterations
    for _ in range(max_iter):
        theo = bs_price(S, K, T, r, q, sigma, otype)
        if not np.isfinite(theo):
            break
        diff = theo - market_price
        if abs(diff) < tol:
            return sigma
        vega = bs_vega(S, K, T, r, q, sigma)
        if vega < 1e-8:
            break
        step = damp * diff / vega
        sigma_new = sigma - step
        if sigma_new <= lo or sigma_new >= hi:
            break
        sigma = sigma_new

    # Bisection as fallback
    f_lo = bs_price(S, K, T, r, q, lo, otype) - market_price
    f_hi = bs_price(S, K, T, r, q, hi, otype) - market_price
    expand = 0
    while f_lo * f_hi > 0 and expand < 6:
        hi *= 1.5
        f_hi = bs_price(S, K, T, r, q, hi, otype) - market_price
        expand += 1
    if f_lo * f_hi > 0:
        return float("nan")

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, T, r, q, mid, otype) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


# (Optional) keep CRR tree around if you ever want American-style checks
def crr_tree_price(S, K, T, r, q, sigma, steps=200, otype="call"):
    """
    Cox–Ross–Rubinstein binomial tree.
    Not used for IV anymore, but kept if you want to experiment.
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if otype == "call" else max(K - S, 0.0)
        return intrinsic

    if sigma <= 0 or S <= 0 or K <= 0 or steps <= 0:
        return float("nan")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = max(0.0, min(1.0, p))

    ST = np.array([S * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])

    if otype == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    for step in range(steps - 1, -1, -1):
        ST = ST[:-1] * u
        cont = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if otype == "call":
            exer = np.maximum(ST - K, 0.0)
        else:
            exer = np.maximum(K - ST, 0.0)
        values = np.maximum(cont, exer)

    return float(values[0])


# --- Neural SDE core (inference only) -------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class NeuralSDE(nn.Module):
    # dX_t = mu(t,X_t)dt + sigma(t,X_t)dW_t, S_t = exp(X_t)
    def __init__(self, hidden_dim=32, T_max=0.5, x_center=0.0):
        super().__init__()
        self.T_max = T_max
        self.x_center = x_center
        self.drift_net = MLP(2, 1, hidden_dim)
        self.diff_net = MLP(2, 1, hidden_dim)

    def _make_inp(self, t, x):
        if not torch.is_tensor(t):
            t = torch.full_like(x, float(t))
        elif t.ndim == 0:
            t = t.expand_as(x)
        t_scaled = t / max(self.T_max, 1e-6)
        x_scaled = x - self.x_center
        return torch.cat([t_scaled, x_scaled], dim=-1)

    def f(self, t, x):
        return self.drift_net(self._make_inp(t, x))

    def g(self, t, x):
        return F.softplus(self.diff_net(self._make_inp(t, x))) + 1e-4


def simulate_terminal_prices(model, S0, T, n_paths=256, n_steps=30, device="cpu"):
    model.to(device)
    model.train(False)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    x = torch.full((n_paths, 1), math.log(S0), device=device)
    for k in range(n_steps):
        t_k = (k + 1) * dt
        drift = model.f(t_k, x)
        diff = model.g(t_k, x)
        z = torch.randn_like(x)
        x = x + drift * dt + diff * sqrt_dt * z
    return torch.exp(x)


def mc_price_option_neural_sde(
    model,
    S0,
    K,
    T,
    r,
    q=0.0,
    n_paths=256,
    n_steps=30,
    device="cpu",
    right: OptionRight = OptionRight.Call,
):
    """
    Monte Carlo price of a European option using the Neural SDE.
    right = OptionRight.Call or OptionRight.Put
    """
    S_T = simulate_terminal_prices(
        model=model, S0=S0, T=T,
        n_paths=n_paths, n_steps=n_steps, device=device,
    )

    if right == OptionRight.Call:
        payoff = torch.clamp(S_T - K, min=0.0)
    else:
        payoff = torch.clamp(K - S_T, min=0.0)

    return math.exp(-r * T) * payoff.mean()



def forecast_option_price_neural_sde(
    model,
    S0,
    K,
    days_to_maturity,
    r=RISK_FREE,
    q=DIVIDEND_YIELD,
    n_paths=1024,
    n_steps=40,
    device="cpu",
    right: OptionRight = OptionRight.Call,
):
    """
    Forecast Neural-SDE price for a European option (call or put).
    """
    T = max(days_to_maturity, 0) / 365.0
    if T <= 0:
        # intrinsic value at expiry
        if right == OptionRight.Call:
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    with torch.no_grad():
        price_tensor = mc_price_option_neural_sde(
            model=model,
            S0=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            n_paths=n_paths,
            n_steps=n_steps,
            device=device,
            right=right,
        )
    return float(price_tensor.item())



# --- Wrapper: one NeuralSDE per ticker -------------------------------


class NeuralSDEModel:
    """
    Holds one NeuralSDE per ticker, loads **rolling snapshot weights**
    from ObjectStore based on the current backtest date, and can produce
    model-implied IVs via Black–Scholes.

    Expected ObjectStore keys (per ticker, e.g. SPY):
      - "neural_sde_train_dates_SPY.json"  -> ["2024-06-03", "2024-07-01", ...]
      - "neural_sde_SPY_20240603.pt"
      - "neural_sde_SPY_20240701.pt"
      - ...
    """

    def __init__(self, algo: QCAlgorithm):
        self.algo = algo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # One model per ticker + metadata about which snapshot is loaded
        self.models: dict[str, NeuralSDE | None] = {}
        self.train_dates: dict[str, list[datetime.date]] = {}
        self.current_model_date: dict[str, datetime.date | None] = {}

        for ticker in algo.UNIVERSE:
            json_key = f"neural_sde_train_dates_{ticker}.json"
            if not self.algo.ObjectStore.ContainsKey(json_key):
                self.algo.Debug(
                    f"[NeuralSDE] No train-date JSON in ObjectStore for {ticker} "
                    f"({json_key}) – model IV disabled for this ticker."
                )
                self.models[ticker] = None
                self.train_dates[ticker] = []
                self.current_model_date[ticker] = None
                continue

            try:
                bytes_data = self.algo.ObjectStore.ReadBytes(json_key)
                dates_str = bytes_data.decode("utf-8")
                raw_dates = json.loads(dates_str)  # list of "YYYY-MM-DD"
                dates = sorted(
                    datetime.strptime(d, "%Y-%m-%d").date() for d in raw_dates
                )
                self.train_dates[ticker] = dates
                self.models[ticker] = None
                self.current_model_date[ticker] = None
                self.algo.Debug(
                    f"[NeuralSDE] Loaded {len(dates)} snapshot dates for {ticker}"
                )
            except Exception as e:
                self.algo.Debug(
                    f"[NeuralSDE] Failed to parse train-date JSON for {ticker}: {e}"
                )
                self.models[ticker] = None
                self.train_dates[ticker] = []
                self.current_model_date[ticker] = None

    # ---- internal helpers -----------------------------------------

    def _load_snapshot(self, ticker: str, snapshot_date: datetime.date) -> bool:
        """
        Load a specific snapshot into memory from ObjectStore.
        """
        key = f"neural_sde_{ticker}_{snapshot_date.strftime('%Y%m%d')}.pt"
        if not self.algo.ObjectStore.ContainsKey(key):
            self.algo.Debug(
                f"[NeuralSDE] Snapshot missing for {ticker} on {snapshot_date}: {key}"
            )
            return False

        try:
            bytes_data = self.algo.ObjectStore.ReadBytes(key)
            buf = io.BytesIO(bytes_data)
            state = torch.load(buf, map_location=self.device)

            T_max = float(state.get("T_max", 0.5))
            x_center = float(state.get("x_center", 0.0))

            model = NeuralSDE(
                hidden_dim=32, T_max=T_max, x_center=x_center
            ).to(self.device)
            model.load_state_dict(state["state_dict"])
            model.eval()

            self.models[ticker] = model
            self.current_model_date[ticker] = snapshot_date
            self.algo.Debug(
                f"[NeuralSDE] Loaded snapshot for {ticker} @ {snapshot_date} from {key}"
            )
            return True

        except Exception as e:
            self.algo.Debug(
                f"[NeuralSDE] Failed to load snapshot {key} for {ticker}: {e}"
            )
            self.models[ticker] = None
            self.current_model_date[ticker] = None
            return False

    def _ensure_model_for_today(self, ticker: str) -> bool:
        """
        Ensure that for the current algo time we have the correct snapshot loaded.

        Chooses the latest train_date <= today. If today's date is before the first
        train_date, returns False and disables model IV for that day.
        """
        dates = self.train_dates.get(ticker, [])
        if not dates:
            return False

        today = self.algo.Time.date()
        eligible = [d for d in dates if d <= today]
        if not eligible:
            # Backtest date is before first training snapshot
            self.algo.Debug(
                f"[NeuralSDE] No snapshot for {ticker} on {today} "
                f"(before first train date {dates[0]})"
            )
            return False

        best_date = max(eligible)

        # Already loaded the right snapshot
        if (
            self.current_model_date.get(ticker) == best_date
            and self.models.get(ticker) is not None
        ):
            return True

        # Need to (re)load snapshot
        return self._load_snapshot(ticker, best_date)

    # ---- public API used by your strategy -------------------------

    def get_model_iv(self, ticker, spot, strike, days_to_expiry, right: OptionRight):
        """
        Computes model-implied IV using the Neural SDE snapshot appropriate
        for the current date.
        """
        # Make sure we have the right model loaded for today's date
        if not self._ensure_model_for_today(ticker):
            return None

        if spot is None or spot <= 0 or strike <= 0 or days_to_expiry < 0:
            return None

        T_years = days_to_expiry / 365.0
        if T_years <= 0:
            return None

        model = self.models.get(ticker, None)
        if model is None:
            return None

        price_model = forecast_option_price_neural_sde(
            model=model,
            S0=spot,
            K=strike,
            days_to_maturity=days_to_expiry,
            r=RISK_FREE,
            q=DIVIDEND_YIELD,
            n_paths=512,
            n_steps=30,
            device=self.device,
            right=right,
        )

        otype = "call" if right == OptionRight.Call else "put"
        iv_model = implied_vol(
            S=spot,
            K=strike,
            T=T_years,
            r=RISK_FREE,
            q=DIVIDEND_YIELD,
            market_price=price_model,
            otype=otype,
        )
        if not np.isfinite(iv_model) or iv_model <= 0:
            return None
        return iv_model


# --- QC Algorithm using the Neural SDE signal ------------------------


class VolatilityStrategiesQC(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 6, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(200000)

        # Focus on SPY European-style modeling with Black–Scholes
        self.UNIVERSE = ["SPY"]
        self.index_symbol = "SPY"

        self.starting_equity = self.Portfolio.TotalPortfolioValue
        self.max_drawdown = 0.05
        self.target_return = 10.0
        self.max_capital_per_trade = 0.03
        self.hold_days = 5
        self.longVolGap = 0.05
        self.tailGap = 0.08
        self.shortVolGap = 0.05      # symmetric 5 vol points
        self.tailShortGap = 0.08
        self.max_contracts_per_leg = 5

        self.current_slice = None
        self.option_map = {}
        for t in self.UNIVERSE:
            self.AddEquity(t, Resolution.Minute)
            opt = self.AddOption(t)
            opt.SetFilter(self.OptionFilter)
            self.option_map[t] = opt.Symbol

        self.trade_records = {}
        self.rv_lookback_days = 30

        self.sde_model = NeuralSDEModel(self)

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen(self.UNIVERSE[0], 10),
            self.DailyRebalance,
        )

        self.Debug("Neural SDE Vol Strategies Initialized")

    def OptionFilter(self, universe):
        return (
            universe.Strikes(-10, 10)
            .Expiration(timedelta(days=7), timedelta(days=180))
        )

    def OnData(self, slice: Slice):
        self.current_slice = slice

    # --- utilities ---
    def get_underlying_price(self, ticker):
        sym = self.Securities[ticker].Symbol
        return self.Securities[sym].Price

    def choose_atm_strike(self, contracts, spot):
        strikes = sorted({c.Strike for c in contracts})
        if not strikes:
            return None
        return min(strikes, key=lambda s: abs(s - spot))

    def choose_otm_strike(self, contracts, spot, direction="call", moneyness=1.06):
        target = spot * moneyness if direction == "call" else spot / moneyness
        strikes = sorted({c.Strike for c in contracts})
        if not strikes:
            return None
        return min(strikes, key=lambda s: abs(s - target))

    def find_contract(self, contracts, expiry, strike, right):
        for c in contracts:
            if (
                c.Expiry.date() == expiry.date()
                and abs(c.Strike - strike) < 1e-6
                and c.Right == right
            ):
                return c
        return None

    def model_iv_gap(self, ticker, contract):
        """
        Model IV gap = IV(model) - IV(market), using:
        - Neural SDE -> price -> IV via Black–Scholes
        - Market IV from contract.ImpliedVolatility if available,
          otherwise back out from option mid price via Black–Scholes.
        """
        if contract is None:
            return None

        spot = self.get_underlying_price(ticker)
        if spot is None or spot <= 0:
            return None

        days_to_exp = (contract.Expiry.date() - self.Time.date()).days
        if days_to_exp <= 0:
            return None
        T_years = days_to_exp / 365.0

        # --- market IV ---
        mkt_iv = getattr(contract, "ImpliedVolatility", None)
        if mkt_iv is None or np.isnan(mkt_iv) or mkt_iv <= 0:
            bid, ask, last = contract.BidPrice, contract.AskPrice, contract.LastPrice
            if (
                np.isfinite(bid)
                and np.isfinite(ask)
                and bid > 0
                and ask > 0
            ):
                mid_price = 0.5 * (bid + ask)
            else:
                mid_price = last

            if not np.isfinite(mid_price) or mid_price <= 0:
                return None

            otype = "call" if contract.Right == OptionRight.Call else "put"
            mkt_iv = implied_vol(
                S=spot,
                K=contract.Strike,
                T=T_years,
                r=RISK_FREE,
                q=DIVIDEND_YIELD,
                market_price=mid_price,
                otype=otype,
            )
            if not np.isfinite(mkt_iv) or mkt_iv <= 0:
                return None

        # --- model IV ---
        iv_model = self.sde_model.get_model_iv(
            ticker=ticker,
            spot=spot,
            strike=contract.Strike,
            days_to_expiry=days_to_exp,
            right=contract.Right,
        )
        if iv_model is None:
            return None

        return iv_model - mkt_iv

    # --- order helpers ---
    def place_option_legs_limit(self, legs, tag, quantity=1):
        order_ids = []
        for contract_symbol, dirn in legs:
            if contract_symbol not in self.Securities:
                self.Debug(f"Contract not in cache: {contract_symbol}")
                continue
            sec = self.Securities[contract_symbol]
            bid, ask = sec.BidPrice, sec.AskPrice
            if (
                not np.isfinite(bid)
                or not np.isfinite(ask)
                or ask <= 0
                or bid <= 0
            ):
                price = sec.Price
            else:
                price = 0.5 * (bid + ask)
            order = self.LimitOrder(contract_symbol, dirn * quantity, price, tag=tag)
            order_ids.append(order.OrderId)
        return order_ids

    def long_straddle(self, ticker, contracts, expiry, strike, notional_cap):
        m = 100
        call = self.find_contract(contracts, expiry, strike, OptionRight.Call)
        put = self.find_contract(contracts, expiry, strike, OptionRight.Put)
        if call is None or put is None:
            return None

        call_mid = (
            (call.BidPrice + call.AskPrice) * 0.5
            if call.BidPrice > 0 and call.AskPrice > 0
            else call.LastPrice
        )
        put_mid = (
            (put.BidPrice + put.AskPrice) * 0.5
            if put.BidPrice > 0 and put.AskPrice > 0
            else put.LastPrice
        )
        if not np.isfinite(call_mid) or not np.isfinite(put_mid):
            return None

        cost_per = (call_mid + put_mid) * m
        if cost_per <= 0:
            return None

        n = max(1, int(notional_cap // cost_per))
        n = min(n, self.max_contracts_per_leg)

        tag = f"LongStraddle_{ticker}_{strike}_{expiry.date()}"
        legs = [(call.Symbol, +1), (put.Symbol, +1)]

        self.Debug(
            f"Placing Long Straddle {ticker} {expiry.date()} {strike} x{n} cost~{cost_per * n:.2f}"
        )
        ids = self.place_option_legs_limit(legs, tag, quantity=n)
        if ids:
            self.trade_records[tag] = {
                "type": "long_straddle",
                "ticker": ticker,
                "expiry": expiry,
                "strike": strike,
                "contracts": n,
                "enter_time": self.Time,
                "tag": tag,
            }
        return ids

    def long_strangle(
        self, ticker, contracts, expiry, call_strike, put_strike, notional_cap
    ):
        m = 100
        call = self.find_contract(contracts, expiry, call_strike, OptionRight.Call)
        put = self.find_contract(contracts, expiry, put_strike, OptionRight.Put)
        if call is None or put is None:
            return None

        call_mid = (
            (call.BidPrice + call.AskPrice) * 0.5
            if call.BidPrice > 0 and call.AskPrice > 0
            else call.LastPrice
        )
        put_mid = (
            (put.BidPrice + put.AskPrice) * 0.5
            if put.BidPrice > 0 and put.AskPrice > 0
            else put.LastPrice
        )
        if not np.isfinite(call_mid) or not np.isfinite(put_mid):
            return None

        cost_per = (call_mid + put_mid) * m
        if cost_per <= 0:
            return None

        n = max(1, int(notional_cap // cost_per))
        n = min(n, self.max_contracts_per_leg)

        tag = f"LongStrangle_{ticker}_{put_strike}_{call_strike}_{expiry.date()}"
        legs = [(call.Symbol, +1), (put.Symbol, +1)]

        self.Debug(
            f"Placing Long Strangle {ticker} {expiry.date()} "
            f"{put_strike}/{call_strike} x{n} cost~{cost_per * n:.2f}"
        )
        ids = self.place_option_legs_limit(legs, tag, quantity=n)
        if ids:
            self.trade_records[tag] = {
                "type": "long_strangle",
                "ticker": ticker,
                "expiry": expiry,
                "call_strike": call_strike,
                "put_strike": put_strike,
                "contracts": n,
                "enter_time": self.Time,
                "tag": tag,
            }
        return ids

    # --- exits ---
    def exit_by_time(self):
        to_close = []
        for tag, rec in list(self.trade_records.items()):
            if (self.Time - rec["enter_time"]).days >= self.hold_days:
                to_close.append((tag, rec))
        for tag, rec in to_close:
            self.CloseRecord(rec, reason="time_expired")

    def CloseRecord(self, rec, reason="manual"):
        self.Debug(f"Closing record {rec['tag']} reason={reason}")
        ticker = rec.get("ticker", rec.get("single", None))
        if ticker is None:
            return
        under_symbol = self.Securities[ticker].Symbol
        for symbol in list(self.Portfolio.Keys):
            if symbol.SecurityType != SecurityType.Option:
                continue
            if symbol.Underlying != under_symbol:
                continue
            qty = self.Portfolio[symbol].Quantity
            if qty != 0:
                self.MarketOrder(symbol, -qty)
        if rec["tag"] in self.trade_records:
            del self.trade_records[rec["tag"]]
    
    def has_open_for_expiry(self, ticker, expiry):
        """
        Returns True if there is any open option position for this ticker
        with the given expiry date.
        """
        under_symbol = self.Securities[ticker].Symbol
        for symbol, holding in self.Portfolio.items():
            if symbol.SecurityType != SecurityType.Option:
                continue
            if symbol.Underlying != under_symbol:
                continue
            if holding.Quantity == 0:
                continue
            # symbol.ID.Date is the option expiry date
            if symbol.ID.Date == expiry:
                return True
        return False

    # --- daily rebalance ---
    def DailyRebalance(self):
        # ---- termination checks ----
        equity = self.Portfolio.TotalPortfolioValue
        dd = (equity - self.starting_equity) / self.starting_equity

        if dd <= -self.max_drawdown:
            self.Debug(f"Max drawdown hit ({dd:.1%}), liquidating and quitting.")
            self.Liquidate()
            self.Quit("Max drawdown reached")
            return

        if dd >= self.target_return:
            self.Debug(f"Target return hit ({dd:.1%}), locking in and quitting.")
            self.Liquidate()
            self.Quit("Target return reached")
            return

        self.exit_by_time()
        slice_data = self.current_slice
        if slice_data is None:
            return

        for ticker in self.UNIVERSE:
            opt_symbol = self.option_map.get(ticker)
            if opt_symbol is None:
                continue
            chain = slice_data.OptionChains.get(opt_symbol)
            if chain is None or len(chain) == 0:
                continue

            contracts = [c for c in chain]
            expiries = sorted({c.Expiry for c in contracts})
            if not expiries:
                continue

            near_expiry = expiries[0]

            # don't stack multiple structures on the same ticker+expiry
            if self.has_open_for_expiry(ticker, near_expiry):
                continue

            spot = self.get_underlying_price(ticker)
            if spot is None or spot <= 0:
                continue


            atm_strike = self.choose_atm_strike(contracts, spot)
            otm_call_strike = self.choose_otm_strike(contracts, spot, "call", 1.06)
            otm_put_strike = self.choose_otm_strike(contracts, spot, "put", 1.06)
            if atm_strike is None:
                continue

            notional_cap = self.Portfolio.TotalPortfolioValue * self.max_capital_per_trade

            atm_call = self.find_contract(
                contracts, near_expiry, atm_strike, OptionRight.Call
            )
            atm_put = self.find_contract(
                contracts, near_expiry, atm_strike, OptionRight.Put
            )
            gap_call = self.model_iv_gap(ticker, atm_call)
            gap_put = self.model_iv_gap(ticker, atm_put)

            if (
                gap_call is not None
                and gap_put is not None
                and gap_call > self.longVolGap
                and gap_put > self.longVolGap
            ):
                self.long_straddle(
                    ticker, contracts, near_expiry, atm_strike, notional_cap
                )

            if otm_call_strike and otm_put_strike:
                otm_call = self.find_contract(
                    contracts, near_expiry, otm_call_strike, OptionRight.Call
                )
                otm_put = self.find_contract(
                    contracts, near_expiry, otm_put_strike, OptionRight.Put
                )
                gap_call_otm = self.model_iv_gap(ticker, otm_call)
                gap_put_otm = self.model_iv_gap(ticker, otm_put)

                if (
                    (gap_call_otm is not None and gap_call_otm > self.tailGap)
                    or (gap_put_otm is not None and gap_put_otm > self.tailGap)
                ):
                    self.long_strangle(
                        ticker,
                        contracts,
                        near_expiry,
                        otm_call_strike,
                        otm_put_strike,
                        notional_cap,
                    )
                # --- Short ATM straddle when vol is rich ---
                '''
                if (
                    gap_call is not None
                    and gap_put is not None
                    and gap_call < -self.shortVolGap
                    and gap_put < -self.shortVolGap
                ):
                    self.short_straddle(
                        ticker, contracts, near_expiry, atm_strike, notional_cap
                    )

                # --- Short OTM strangle when wings are rich ---
                if otm_call_strike and otm_put_strike:
                    ...
                    if (
                        (gap_call_otm is not None and gap_call_otm < -self.tailShortGap)
                        or (gap_put_otm is not None and gap_put_otm < -self.tailShortGap)
                    ):
                        self.short_strangle(
                            ticker,
                            contracts,
                            near_expiry,
                            otm_call_strike,
                            otm_put_strike,
                            notional_cap,
                        )
    '''

        self.Debug(f"EOD {self.Time.date()} open_trades={len(self.trade_records)}")
