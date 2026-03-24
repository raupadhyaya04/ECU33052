"""
le_fund/core/reporting.py
──────────────────────────
ReportEngine:    performance metrics, allocation tables, charts.
HoldingsAnalyser: live FX rates, whole-share blotter, CSV export.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from tqdm import tqdm

from le_fund.config import FUND_CAPITAL, OUTPUT_DIR, RISK_FREE_RATE
from le_fund.core.risk import RiskEngine


# ════════════════════════════════════════════════════════════════════════════
# ReportEngine
# ════════════════════════════════════════════════════════════════════════════

class ReportEngine:
    """Performance reports, risk analytics, and visualisations."""

    @staticmethod
    def performance_metrics(result: pd.DataFrame, name: str) -> Dict:
        rets = result["daily_return"].dropna()
        nav  = result["nav"]

        total_return  = nav.iloc[-1] / nav.iloc[0] - 1
        years         = (nav.index[-1] - nav.index[0]).days / 365.25
        cagr          = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        vol           = rets.std() * np.sqrt(252)
        sharpe        = (cagr - RISK_FREE_RATE) / vol if vol > 0 else 0
        sortino_vol   = rets[rets < 0].std() * np.sqrt(252)
        sortino       = (cagr - RISK_FREE_RATE) / sortino_vol if sortino_vol > 0 else 0
        max_dd        = result["drawdown"].min()
        calmar        = cagr / abs(max_dd) if max_dd != 0 else 0
        avg_turnover  = result["turnover"].mean()

        return {
            "Strategy":     name,
            "Total Return": f"{total_return:.1%}",
            "CAGR":         f"{cagr:.1%}",
            "Volatility":   f"{vol:.1%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio":f"{sortino:.2f}",
            "Max Drawdown": f"{max_dd:.1%}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Avg Turnover": f"{avg_turnover:.2%}",
            "Final NAV":    f"${nav.iloc[-1]:,.0f}",
        }

    @staticmethod
    def print_allocation(
        weights: pd.Series,
        market_map: Dict[str, str],
        ticker_names: Dict[str, str],
        capital: float,
    ):
        """Pretty-print the current portfolio allocation."""
        active = weights[weights > 0.001].sort_values(ascending=False)
        cash   = max(0, 1 - active.sum())

        print(f"\n{'═' * 75}")
        print(f"  PORTFOLIO ALLOCATION  ({len(active)} positions, ${capital:,.0f} capital)")
        print(f"{'═' * 75}")
        print(f"  {'Ticker':<10} {'Name':<28} {'Market':<22} {'Weight':>7} {'$ Value':>10}")
        print(f"  {'─' * 10} {'─' * 28} {'─' * 22} {'─' * 7} {'─' * 10}")

        for ticker, w in active.items():
            name   = ticker_names.get(ticker, ticker)[:27]
            market = market_map.get(ticker, "Unknown")[:21]
            print(f"  {ticker:<10} {name:<28} {market:<22} {w:>6.1%} ${w * capital:>9,.0f}")

        print(f"  {'─' * 10} {'─' * 28} {'─' * 22} {'─' * 7} {'─' * 10}")
        print(f"  {'CASH':<10} {'Cash Reserve':<28} {'—':<22} {cash:>6.1%} ${cash * capital:>9,.0f}")
        print(f"  {'TOTAL':<10} {'':<28} {'':<22} {1.0:>6.1%} ${capital:>9,.0f}")

    @staticmethod
    def plot_results(results: Dict[str, pd.DataFrame]):
        """NAV, drawdown, and rolling Sharpe charts."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), dpi=120)
        fig.suptitle(
            "Life on the Edge Fund — Backtest Results",
            fontsize=16, fontweight="bold", y=0.98,
        )
        colors = {
            "Max Sharpe":    "#2196F3",
            "Min Volatility":"#4CAF50",
            "Risk Parity":   "#FF9800",
        }

        ax1 = axes[0]
        for name, df in results.items():
            ax1.plot(df.index, df["nav"], label=name,
                     color=colors.get(name, "#999"), linewidth=1.5)
        ax1.axhline(y=FUND_CAPITAL, color="gray", linestyle="--", alpha=0.5,
                    label=f"Initial (${FUND_CAPITAL:,.0f})")
        ax1.set_title("Portfolio NAV", fontweight="bold")
        ax1.set_ylabel("NAV ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        ax2 = axes[1]
        for name, df in results.items():
            ax2.fill_between(df.index, df["drawdown"], 0,
                             color=colors.get(name, "#999"), alpha=0.3, label=name)
        ax2.set_title("Drawdowns", fontweight="bold")
        ax2.set_ylabel("Drawdown")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        ax3 = axes[2]
        for name, df in results.items():
            rolling_ret  = df["daily_return"].rolling(63).mean() * 252
            rolling_vol  = df["daily_return"].rolling(63).std() * np.sqrt(252)
            rolling_sh   = (rolling_ret - RISK_FREE_RATE) / rolling_vol
            ax3.plot(df.index, rolling_sh, label=name,
                     color=colors.get(name, "#999"), linewidth=1, alpha=0.8)
        ax3.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax3.set_title("Rolling 3-Month Sharpe Ratio", fontweight="bold")
        ax3.set_ylabel("Sharpe")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        plt.tight_layout()
        chart_path = OUTPUT_DIR / "backtest_results.png"
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        print(f"\n  📊 Charts saved to {chart_path}")

    @staticmethod
    def plot_risk_attribution(
        weights: pd.Series,
        cov: pd.DataFrame,
        market_map: Dict[str, str],
    ):
        """Risk contribution by asset and by market."""
        active = weights[weights > 0.001]
        if len(active) == 0:
            return

        cov_sub   = cov.loc[active.index, active.index]
        rc        = RiskEngine.risk_contributions(active.values, cov_sub.values)
        rc_series = pd.Series(rc, index=active.index)

        market_rc: Dict[str, float] = {}
        for ticker, risk_c in rc_series.items():
            market = market_map.get(ticker, "Other")
            market_rc[market] = market_rc.get(market, 0) + risk_c

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=120)
        fig.suptitle("Risk Attribution", fontsize=14, fontweight="bold")

        rc_series.sort_values().plot(kind="barh", ax=ax1, color="#2196F3")
        ax1.set_title("Risk Contribution by Asset")
        ax1.set_xlabel("% of Total Risk")

        pd.Series(market_rc).sort_values().plot(kind="barh", ax=ax2, color="#FF9800")
        ax2.set_title("Risk Contribution by Market")
        ax2.set_xlabel("% of Total Risk")

        plt.tight_layout()
        risk_path = OUTPUT_DIR / "risk_attribution.png"
        plt.savefig(risk_path, bbox_inches="tight")
        plt.close()
        print(f"  📊 Risk attribution saved to {risk_path}")

    @staticmethod
    def plot_correlation_heatmap(returns: pd.DataFrame, weights: pd.Series):
        """Correlation heatmap for the selected portfolio assets."""
        active = weights[weights > 0.001].index.tolist()
        if len(active) < 2:
            return

        corr = RiskEngine.correlation_matrix(returns[active])
        fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(
            "Correlation Matrix — Selected Portfolio Assets",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        corr_path = OUTPUT_DIR / "correlation_heatmap.png"
        plt.savefig(corr_path, bbox_inches="tight")
        plt.close()
        print(f"  📊 Correlation heatmap saved to {corr_path}")


# ════════════════════════════════════════════════════════════════════════════
# HoldingsAnalyser
# ════════════════════════════════════════════════════════════════════════════

class HoldingsAnalyser:
    """
    Produces an executable trade blotter for the portfolio.

    · Prices are shown in the stock's HOME CURRENCY (GBp, KRW, EUR …)
    · Quantities are WHOLE SHARES
    · Notional (USD) = qty × local_price × FX
    · Live FX rates from yfinance
    """

    SUFFIX_CURRENCY = {
        ".L":  "GBp", ".PA": "EUR", ".DE": "EUR",
        ".SW": "CHF", ".CO": "DKK", ".T":  "JPY",
        ".KS": "KRW", ".HK": "HKD", ".NS": "INR",
        ".F":  "EUR",
    }
    CCY_SYMBOL = {
        "USD": "$",  "GBp": "p",    "GBP": "£",   "EUR": "€",
        "CHF": "CHF ","JPY": "¥",   "KRW": "₩",   "HKD": "HK$",
        "INR": "₹",  "DKK": "kr",
    }

    def __init__(
        self,
        weights: pd.Series,
        capital: float,
        market_map: Dict[str, str],
        ticker_names: Dict[str, str],
        prices: pd.DataFrame,
    ):
        self.weights      = weights[weights > 0.001].sort_values(ascending=False)
        self.capital      = capital
        self.market_map   = market_map
        self.ticker_names = ticker_names
        self.prices       = prices
        self.purchase_date = datetime.now().strftime("%Y-%m-%d")
        self._fx_cache: Dict[str, float] = {"USD": 1.0}

    # ── FX helpers ─────────────────────────────────────────────────────

    def _fetch_fx_rates(self, currencies: set):
        need = {("GBP" if c == "GBp" else c) for c in currencies if c != "USD"}
        to_fetch = [base for base in need if base not in self._fx_cache]
        if not to_fetch:
            if "GBP" in self._fx_cache:
                self._fx_cache["GBp"] = self._fx_cache["GBP"] / 100.0
            return

        def _get_rate(base: str) -> tuple[str, float]:
            pair = f"{base}USD=X"
            try:
                hist = yf.Ticker(pair).history(period="1d")
                if not hist.empty:
                    return base, float(hist["Close"].iloc[-1])
                inv  = f"USD{base}=X"
                hist2 = yf.Ticker(inv).history(period="1d")
                if not hist2.empty:
                    return base, 1.0 / float(hist2["Close"].iloc[-1])
                print(f"    ⚠ No FX data for {base}, using 1.0")
                return base, 1.0
            except Exception as exc:
                print(f"    ⚠ FX fetch failed for {base}: {exc}")
                return base, 1.0

        with ThreadPoolExecutor(max_workers=len(to_fetch)) as pool:
            for base, rate in pool.map(_get_rate, to_fetch):
                self._fx_cache[base] = rate

        if "GBP" in self._fx_cache:
            self._fx_cache["GBp"] = self._fx_cache["GBP"] / 100.0

    def _to_usd(self, amount_local: float, ccy: str) -> float:
        return amount_local * self._fx_cache.get(ccy, 1.0)

    @staticmethod
    def _infer_currency(ticker: str) -> str:
        for suffix, ccy in HoldingsAnalyser.SUFFIX_CURRENCY.items():
            if ticker.endswith(suffix):
                return ccy
        return "USD"

    @staticmethod
    def _format_market_cap(cap) -> str:
        if cap is None or (isinstance(cap, float) and math.isnan(cap)):
            return "N/A"
        if cap >= 1e12:
            return f"${cap / 1e12:.1f}T"
        elif cap >= 1e9:
            return f"${cap / 1e9:.1f}B"
        elif cap >= 1e6:
            return f"${cap / 1e6:.0f}M"
        return f"${cap:,.0f}"

    def _fmt_price(self, price: float, ccy: str) -> str:
        sym = self.CCY_SYMBOL.get(ccy, ccy + " ")
        return f"{sym}{price:>,.0f}" if ccy in ("KRW", "GBp") else f"{sym}{price:>,.2f}"

    # ── Main fetch ─────────────────────────────────────────────────────

    def fetch_metadata(self) -> pd.DataFrame:
        """Parallel per-ticker metadata fetch; returns blotter DataFrame."""
        active_tickers = self.weights.index.tolist()
        currencies     = {self._infer_currency(t) for t in active_tickers}
        print(f"\n  Fetching FX rates for: {currencies - {'USD'}}")
        self._fetch_fx_rates(currencies)
        print(f"  FX rates (→ USD): { {k: round(v, 6) for k, v in self._fx_cache.items() if k != 'USD'} }")
        print(f"  Fetching metadata for {len(active_tickers)} holdings in parallel...")

        def _fetch_one(sym: str) -> dict:
            target_usd  = self.weights[sym] * self.capital
            name        = self.ticker_names.get(sym, sym)
            market      = self.market_map.get(sym, "Unknown")
            currency    = self._infer_currency(sym)
            local_price = float(self.prices[sym].iloc[-1]) if sym in self.prices.columns else float("nan")
            beta        = float("nan")
            market_cap  = float("nan")
            try:
                info       = yf.Ticker(sym).info
                beta       = info.get("beta", float("nan"))
                market_cap = info.get("marketCap", float("nan"))
                api_ccy    = info.get("currency", None)
                if api_ccy:
                    currency = api_ccy
            except Exception:
                pass

            fx_rate       = self._fx_cache.get(currency, 1.0)
            target_local  = target_usd / fx_rate if fx_rate > 0 else 0
            quantity      = int(target_local / local_price) if local_price > 0 else 0
            notional_usd  = self._to_usd(quantity * local_price, currency)

            return {
                "Name":           name,
                "Symbol":         sym,
                "Price (Local)":  local_price,
                "Currency":       currency,
                "Quantity":       quantity,
                "Notional (Local)": quantity * local_price,
                "Notional (USD)": notional_usd,
                "Beta":           beta,
                "Market":         market,
                "Market Cap":     market_cap,
            }

        with ThreadPoolExecutor(max_workers=min(15, len(active_tickers))) as pool:
            records = list(tqdm(
                pool.map(_fetch_one, active_tickers),
                total=len(active_tickers),
                desc="  Holdings",
            ))

        df = pd.DataFrame(records)
        df["Weight"] = df["Notional (USD)"] / self.capital
        self._cash            = self.capital - df["Notional (USD)"].sum()
        self._total_invested  = df["Notional (USD)"].sum()
        return df

    # ── Reporting ──────────────────────────────────────────────────────

    def print_holdings_table(self, df: pd.DataFrame, strategy_name: str):
        width = 160
        print(f"\n{'═' * width}")
        print(
            f"  HOLDINGS DETAIL — {strategy_name}  "
            f"(${self.capital:,.0f} capital, {len(df)} positions, "
            f"as of {self.purchase_date})  •  PRICES IN HOME CCY  •  WHOLE SHARES"
        )
        print(f"{'═' * width}")
        print(
            f"  {'Name':<25} {'Symbol':<12} {'Price':>14} {'CCY':<5} "
            f"{'Qty':>7} {'Notional (USD)':>15} "
            f"{'Weight':>8} {'Beta':>6} {'Market':<22} {'Mkt Cap':>12}"
        )
        sep = (
            f"  {'─' * 25} {'─' * 12} {'─' * 14} {'─' * 5} "
            f"{'─' * 7} {'─' * 15} "
            f"{'─' * 8} {'─' * 6} {'─' * 22} {'─' * 12}"
        )
        print(sep)

        total_notional = 0.0
        for _, row in df.iterrows():
            price_str = self._fmt_price(row["Price (Local)"], row["Currency"])
            beta_str  = f"{row['Beta']:>5.2f}" if pd.notna(row["Beta"]) else "  N/A"
            total_notional += row["Notional (USD)"]
            print(
                f"  {str(row['Name'])[:24]:<25} {str(row['Symbol'])[:11]:<12} "
                f"{price_str:>14} {str(row['Currency'])[:4]:<5} "
                f"{int(row['Quantity']):>7,d} ${row['Notional (USD)']:>13,.2f} "
                f"{row['Weight']:>7.1%} {beta_str:>6} "
                f"{str(row['Market'])[:21]:<22} {self._format_market_cap(row['Market Cap']):>12}"
            )

        betas            = df["Beta"].dropna()
        w_beta           = df.loc[betas.index, "Weight"]
        port_beta        = (betas * w_beta).sum() / w_beta.sum() if len(betas) > 0 else 0
        cash             = self.capital - total_notional

        print(sep)
        print(
            f"  {'CASH':<25} {'—':<12} {'—':>14} {'USD':<5} "
            f"{'—':>7} ${cash:>13,.2f} "
            f"{cash / self.capital:>7.1%} {'—':>6} {'—':<22} {'—':>12}"
        )
        print(
            f"  {'TOTAL':<25} {'':12} {'':>14} {'':5} "
            f"{'':>7} ${self.capital:>13,.2f} "
            f"{'100.0%':>8} {port_beta:>5.2f} {'':22} {'':>12}"
        )
        print(f"{'═' * width}")
        print(
            f"  ℹ  Invested: ${total_notional:,.2f} "
            f"({total_notional / self.capital:.1%})  |  "
            f"Cash remainder: ${cash:,.2f} ({cash / self.capital:.1%})"
        )

    def save_to_csv(self, df: pd.DataFrame, strategy_name: str):
        safe_name = strategy_name.lower().replace(" ", "_")
        path      = OUTPUT_DIR / f"holdings_{safe_name}.csv"
        df_out    = df.copy()
        df_out["Market Cap (formatted)"] = df_out["Market Cap"].apply(
            self._format_market_cap
        )
        df_out.to_csv(path, index=False, float_format="%.4f")
        print(f"  💾 Holdings saved to {path}")
        return path
