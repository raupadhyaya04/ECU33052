"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    LIFE ON THE EDGE FUND (LE Fund)                      ║
║                                                                         ║
║  Multi-Market Portfolio Optimization Engine                             ║
║  Capital: $50,000 USD | Max Positions: 15                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

Entry point — run with:
    python main.py

All business logic lives in the le_fund package:
    le_fund/config.py              — all tunable constants & paths
    le_fund/core/numba_kernels.py  — JIT-compiled inner loops
    le_fund/core/universe.py       — UniverseLoader
    le_fund/core/data.py           — DataEngine
    le_fund/core/risk.py           — RiskEngine
    le_fund/core/optimisation.py   — OptimisationEngine
    le_fund/core/volatility.py     — VolatilityTargeter
    le_fund/core/backtest.py       — Backtester
    le_fund/core/reporting.py      — ReportEngine, HoldingsAnalyser
    le_fund/sentiment/engine.py    — SentimentEngine (FinBERT)
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()   # load HF_TOKEN from .env before anything else

warnings.filterwarnings("ignore")

# ── Package imports ──────────────────────────────────────────────────────
from le_fund.config import (
    FUND_CAPITAL,
    LOOKBACK_DAYS,
    OUTPUT_DIR,
)
from le_fund.core.universe     import UniverseLoader
from le_fund.core.data         import DataEngine
from le_fund.core.risk         import RiskEngine
from le_fund.core.optimisation import OptimisationEngine
from le_fund.core.volatility   import VolatilityTargeter
from le_fund.core.backtest     import Backtester
from le_fund.core.reporting    import ReportEngine, HoldingsAnalyser
from le_fund.sentiment.engine  import SentimentEngine


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          LIFE ON THE EDGE FUND — MODEL ENGINE             ║")
    print("║          Capital: $50,000 | Max Positions: 15             ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # ── Universe ───────────────────────────────────────────────────────
    loader       = UniverseLoader()
    tickers      = loader.get_all_tickers()
    market_map   = loader.get_market_map()
    ticker_names = loader.get_ticker_names()
    print(loader.summary())
    print()

    # ── Data ───────────────────────────────────────────────────────────
    engine = DataEngine(tickers, start="2020-01-01")
    prices = engine.download()

    # ── Risk & regime ──────────────────────────────────────────────────
    returns  = RiskEngine.compute_returns(prices)
    cov      = RiskEngine.shrinkage_covariance(returns.iloc[-LOOKBACK_DAYS:])
    ewma_vol = RiskEngine.ewma_volatility(returns)
    regime   = RiskEngine.detect_vol_regime(returns)

    print(f"\n  📈 Current Volatility Regime: {regime}")
    print(f"  📊 Avg EWMA Vol (annualised): {ewma_vol.mean():.1%}")
    print(f"  📊 Universe coverage: {len(returns.columns)} assets")

    # ── Sentiment (FinBERT via HF API, local fallback) ─────────────────
    sentiment_engine = SentimentEngine()
    sentiment_scores = sentiment_engine.score_universe(returns.columns.tolist())

    # ── Current-snapshot optimisation ─────────────────────────────────
    opt      = OptimisationEngine(
        returns.iloc[-LOOKBACK_DAYS:], cov,
        prices=prices, sentiment=sentiment_scores,
    )
    targeter = VolatilityTargeter()

    def _snapshot(method_name: str):
        w = getattr(opt, method_name)()
        return targeter.scale_weights(w, cov, regime)

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_sharpe = pool.submit(_snapshot, "max_sharpe")
        fut_minvol = pool.submit(_snapshot, "min_volatility")
        fut_rp     = pool.submit(_snapshot, "risk_parity")
        w_sharpe = fut_sharpe.result()
        w_minvol = fut_minvol.result()
        w_rp     = fut_rp.result()


    def print_portfolio_beta(weights, betas, label):
        # Align weights and betas by ticker order
        w = weights.reindex(opt.tickers).fillna(0).values
        b = opt.betas
        portfolio_beta = float(np.dot(w, b))
        print(f"    Portfolio Beta: {portfolio_beta:.3f}")

    print("\n\n" + "▓" * 60)
    print("  STRATEGY 1: MAXIMUM SHARPE RATIO")
    print("▓" * 60)
    ReportEngine.print_allocation(w_sharpe, market_map, ticker_names, FUND_CAPITAL)
    print_portfolio_beta(w_sharpe, opt.betas, "Max Sharpe")

    print("\n\n" + "▓" * 60)
    print("  STRATEGY 2: MINIMUM VOLATILITY")
    print("▓" * 60)
    ReportEngine.print_allocation(w_minvol, market_map, ticker_names, FUND_CAPITAL)
    print_portfolio_beta(w_minvol, opt.betas, "Min Volatility")

    print("\n\n" + "▓" * 60)
    print("  STRATEGY 3: RISK PARITY")
    print("▓" * 60)
    ReportEngine.print_allocation(w_rp, market_map, ticker_names, FUND_CAPITAL)
    print_portfolio_beta(w_rp, opt.betas, "Risk Parity")

    # ── Walk-forward backtests (3 in parallel) ─────────────────────────
    bt = Backtester(prices)
    backtest_jobs = [
        ("max_sharpe",  "Max Sharpe"),
        ("min_vol",     "Min Volatility"),
        ("risk_parity", "Risk Parity"),
    ]
    results = {}
    print("\n\n  Running 3 backtests in parallel (threads + numba GIL release)...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(bt.run, label, LOOKBACK_DAYS, method): label
            for method, label in backtest_jobs
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    # ── Performance summary ────────────────────────────────────────────
    print("\n\n" + "═" * 75)
    print("  PERFORMANCE SUMMARY")
    print("═" * 75)
    summary = pd.DataFrame(
        [ReportEngine.performance_metrics(df, name) for name, df in results.items()]
    ).set_index("Strategy")
    print(summary.to_string())

    # ── Charts ─────────────────────────────────────────────────────────
    print("\n\n  Generating charts...")
    ReportEngine.plot_results(results)
    ReportEngine.plot_risk_attribution(w_sharpe, cov, market_map)
    ReportEngine.plot_correlation_heatmap(returns.iloc[-LOOKBACK_DAYS:], w_sharpe)

    # ── Detailed holdings tables ───────────────────────────────────────
    strategy_specs = [
        ("Max Sharpe",     w_sharpe),
        ("Min Volatility", w_minvol),
        ("Risk Parity",    w_rp),
    ]

    def _fetch_holdings(spec):
        label, sw = spec
        a = HoldingsAnalyser(
            weights=sw, capital=FUND_CAPITAL,
            market_map=market_map, ticker_names=ticker_names, prices=prices,
        )
        hdf = a.fetch_metadata()
        return label, a, hdf

    with ThreadPoolExecutor(max_workers=3) as pool:
        for label, analyser, hdf in pool.map(_fetch_holdings, strategy_specs):
            analyser.print_holdings_table(hdf, label)
            analyser.save_to_csv(hdf, label)

    # ── Save CSVs ──────────────────────────────────────────────────────
    alloc_df = pd.DataFrame({
        "Max Sharpe": w_sharpe, "Min Vol": w_minvol, "Risk Parity": w_rp,
    })
    alloc_df = alloc_df[alloc_df.sum(axis=1) > 0.001]
    alloc_df["Market"] = alloc_df.index.map(market_map)
    alloc_df["Name"]   = alloc_df.index.map(ticker_names)
    alloc_df.to_csv(OUTPUT_DIR / "allocations.csv")
    print(f"\n  💾 Allocations saved to {OUTPUT_DIR / 'allocations.csv'}")

    for name, df in results.items():
        df.to_csv(OUTPUT_DIR / f"backtest_{name.lower().replace(' ', '_')}.csv")
    print(f"  💾 Backtest results saved to {OUTPUT_DIR}/")

    print("\n" + "═" * 75)
    print("  ✅ LE Fund model run complete!")
    print("═" * 75)


if __name__ == "__main__":
    main()