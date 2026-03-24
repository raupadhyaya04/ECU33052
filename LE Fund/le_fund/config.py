"""
le_fund/config.py
─────────────────
Central configuration for the Life on the Edge Fund.

All tunable constants live here so every other module imports from one
place.  Paths are resolved relative to this file's parent directory so
the package works regardless of the working directory when invoked.
"""

from pathlib import Path

# ── Directory layout ────────────────────────────────────────────────────
# config.py lives at  le_fund/config.py
# → le_fund/          is  _PKG_DIR
# → LE Fund/          is  _ROOT_DIR  (project root)
_PKG_DIR  = Path(__file__).parent          # le_fund/
_ROOT_DIR = _PKG_DIR.parent                # LE Fund/

DATA_DIR      = _ROOT_DIR / "data"
OUTPUT_DIR    = _ROOT_DIR / "output"
UNIVERSE_FILE = _ROOT_DIR / "universe.json"

# ── Fund parameters ──────────────────────────────────────────────────────

FUND_CAPITAL       = 50_000.0   # USD
MAX_POSITIONS      = 15
MIN_POSITIONS      = 7          # minimum number of positions
MIN_WEIGHT         = 0.03       # 3 % floor per position
MAX_WEIGHT         = 0.25       # 25 % cap per position
MAX_CASH           = 0.025      # 2.5 % max cash allocation
RISK_FREE_RATE     = 0.045      # annualised (~current T-bill)
TARGET_VOL         = 0.28       # 28 % annualised — sized for a high-beta universe
LOOKBACK_DAYS      = 252        # 1-year rolling window
EWMA_SPAN          = 60         # EWMA half-life for vol forecasting
REBALANCE_FREQ     = "ME"       # monthly rebalancing (month-end)
TRANSACTION_COST   = 0.001      # 10 bps round-trip
CHUNK_SIZE         = 20         # tickers per download chunk

# ── Drawdown / risk control ──────────────────────────────────────────────
CVAR_ALPHA         = 0.05       # 95 % CVaR confidence level
DD_PENALTY         = 0.5        # lighter CVaR penalty — lets Sharpe objective breathe
MAX_CORR_FILTER    = 0.92       # high-beta universe is correlated — trust the optimiser
DD_CIRCUIT_BREAK   = -0.10      # reduce exposure when DD exceeds -10 %
DD_RESUME_LEVEL    = -0.05      # resume full exposure above -5 %
DD_SCALE_FACTOR    = 0.40       # 40 % exposure during circuit break
MOMENTUM_WINDOW    = 126        # 6-month momentum lookback
SHRINKAGE_ALPHA    = 0.40       # lean into momentum signal
MIN_PORTFOLIO_BETA = 0.95       # SLSQP inequality constraint
MIN_ASSET_BETA     = 0.60       # pre-screener: exclude low-beta names

# ── Sentiment ────────────────────────────────────────────────────────────
SENTIMENT_WEIGHT    = 0.20      # share of blended expected return from FinBERT
SENTIMENT_HEADLINES = 10        # recent headlines to score per ticker
