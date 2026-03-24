"""
le_fund/core/universe.py
─────────────────────────
Loads the multi-market ticker universe from universe.json.
"""

import json
from pathlib import Path
from typing import Dict, List

from le_fund.config import UNIVERSE_FILE


class UniverseLoader:
    """Loads the multi-market universe from the JSON config file."""

    def __init__(self, filepath: Path = UNIVERSE_FILE):
        with open(filepath, "r") as f:
            self.config = json.load(f)
        self.fund_config = self.config["fund"]
        self.markets = self.config["markets"]

    def get_all_tickers(self) -> List[str]:
        """Flat, de-duplicated list of every ticker in the universe."""
        tickers = []
        for market_data in self.markets.values():
            for asset in market_data["assets"]:
                tickers.append(asset["ticker"])
        return sorted(set(tickers))

    def get_market_map(self) -> Dict[str, str]:
        """ticker → market name."""
        mapping: Dict[str, str] = {}
        for market_data in self.markets.values():
            for asset in market_data["assets"]:
                mapping[asset["ticker"]] = market_data["name"]
        return mapping

    def get_ticker_names(self) -> Dict[str, str]:
        """ticker → human-readable company name."""
        names: Dict[str, str] = {}
        for market_data in self.markets.values():
            for asset in market_data["assets"]:
                names[asset["ticker"]] = asset["name"]
        return names

    def summary(self) -> str:
        lines = [f"{'Market':<30} {'Assets':>6}"]
        lines.append("─" * 38)
        total = 0
        for market_data in self.markets.values():
            n = len(market_data["assets"])
            total += n
            lines.append(f"{market_data['name']:<30} {n:>6}")
        lines.append("─" * 38)
        lines.append(f"{'TOTAL':<30} {total:>6}")
        return "\n".join(lines)
