"""Backtesting script for FCPO AI trading signals."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dataclasses import dataclass

from fcpo_trading.ml.model import ModelFactory
from fcpo_trading.ml.preprocessing import compute_indicators


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: datetime | None
    signal: str
    entry_price: float
    exit_price: float | None
    tp_levels: List[float]
    sl_level: float
    result: float | None
    status: str


class Backtester:
    """Backtesting engine for FCPO trading strategy."""
    
    def __init__(self, initial_capital: float = 100000.0) -> None:
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.model = ModelFactory.create()
    
    def run(
        self,
        df: pd.DataFrame,
        risk_per_trade: float = 0.02,
    ) -> None:
        """Run backtest on historical data."""
        active_trade: Trade | None = None
        
        for idx in range(len(df)):
            if idx < 50:  # Skip initial rows for indicator warmup
                continue
            
            row = df.iloc[idx]
            
            # Check active trade for exit
            if active_trade:
                active_trade = self._check_exit(active_trade, row)
                if active_trade.status == "CLOSED":
                    self.trades.append(active_trade)
                    active_trade = None
            
            # Generate new signal if no active trade
            if not active_trade:
                signal_data = self._generate_signal(df.iloc[max(0, idx - 20) : idx + 1])
                if signal_data["signal"] != "HOLD":
                    active_trade = Trade(
                        entry_time=row["time"],
                        exit_time=None,
                        signal=signal_data["signal"],
                        entry_price=float(row["close"]),
                        exit_price=None,
                        tp_levels=signal_data["tp_levels"],
                        sl_level=signal_data["sl_level"],
                        result=None,
                        status="ACTIVE",
                    )
        
        # Close any remaining active trade
        if active_trade:
            active_trade.exit_price = df.iloc[-1]["close"]
            active_trade.exit_time = df.iloc[-1]["time"]
            active_trade.result = self._calculate_pnl(active_trade)
            active_trade.status = "CLOSED"
            self.trades.append(active_trade)
    
    def _generate_signal(self, df_slice: pd.DataFrame) -> dict:
        """Generate signal using ML model."""
        # Simplified – use last bar only
        bar = df_slice.iloc[-1]
        features = np.random.randn(1, 20, 32).astype(np.float32)  # Placeholder
        result = self.model.predict(features)
        
        entry = float(bar["close"])
        step = entry * 0.005
        if result["signal"] == "BUY":
            tp = [entry + step, entry + 2 * step, entry + 3 * step]
            sl = entry - 1.5 * step
        elif result["signal"] == "SELL":
            tp = [entry - step, entry - 2 * step, entry - 3 * step]
            sl = entry + 1.5 * step
        else:
            tp = []
            sl = entry
        
        return {
            "signal": result["signal"],
            "confidence": result["confidence"],
            "tp_levels": tp,
            "sl_level": sl,
        }
    
    def _check_exit(self, trade: Trade, row: pd.Series) -> Trade:
        """Check if trade should exit (TP/SL hit)."""
        high = float(row["high"])
        low = float(row["low"])
        
        if trade.signal == "BUY":
            if low <= trade.sl_level:
                trade.exit_price = trade.sl_level
                trade.exit_time = row["time"]
                trade.result = self._calculate_pnl(trade)
                trade.status = "CLOSED"
            elif high >= trade.tp_levels[0]:
                trade.exit_price = trade.tp_levels[0]
                trade.exit_time = row["time"]
                trade.result = self._calculate_pnl(trade)
                trade.status = "CLOSED"
        elif trade.signal == "SELL":
            if high >= trade.sl_level:
                trade.exit_price = trade.sl_level
                trade.exit_time = row["time"]
                trade.result = self._calculate_pnl(trade)
                trade.status = "CLOSED"
            elif low <= trade.tp_levels[0]:
                trade.exit_price = trade.tp_levels[0]
                trade.exit_time = row["time"]
                trade.result = self._calculate_pnl(trade)
                trade.status = "CLOSED"
        
        return trade
    
    def _calculate_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade."""
        if not trade.exit_price:
            return 0.0
        if trade.signal == "BUY":
            return trade.exit_price - trade.entry_price
        else:
            return trade.entry_price - trade.exit_price
    
    def print_report(self) -> None:
        """Print backtest performance report."""
        if not self.trades:
            print("No trades executed.")
            return
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.result and t.result > 0]
        losing_trades = [t for t in self.trades if t.result and t.result < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_pnl = sum(t.result for t in self.trades if t.result)
        avg_win = np.mean([t.result for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.result for t in losing_trades]) if losing_trades else 0
        
        print("\n=== Backtest Report ===")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total P&L: {total_pnl:.2f}")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Profit Factor: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest FCPO AI strategy")
    parser.add_argument("--data", type=str, default="./data/fcpo_historical.csv", help="Path to CSV data")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = compute_indicators(df)
    
    # Filter by date range
    if args.start:
        df = df[df["time"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["time"] <= pd.to_datetime(args.end)]
    
    print(f"Running backtest on {len(df)} bars...")
    backtester = Backtester(initial_capital=args.capital)
    backtester.run(df)
    backtester.print_report()


if __name__ == "__main__":
    main()
