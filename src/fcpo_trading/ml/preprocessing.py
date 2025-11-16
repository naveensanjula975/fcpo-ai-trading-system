from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on OHLCV data."""
    # Add RSI, EMA, etc. For single-row input, use simple features
    if len(df) >= 20:
        df["sma_20"] = df["close"].rolling(window=20).mean()
    else:
        df["sma_20"] = df["close"]  # Fallback for short sequences
    
    if len(df) >= 50:
        df["sma_50"] = df["close"].rolling(window=50).mean()
    else:
        df["sma_50"] = df["close"]
    
    if len(df) >= 14:
        df["rsi_14"] = _compute_rsi(df["close"], period=14)
    else:
        df["rsi_14"] = 50.0  # Neutral RSI for short sequences
    
    return df.bfill().ffill()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain).rolling(period).mean()
    roll_down = pd.Series(loss).rolling(period).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=series.index)
