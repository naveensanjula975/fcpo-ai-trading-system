from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on OHLCV data."""
    # Add RSI, EMA, etc. Placeholder: simple moving averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["rsi_14"] = _compute_rsi(df["close"], period=14)
    return df.dropna()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain).rolling(period).mean()
    roll_down = pd.Series(loss).rolling(period).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=series.index)
