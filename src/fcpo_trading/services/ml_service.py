from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from fcpo_trading.ml.model import ModelFactory
from fcpo_trading.ml.preprocessing import compute_indicators


class MLService:
    """Service layer for model inference and TP/SL calculation."""

    def __init__(self) -> None:
        self.model = ModelFactory.create()

    def generate_signal(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal from a single bar of OHLCV data."""
        df = pd.DataFrame([bar])
        df = compute_indicators(df)
        features = self._to_features(df)
        result = self.model.predict(features)
        signal = result["signal"]
        confidence = result["confidence"]
        entry_price = float(bar["close"])
        tp_levels, sl_level = self._compute_tp_sl(signal, entry_price)
        return {
            "signal": signal,
            "confidence": confidence,
            "entry_price": entry_price,
            "tp_levels": tp_levels,
            "sl_level": sl_level,
        }

    def _to_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [col for col in df.columns if col not in ("time",)]
        return df[feature_cols].to_numpy()[None, :, :]

    def _compute_tp_sl(self, signal: str, entry: float) -> tuple[list[float], float]:
        # Simple placeholder – later implement Fibonacci/ATR-based logic
        step = entry * 0.005
        if signal == "BUY":
            tp = [entry + step, entry + 2 * step, entry + 3 * step]
            sl = entry - 1.5 * step
        elif signal == "SELL":
            tp = [entry - step, entry - 2 * step, entry - 3 * step]
            sl = entry + 1.5 * step
        else:
            tp = []
            sl = entry
        return tp, sl
