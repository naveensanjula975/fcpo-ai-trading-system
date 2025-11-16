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
        """Convert dataframe to feature array with padding to match model input size."""
        feature_cols = [col for col in df.columns if col not in ("time",)]
        features = df[feature_cols].to_numpy()
        
        # Pad features to match model input_size (32)
        # This is a simplified approach for testing
        target_size = 32
        if features.shape[1] < target_size:
            padding = np.zeros((features.shape[0], target_size - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > target_size:
            features = features[:, :target_size]
        
        # Add sequence dimension: (batch=1, seq_len, features)
        return features[None, :, :]

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
