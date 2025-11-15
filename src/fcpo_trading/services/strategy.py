from __future__ import annotations

from typing import Protocol, Literal, Any, Dict

from fcpo_trading.services.ml_service import MLService


SignalType = Literal["BUY", "SELL", "HOLD"]


class SignalStrategy(Protocol):
    """Strategy interface for signal generation."""

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AIMLSignalStrategy:
    """ML-based signal generation strategy."""

    def __init__(self, ml_service: MLService) -> None:
        self._ml_service = ml_service

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        bar = payload["bar"]
        return self._ml_service.generate_signal(bar)


class RuleBasedSignalStrategy:
    """Simple rule-based fallback strategy."""

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        close = payload["bar"]["close"]
        open_ = payload["bar"]["open"]
        signal: SignalType = "HOLD"
        confidence = 50.0
        if close > open_:
            signal = "BUY"
        elif close < open_:
            signal = "SELL"
        return {
            "signal": signal,
            "confidence": confidence,
            "entry_price": close,
            "tp_levels": [],
            "sl_level": close,
        }
