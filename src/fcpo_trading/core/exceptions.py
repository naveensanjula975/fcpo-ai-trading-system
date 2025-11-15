from __future__ import annotations

from typing import Any


class TradingViewAuthError(Exception):
    """Raised when TradingView webhook authentication fails."""

    def __init__(self, message: str = "Invalid TradingView API key") -> None:
        super().__init__(message)
        self.message = message


class ModelInferenceError(Exception):
    """Raised when the ML model fails to generate a prediction."""

    def __init__(self, details: Any | None = None) -> None:
        super().__init__("Model inference failed")
        self.details = details
