from __future__ import annotations

from fcpo_trading.services.ml_service import MLService


def test_ml_service_generate_signal_basic() -> None:
    service = MLService()
    bar = {
        "open": 3845.0,
        "high": 3852.0,
        "low": 3843.0,
        "close": 3850.0,
        "volume": 1250.0,
    }
    result = service.generate_signal(bar)
    assert "signal" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 100.0
