from __future__ import annotations

from typing import Protocol, List

from fcpo_trading.schemas.signal import SignalResponse


class SignalObserver(Protocol):
    """Observer interface for signal notifications."""

    async def notify(self, signal: SignalResponse) -> None:
        ...


class NotificationManager:
    """Manage observers and notify them when new signals are created."""

    def __init__(self) -> None:
        self._observers: List[SignalObserver] = []

    def register(self, observer: SignalObserver) -> None:
        self._observers.append(observer)

    def unregister(self, observer: SignalObserver) -> None:
        self._observers.remove(observer)

    async def notify_all(self, signal: SignalResponse) -> None:
        for observer in self._observers:
            await observer.notify(signal)
