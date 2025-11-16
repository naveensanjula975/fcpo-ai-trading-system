from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from fcpo_trading.core.config import settings


class ModelProtocol(Protocol):
    """Protocol representing required model interface."""

    def predict(self, features: np.ndarray) -> dict:
        ...


class LSTMModel(nn.Module):
    # Placeholder architecture
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


@dataclass
class TorchModelWrapper(ModelProtocol):
    """AI model wrapper providing inference for trading signals."""
    model: nn.Module
    device: torch.device

    def predict(self, features: np.ndarray) -> dict:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # Example mapping: [BUY, SELL, HOLD]
        classes = ["BUY", "SELL", "HOLD"]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx] * 100.0)
        return {"signal": classes[idx], "confidence": confidence}


class ModelFactory:
    """Factory for loading model instances."""

    @staticmethod
    def create(model_type: str = "lstm") -> ModelProtocol:
        model_path = Path(settings.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "lstm":
            # NOTE: adjust sizes according to training config
            model = LSTMModel(input_size=32, hidden_size=64, num_layers=2, output_size=3)
            
            # Load pre-trained weights if available, otherwise use random initialization
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
            else:
                # For testing/development: use untrained model with warning
                import logging
                logging.warning(f"Model file not found at {model_path}. Using untrained model.")
            
            model.to(device)
            return TorchModelWrapper(model=model, device=device)

        msg = f"Unsupported model_type: {model_type}"
        raise ValueError(msg)
