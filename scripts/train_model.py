"""Training script for FCPO AI trading model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from fcpo_trading.ml.model import LSTMModel
from fcpo_trading.ml.preprocessing import compute_indicators


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load FCPO historical data from CSV."""
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators and prepare features."""
    df = compute_indicators(df)
    return df


def create_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Create labels based on future price movement.
    
    BUY (0): price increases > 0.5%
    SELL (1): price decreases > 0.5%
    HOLD (2): otherwise
    """
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(2, index=df.index)  # Default HOLD
    labels[future_return > 0.005] = 0  # BUY
    labels[future_return < -0.005] = 1  # SELL
    return labels


def prepare_sequences(
    df: pd.DataFrame,
    labels: pd.Series,
    seq_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    feature_cols = [col for col in df.columns if col not in ("time",)]
    X_list = []
    y_list = []
    
    for i in range(len(df) - seq_length):
        X_list.append(df[feature_cols].iloc[i : i + seq_length].values)
        y_list.append(labels.iloc[i + seq_length])
    
    return np.array(X_list), np.array(y_list)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """Train the LSTM model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = 100.0 * correct / total
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss / len(train_loader):.4f} "
            f"Val Loss: {val_loss / len(val_loader):.4f} "
            f"Val Accuracy: {accuracy:.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FCPO AI model")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=20, help="Sequence length")
    parser.add_argument("--output", type=str, default="./models/fcpo_model.pt", help="Output path")
    args = parser.parse_args()
    
    # Load and prepare data
    print("Loading data...")
    df = load_data(Path(args.data))
    df = prepare_features(df)
    labels = create_labels(df)
    
    # Create sequences
    print("Creating sequences...")
    X, y = prepare_sequences(df, labels, seq_length=args.seq_length)
    
    # Normalize features
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[-1]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=3)
    model.to(device)
    
    print(f"Training on {device}...")
    train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
