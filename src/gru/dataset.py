"""
Telemetry Dataset Module for GRU Time-Series Training.

Extracts sliding-window sequences of physical edge metrics grouped strictly by episode,
preventing cross-episode boundary leakage.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

INPUT_FEATURES: list[str] = [
    "battery_level",
    "cpu_usage",
    "memory_usage",
    "temperature_C",
    "energy_consumed_mJ",
    "queue_size",
    "latency_ms",
]

TARGET_FEATURES: list[str] = [
    "battery_level",
    "cpu_usage",
    "temperature_C",
    "energy_consumed_mJ",
]


@dataclass
class TelemetryScaler:
    """Standardizes input and target features using mean and standard deviation."""

    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str]) -> "TelemetryScaler":
        means = {feat: float(df[feat].mean()) for feat in features}
        stds = {feat: float(df[feat].std() + 1e-8) for feat in features}
        return cls(means=means, stds=stds)

    def transform(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        scaled = np.zeros((len(df), len(features)), dtype=np.float32)
        for idx, feat in enumerate(features):
            scaled[:, idx] = (df[feat].values - self.means[feat]) / self.stds[feat]
        return scaled

    def inverse_transform_targets(
        self, array: np.ndarray, target_features: list[str]
    ) -> np.ndarray:
        """Inverses target predictions back to original physical units."""
        orig = np.zeros_like(array)
        for idx, feat in enumerate(target_features):
            orig[..., idx] = (array[..., idx] * self.stds[feat]) + self.means[feat]
        return orig

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"means": self.means, "stds": self.stds}, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "TelemetryScaler":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(means=data["means"], stds=data["stds"])


class TelemetryDataset(Dataset):
    """PyTorch Dataset wrapping sliding window historical sequences and target futures."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_sequences_from_episodes(
    df: pd.DataFrame,
    scaler: TelemetryScaler,
    input_features: list[str] = INPUT_FEATURES,
    target_features: list[str] = TARGET_FEATURES,
    seq_len: int = 6,
    forecast_horizon: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates (X, y) sliding window sequences strictly within individual episodes.
    """
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    target_indices = [input_features.index(col) for col in target_features]

    for _, ep_df in df.groupby("episode"):
        if len(ep_df) < (seq_len + forecast_horizon):
            continue

        scaled_matrix = scaler.transform(ep_df, input_features)

        for i in range(len(scaled_matrix) - seq_len - forecast_horizon + 1):
            x_seq = scaled_matrix[i : i + seq_len]
            y_seq = scaled_matrix[
                i + seq_len : i + seq_len + forecast_horizon, target_indices
            ]
            x_list.append(x_seq)
            y_list.append(y_seq)

    if not x_list:
        return np.empty((0, seq_len, len(input_features)), dtype=np.float32), np.empty(
            (0, forecast_horizon, len(target_features)), dtype=np.float32
        )

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def load_and_preprocess_data(
    csv_path: str | Path = "data/raw/iot_telemetry/FINAL IoT RL dataset 2026.csv",
    seq_len: int = 6,
    forecast_horizon: int = 3,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader, TelemetryScaler]:
    """
    Splits episodes into train/val/test sets and returns DataLoaders with the fitted scaler.
    """
    df = pd.read_csv(csv_path)

    # Unique episodes for clean group split without temporal leakage
    unique_episodes = df["episode"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_episodes)

    n_total = len(unique_episodes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_eps = set(unique_episodes[:n_train])
    val_eps = set(unique_episodes[n_train : n_train + n_val])
    test_eps = set(unique_episodes[n_train + n_val :])

    train_df = df[df["episode"].isin(train_eps)].copy()
    val_df = df[df["episode"].isin(val_eps)].copy()
    test_df = df[df["episode"].isin(test_eps)].copy()

    # Fit scaler on train split only
    scaler = TelemetryScaler.fit(train_df, INPUT_FEATURES)

    x_train, y_train = create_sequences_from_episodes(
        train_df, scaler, seq_len=seq_len, forecast_horizon=forecast_horizon
    )
    x_val, y_val = create_sequences_from_episodes(
        val_df, scaler, seq_len=seq_len, forecast_horizon=forecast_horizon
    )
    x_test, y_test = create_sequences_from_episodes(
        test_df, scaler, seq_len=seq_len, forecast_horizon=forecast_horizon
    )

    print(
        f"Generated Sequences -> Train: {len(x_train):,}, Val: {len(x_val):,}, Test: {len(x_test):,}"
    )

    train_loader = DataLoader(
        TelemetryDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TelemetryDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TelemetryDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler
