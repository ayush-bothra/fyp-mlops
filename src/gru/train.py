"""
Combined Training and Testing Pipeline for GRU Telemetry Forecaster.

Trains the model with validation checkpoints and evaluates on held-out test episodes.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import INPUT_FEATURES, TARGET_FEATURES, load_and_preprocess_data, TelemetryScaler
from .model import TelemetryGRU


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Calculates MAE, RMSE, and R2 per target feature."""
    metrics: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(TARGET_FEATURES):
        true_feat = y_true[..., idx].flatten()
        pred_feat = y_pred[..., idx].flatten()

        mae = float(np.mean(np.abs(true_feat - pred_feat)))
        rmse = float(np.sqrt(np.mean((true_feat - pred_feat) ** 2)))
        ss_tot = np.sum((true_feat - np.mean(true_feat)) ** 2)
        ss_res = np.sum((true_feat - pred_feat) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        metrics[feature] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
    return metrics


def train_and_evaluate(
    csv_path: str | Path = "data/raw/iot_telemetry/FINAL IoT RL dataset 2026.csv",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    seq_len: int = 6,
    forecast_horizon: int = 3,
    save_dir: str | Path = "models",
    device: str | None = None,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if device is not None:
        target_device = torch.device(device)
    elif torch.cuda.is_available():
        target_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        target_device = torch.device("mps")
    else:
        target_device = torch.device("cpu")

    print(f"Device: {target_device}")
    print(f"Loading and preprocessing data from {csv_path}...")

    train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(
        csv_path=csv_path,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
    )

    # Save scaler for future inference
    scaler_path = save_path / "telemetry_scaler.json"
    scaler.save(scaler_path)
    print(f"Saved scaler parameters to {scaler_path}")

    model = TelemetryGRU(
        input_dim=len(INPUT_FEATURES),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=len(TARGET_FEATURES),
        forecast_horizon=forecast_horizon,
    ).to(target_device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    model_ckpt_path = save_path / "gru_telemetry.pt"

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(target_device)
            y_batch = y_batch.to(target_device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x_batch)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(target_device)
                y_batch = y_batch.to(target_device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(x_batch)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(model_ckpt_path)
            saved_indicator = " [*Best Model Saved]"
        else:
            saved_indicator = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}{saved_indicator}")

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.5f}")

    # --- Test Evaluation Section ---
    print("\n--- Evaluating on Held-Out Test Episodes ---")
    best_model = TelemetryGRU.load_checkpoint(model_ckpt_path, device=target_device)

    all_preds_list: list[np.ndarray] = []
    all_trues_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(target_device)
            preds = best_model(x_batch)
            all_preds_list.append(preds.cpu().numpy())
            all_trues_list.append(y_batch.numpy())

    y_pred_scaled = np.concatenate(all_preds_list, axis=0)
    y_true_scaled = np.concatenate(all_trues_list, axis=0)

    # Invert back to physical units
    y_pred_orig = scaler.inverse_transform_targets(y_pred_scaled, TARGET_FEATURES)
    y_true_orig = scaler.inverse_transform_targets(y_true_scaled, TARGET_FEATURES)

    metrics = calculate_metrics(y_true_orig, y_pred_orig)

    print("\n" + "=" * 55)
    print(f"{'Target Feature':<22} | {'MAE':<8} | {'RMSE':<8} | {'R2 Score':<8}")
    print("=" * 55)
    for feat, m in metrics.items():
        print(f"{feat:<22} | {m['MAE']:<8} | {m['RMSE']:<8} | {m['R2']:<8}")
    print("=" * 55)
    print(f"Artifacts saved in '{save_path.resolve()}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test GRU telemetry forecaster.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GRU hidden units")
    parser.add_argument("--num-layers", type=int, default=2, help="GRU layers")
    parser.add_argument("--seq-len", type=int, default=6, help="Lookback sequence steps")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon steps")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model checkpoint")

    args = parser.parse_args()

    train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        forecast_horizon=args.horizon,
        save_dir=args.save_dir,
    )
