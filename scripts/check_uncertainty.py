from src.gru.dataset import load_and_preprocess_data
from src.gru.model import TelemetryGRU
import numpy as np 

train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(
    csv_path="results/telemetry/combined_episodes.csv",
    train_ratio=0.6,
    val_ratio=0.2,
)

model = TelemetryGRU.load_checkpoint("models/gru_telemetry.pt")

x_batch, y_batch = next(iter(test_loader))
mean, std = model.predict_with_uncertainty(x_batch, mc_samples=20)

print("std shape:", std.shape)
print("std min/mean/max:", std.min().item(), std.mean().item(), std.max().item())
print(std)

target_features = ["battery_level", "cpu_usage", "temperature_C", "energy_consumed_mJ"]
physical_std = std.numpy() * np.array([scaler.stds[f] for f in target_features])
print(physical_std[:, 0, :])