"""
PyTorch GRU Model Architecture for Edge Telemetry Forecasting.
"""

from pathlib import Path

import torch
from torch import nn


class TelemetryGRU(nn.Module):
    """
    Recurrent neural network that forecasts near-future edge device telemetry
    (battery, cpu, temperature, energy consumption) from recent historical readings.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 4,
        forecast_horizon: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim * forecast_horizon),
        )

    def predict_with_uncertainty(self, x: torch.Tensor, mc_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train()

        predictions = []
        with torch.no_grad():
            for _ in range(mc_samples):
                predictions.append(self(x))

        self.train(was_training)

        stacked = torch.stack(predictions, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Forecast tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        out, _ = self.gru(x)
        # Use final hidden state from the last sequence step
        last_step = out[:, -1, :]
        preds = self.head(last_step)
        return preds.view(-1, self.forecast_horizon, self.output_dim)

    def save_checkpoint(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "output_dim": self.output_dim,
                    "forecast_horizon": self.forecast_horizon,
                },
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, filepath: str | Path, device: torch.device | str = "cpu"
    ) -> "TelemetryGRU":
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
