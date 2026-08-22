"""
Exploratory Data Analysis (EDA) for IoT Telemetry Dataset.
Inspects time-series structure, feature distributions, and predictability for GRU design.
"""

from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/raw/iot_telemetry/FINAL IoT RL dataset 2026.csv")


def run_analysis() -> None:
    df = pd.read_csv(CSV_PATH)

    print("=" * 60)
    print("1. Dataset Overview and Shape")
    print("=" * 60)
    print(f"Total Rows:    {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"All Columns:   {df.columns.tolist()}\n")

    print("=" * 60)
    print("2. TIME-SERIES & EPISODE STRUCTURE")
    print("=" * 60)
    num_episodes = df["episode"].nunique()
    steps_per_ep = df.groupby("episode")["step"].count()
    print(f"Number of Episodes:       {num_episodes}")
    print(
        f"Avg Steps per Episode:    {steps_per_ep.mean():.1f} (min={steps_per_ep.min()}, max={steps_per_ep.max()})"
    )

    df["dt"] = pd.to_datetime(df["timestamp"])
    time_deltas = df.groupby("episode")["dt"].diff().dt.total_seconds().dropna()
    print(f"Dominant Timestep Delta: {time_deltas.mode().iloc[0]:.1f} seconds")
    print()

    print("=" * 60)
    print("3. MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        print("No missing / NaN values found!")
    else:
        print(missing_cols)
    print()

    print("=" * 60)
    print("4. KEY TELEMETRY SUMMARY STATISTICS")
    print("=" * 60)
    key_features = [
        "battery_level",
        "cpu_usage",
        "memory_usage",
        "temperature_C",
        "energy_consumed_mJ",
        "energy_harvested_mJ",
        "net_energy_mJ",
        "queue_size",
        "latency_ms",
    ]
    existing_features = [col for col in key_features if col in df.columns]
    stats_df = (
        df[existing_features]
        .describe()
        .T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    print(stats_df.round(4).to_string())
    print()

    print("=" * 60)
    print("5. TEMPORAL AUTOCORRELATION (Lag-1 to Lag-5)")
    print("=" * 60)
    print("Checks if sequential steps have strong predictable memory:")
    for col in ["battery_level", "cpu_usage", "temperature_C", "net_energy_mJ"]:
        if col in df.columns:
            lags = [df[col].autocorr(lag=i) for i in [1, 2, 3, 5]]
            print(
                f"{col:<20} | Lag 1: {lags[0]:.3f} | Lag 2: {lags[1]:.3f} | Lag 3: {lags[2]:.3f} | Lag 5: {lags[3]:.3f}"
            )
    print()

    print("=" * 60)
    print("6. ACTION DISTRIBUTION")
    print("=" * 60)
    if "action_name" in df.columns:
        print(
            df["action_name"].value_counts(normalize=True).mul(100).round(2).to_string()
        )
    print("=" * 60)


if __name__ == "__main__":
    run_analysis()
