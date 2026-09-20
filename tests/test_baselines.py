import csv
import numpy as np


def load_temperatures_and_bursts(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    temperatures = np.array([float(row["temperature_C"]) for row in rows])
    burst_ids = [row["queue_size"] for row in rows]
    return temperatures, burst_ids

def burst_start_indices(burst_ids):
    starts = [0]
    for i in range(1, len(burst_ids)):
        if burst_ids[i] != burst_ids[i - 1]:
            starts.append(i)
    return starts


def errors_at_burst_starts(temperatures, starts, horizon):
    errors = []
    for start in starts:
        if start + horizon < len(temperatures):
            errors.append(abs(temperatures[start] - temperatures[start + horizon]))
    return errors

def persistence_baseline_error(temperatures, horizon):
    errors = []
    for i in range(len(temperatures) - horizon):
        predicted = temperatures[i]
        actual = temperatures[i + horizon]
        errors.append(abs(predicted - actual))
    return np.mean(errors), np.max(errors)


def linear_trend_baseline_error(temperatures, horizon, window=10):
    errors = []
    for i in range(window, len(temperatures) - horizon):
        recent = temperatures[i - window:i]
        slope = (recent[-1] - recent[0]) / window
        predicted = temperatures[i] + slope * horizon
        actual = temperatures[i + horizon]
        errors.append(abs(predicted - actual))
    return np.mean(errors), np.max(errors)


def main():
    temperatures, burst_ids = load_temperatures_and_bursts("results/telemetry/thermal_transient_probe.csv")
    starts = burst_start_indices(burst_ids)
    print(f"detected {len(starts)} burst starts")

    for horizon in [5, 10, 30, 60, 120]:
        persistence_mean, persistence_max = persistence_baseline_error(temperatures, horizon)
        trend_mean, trend_max = linear_trend_baseline_error(temperatures, horizon)

        burst_errors = errors_at_burst_starts(temperatures, starts, horizon)
        burst_mean = np.mean(burst_errors) if burst_errors else float("nan")
        burst_max = np.max(burst_errors) if burst_errors else float("nan")

        print(
            f"horizon={horizon}s  whole-stream persistence mean/max={persistence_mean:.2f}/{persistence_max:.2f}  "
            f"whole-stream trend mean/max={trend_mean:.2f}/{trend_max:.2f}  "
            f"AT BURST STARTS ONLY persistence mean/max={burst_mean:.2f}/{burst_max:.2f}"
        )

if __name__ == "__main__":
    main()
