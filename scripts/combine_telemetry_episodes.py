import pandas as pd

combined = pd.concat([
    pd.read_csv("results/telemetry/thermal_transient_probe.csv"),
    pd.read_csv("results/telemetry/probe_run2.csv"),
    pd.read_csv("results/telemetry/probe_idle20.csv"),
    pd.read_csv("results/telemetry/probe_idle60.csv"),
    pd.read_csv("results/telemetry/probe_idle90.csv"),
])
combined.to_csv("results/telemetry/combined_episodes_without_f2.csv", index=False)
print(combined["episode"].value_counts())