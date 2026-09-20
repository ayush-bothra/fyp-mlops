# Edge VLM Active Learning & Telemetry Control

## Project Goal

This project investigates whether an edge AI system can make
resource-aware decisions about **which incoming samples should be
retained for model adaptation and when adaptation should occur** —
without access to ground-truth labels at selection time, and without
risking thermal damage to the device during retraining.

The system runs two independent pipelines that meet at retrain time:

> A **coreset buffer** watches an incoming visual stream and retains a
> label-free, diverse working set using SigLIP embeddings alone. In
> parallel, a **Bayesian GRU** forecasts near-future device telemetry,
> and a **BMRC safety gate** uses that forecast's uncertainty to decide
> whether retraining right now is thermally safe. Only when the gate
> says yes does a QLoRA burst run on the buffer's current contents.

---

## Core Architecture (as implemented and tested)

```text
                 CORe50 Visual Stream (non-IID order)
                         |
                         v
       SmolVLM-256M-Instruct's internal SigLIP encoder (93M params)
                         |
                         v
              L2-normalized image embedding
                         |
                         v
                  +---------------+
                  | CoresetBuffer | -- admit: farthest from buffer contents
                  | (label-free)  | -- evict: most-redundant pair, not oldest
                  +-------+-------+
                          |
                          v
                 Retained buffer (capacity N, no labels used)
                          |
                          |            Real GPU/CPU Telemetry
                          |                    |
                          |                    v
                          |          TelemetryGRU + MC-Dropout
                          |          (forecast mean + uncertainty)
                          |                    |
                          |                    v
                          |            BMRC Z-test gate
                          |          (temperature, optionally energy)
                          |                    |
                          +--------> [ WAIT  |  QLORA ] <--------------+
                                          |
                                          v (if QLORA)
                              QLoRA burst on sampled
                              buffer contents, real GPU training
```

Labels are used only to **score** buffer quality after the fact
(category coverage, entropy) — never to make an admission, eviction,
or retrain decision.

---

## Project Structure

```text
fyp-mlops/
├── data/
│   └── raw/core50/            # CORe50 benchmark visual stream (gitignored)
├── models/                    # GRU checkpoint & scaler (gitignored)
│   ├── gru_telemetry.pt
│   └── telemetry_scaler.json
├── configs/
│   └── feasibility.yaml       # model/dataset/qlora/training config
├── scripts/
│   ├── feasibility/
│   │   ├── f2_qlora_training.py   # F2: QLoRA feasibility, PASS/FAIL/OOM
│   │   ├── f2_vram_sweep.py       # minimum viable vram_fraction sweep
│   │   └── thermal_probe.py       # repeated load/idle burst telemetry probe
│   ├── buffer_non_iid.py          # buffer selection eval vs reservoir baseline
│   ├── combine_telemetry_episodes.py  # merges telemetry CSVs for GRU training
│   ├── check_uncertainty.py       # MC-Dropout uncertainty sanity check
│   ├── full_pipeline_prototype.py # end-to-end integration run
│   ├── analyze_telemetry.py       # EDA on telemetry data
│   └── download_datasets.py
├── src/
│   ├── vlm/                   # SmolVLM loading, quantization, training loop
│   │   ├── loader.py
│   │   ├── quantization.py
│   │   ├── training.py
│   │   └── scorer.py          # standalone usefulness/entropy scorer (not currently wired into buffer selection)
│   ├── buffer/
│   │   └── selector.py        # CoresetBuffer: label-free admit/evict
│   ├── gru/
│   │   ├── dataset.py         # TelemetryScaler, episode-safe sequence generator
│   │   ├── model.py           # TelemetryGRU + predict_with_uncertainty (MC-Dropout)
│   │   └── train.py
│   ├── telemetry/
│   │   └── logger.py          # TelemetryLogger: real GPU/CPU telemetry capture
│   ├── safety/
│   │   └── bmrc.py            # Z-test decision rule + multi-factor combine_decisions
│   └── data/
│       ├── core50.py          # scan_core50, session/object/frame metadata
│       └── splits.py          # deterministic, session-based (non-leaky) splits
├── tests/
│   ├── test_buffer_selector.py
│   ├── test_telemetry_logger.py
│   ├── test_bmrc_decision.py
│   └── test_baselines.py      # persistence/linear-trend forecast baselines
├── docs/
│   └── active_learning_telemetry_summary.qmd  # full feasibility + integration report
└── README.md
```

`src/fusion/` and `src/eval/` from an earlier plan were removed — empty
placeholders never implemented, and their intended role is now covered
by `src/safety/bmrc.py` (retrain decision) and `scripts/full_pipeline_prototype.py`
(stream simulation), respectively.

---

## Prerequisites & Installation

```bash
git clone https://github.com/ayush-bothra/fyp-mlops.git
cd fyp-mlops
pip install -r requirements.txt
```

`nvidia-ml-py` (imports as `pynvml`) is required for `TelemetryLogger`
and is not in `requirements.txt` yet — install separately:
```bash
pip install nvidia-ml-py
```

---

## Quickstart Guide

### 1. Run the F2 QLoRA feasibility test
```bash
python -m scripts.feasibility.f2_qlora_training --config configs/feasibility.yaml
```

### 2. Generate real device telemetry
```bash
python -m scripts.feasibility.thermal_probe --num-bursts 6 --idle-seconds 45 --output results/telemetry/probe.csv
```

### 3. Combine telemetry episodes and train the GRU
```bash
python scripts/combine_telemetry_episodes.py
python -m src.gru.train --epochs 25
```

### 4. Evaluate buffer selection on a non-IID CORe50 stream
```bash
python -m scripts.buffer_non_iid --dataset data/raw/core50/core50_128x128
```

### 5. Run the full integrated pipeline
```bash
python -m scripts.full_pipeline_prototype --config configs/feasibility.yaml
```

### 6. Run the correctness test suite
```bash
python tests/test_buffer_selector.py
python tests/test_telemetry_logger.py
python tests/test_bmrc_decision.py
```

---

## Implementation Status

### Completed and Validated
- [x] **QLoRA feasibility on RTX 3050 (6GB)** — `vram_fraction=0.2` confirmed as the practical default (0.1 OOMs, 0.15 is the minimum viable).
- [x] **Real telemetry infrastructure (`src/telemetry/logger.py`)** — captures live GPU/CPU telemetry matching the GRU's expected schema, used both standalone and wrapped around real training calls.
- [x] **Thermal risk characterization** — steady-state training shows negligible risk; repeated load/idle bursts show error at retrain-onset is 3–8× worse than the whole-stream average, justifying a transient-aware safety layer over a fixed threshold.
- [x] **GRU forecasting (`src/gru/`)** — retrained on real device telemetry (not the original public IoT-RL dataset); R² of 0.51 (cpu_usage), 0.57 (temperature_C), 0.74 (energy_consumed_mJ) on a held-out episode. `battery_level` dropped as a target after repeated near-constant readings collapsed its R².
- [x] **MC-Dropout uncertainty (`predict_with_uncertainty`)** — confirmed non-collapsed, input-dependent variance.
- [x] **BMRC safety gate (`src/safety/bmrc.py`)** — Z-test decision rule verified against hand-computed values; multi-factor `combine_decisions` added (AND-logic across independently tested factors, avoiding an unjustified independence assumption between temperature and energy).
- [x] **Buffer selection (`src/buffer/selector.py`)** — original k-means + SQIR design tested and found to fail on real non-IID CORe50 data (diagnosed as centroid collapse under anisotropic embeddings, confirmed via instrumentation); replaced with a coreset (farthest-point) selector that matches reservoir sampling's full category coverage while remaining label-free.
- [x] **End-to-end integration (`scripts/full_pipeline_prototype.py`)** — buffer, GRU forecast, and BMRC gate run together against a live stream with one continuous telemetry log; produced both WAIT and QLoRA decisions correctly, with a gated retrain burst showing real loss reduction.

### Open / Next Steps
- [ ] **Energy-budget gating** — implemented in `combine_decisions` but never exercised with a real limit; development hardware runs on AC power throughout, so no energy threshold has been derived or validated.
- [ ] **More telemetry episodes** — GRU trained on 6 script-generated episodes; more variety (different burst counts, idle lengths, real usage patterns) would strengthen generalization claims.
- [ ] **Cross-hardware validation** — all thermal results are from one RTX 3050 laptop; a power-envelope proxy for Jetson Orin/Xavier-class hardware was attempted but blocked by vendor firmware locking `nvidia-smi -pl`.
- [ ] **`src/vlm/scorer.py`'s usefulness score is not currently part of the buffer's selection signal** — worth deciding whether to integrate it alongside SigLIP-embedding diversity, or retire it if the coreset approach fully supersedes the original usefulness-score concept.