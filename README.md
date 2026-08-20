# Intelligent Sample Selection for Edge Retraining

## Project Goal

This project investigates whether an edge AI system can make resource-aware decisions about **which incoming samples should be retained for model adaptation and when adaptation should occur**.

The central idea is:

> The **VLM** evaluates the incoming data and estimates its learning value / novelty, while the **GRU** predicts the future physical/resource state of the edge device. A **Decision Engine** combines both signals to determine whether the system should **DISCARD**, **RETAIN/WAIT**, or **ADAPT**.

---

## Core Architecture

```text
                 CORe50 Visual Stream
                         |
                         v
               SmolVLM-256M-Instruct (INT8)
                         |
                         v
              Sample usefulness signal (0.0 - 1.0)
                         |
                         |
                         v
                    +---------+
                    |         |
                    | Decision| ===> [ DISCARD | RETAIN / WAIT | ADAPT ]
                    | Engine  |
                    |         |
                    +----+----+
                         ^
                         |
                 GRU Forecast (Horizon = 3)
                         ^
                         |
              Physical Edge Telemetry (History = 6)
                         |
                 IoT Telemetry Dataset
```

### The Decision Engine Combines:
* **Input-side Information (VLM):** Is the incoming sample novel, uncertain, or informative enough to warrant retraining?
* **Device-side Information (GRU):** What is the predicted near-future battery, CPU, temperature, and energy state? Will retraining cause thermal throttling or power exhaustion?

---

## Project Structure

```text
fyp-mlops/
├── data/
│   ├── raw/                  # Raw datasets (gitignored)
│   │   ├── iot_telemetry/    # IoT-RL Telemetry dataset
│   │   └── core50/           # CORe50 benchmark visual stream
│   └── processed/            # Processed features & splits
├── models/                   # Model checkpoints & scalers (gitignored)
├── scripts/
│   ├── download_datasets.py  # Automated downloader for IoT & CORe50 datasets
│   └── analyze_telemetry.py  # EDA & time-series correlation analysis
├── src/
│   ├── vlm/                  # VLM usefulness scorer (SmolVLM in INT8)
│   │   ├── __init__.py
│   │   └── scorer.py
│   ├── gru/                  # GRU telemetry time-series forecaster
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   ├── fusion/               # Decision Engine combining VLM + GRU signals
│   └── eval/                 # Stream simulation & benchmark evaluation
├── tests/                    # Unit tests
└── README.md
```

---

## Prerequisites & Installation

1. **Clone repository and set up environment:**
   ```bash
   git clone https://github.com/ayush-bothra/fyp-mlops.git
   cd fyp-mlops
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle credentials (for IoT dataset download):**
   ```bash
   python3 -c "import kagglehub; kagglehub.login()"
   ```

---

## Quickstart Guide

### 1. Download Datasets
```bash
# Download both IoT Telemetry and CORe50 datasets
python scripts/download_datasets.py

# Or download individually
python scripts/download_datasets.py --dataset iot
python scripts/download_datasets.py --dataset core50
```

### 2. Run VLM Sample Scoring (SmolVLM-256M in INT8)
```bash
# Test VLM inference on a synthetic sample
python -m src.vlm.scorer

# Test on a specific image with text description
python -m src.vlm.scorer --image path/to/image.jpg --generate-text
```

### 3. Train and Evaluate the GRU Telemetry Forecaster
```bash
# Trains the 2-layer GRU and prints test metrics (MAE, RMSE, R^2)
python -m src.gru.train --epochs 25 --seq-len 6 --horizon 3
```

---

## Implementation Status & Roadmap

### Completed Milestones
- [x] **VLM Scorer Module (`src/vlm/`):** Loaded `SmolVLM-256M-Instruct` with INT8 quantization (~260 MB VRAM), computing predictive entropy and top-1 confidence.
- [x] **Data Ingestion & EDA (`scripts/`):** Automated download pipeline and verified 30s sampling intervals with zero missing values and high temporal autocorrelation.
- [x] **GRU Preprocessing (`src/gru/dataset.py`):** Episode-safe sliding window generator ($T_{\text{in}}=6, T_{\text{out}}=3$) with 7 input features and 4 targets.
- [x] **GRU Architecture (`src/gru/model.py`):** Lightweight PyTorch GRU (~41K params, 165 KB footprint, <0.5 ms latency).
- [x] **Combined Training & Testing (`src/gru/train.py`):** Validation checkpointing and automatic test evaluation.

### Next Steps
- [ ] **Fusion Decision Engine (`src/fusion/`):** Implement the dual-signal rule/threshold logic to map `(VLM_Score, GRU_Forecast)` to `DISCARD`, `RETAIN / WAIT`, or `ADAPT`.
- [ ] **Stream Evaluation Benchmark (`src/eval/`):** Run full continual learning simulation on CORe50 frames paired with IoT telemetry to measure compute/energy savings vs naive retraining.
