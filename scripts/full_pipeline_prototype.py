"""
Full pipeline prototype: SigLIP + CoresetBuffer -> GRU forecast + BMRC
decision -> gated QLoRA retraining, all running against a live CORe50
non-IID stream with one continuous real telemetry log underneath it.

This is an integration run, not a controlled experiment -- it exists to
show the pieces actually work together end to end, not to produce a
publishable metric on its own.

Usage:
    python -m scripts.full_pipeline_prototype --config configs/feasibility.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from scripts.buffer_non_iid import build_non_iid_stream
from scripts.feasibility.f2_qlora_training import (
    coerce_numeric_config_fields,
    load_config,
    resolve_compute_dtype,
)
from src.buffer.selector import CoresetBuffer
from src.data.core50 import scan_core50
from src.gru.dataset import INPUT_FEATURES, TARGET_FEATURES, TelemetryScaler
from src.gru.model import TelemetryGRU
from src.safety.bmrc import combine_decisions, decide_retrain_action
from src.telemetry.logger import TelemetryLogger
from src.vlm.loader import load_quantized_vlm_for_training
from src.vlm.quantization import apply_vram_cap
from src.vlm.training import run_training_loop, stream_supervised_batches

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
EPISODE_NAME = "full_pipeline_prototype"


def load_cpu_vision_encoder(model_id):
    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return model, processor


def extract_embedding(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    pixel_values = pixel_values.view(-1, 3, pixel_values.shape[-2], pixel_values.shape[-1])
    with torch.no_grad():
        vision_outputs = model.model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.last_hidden_state.mean(dim=1)
    return pooled.squeeze(0).numpy()


def read_recent_telemetry(csv_path, episode_name, seq_len):
    df = pd.read_csv(csv_path)
    episode_df = df[df["episode"] == episode_name]
    if len(episode_df) < seq_len:
        return None
    return episode_df.tail(seq_len)


def forecast_feature(gru_model, scaler, recent_df, feature_name):
    scaled = scaler.transform(recent_df, INPUT_FEATURES)
    x = torch.from_numpy(scaled).float().unsqueeze(0)
    mean, std = gru_model.predict_with_uncertainty(x, mc_samples=20)

    feature_index = TARGET_FEATURES.index(feature_name)
    mean_scaled = mean[0, 0, feature_index].item()
    std_scaled = std[0, 0, feature_index].item()

    mean_physical = mean_scaled * scaler.stds[feature_name] + scaler.means[feature_name]
    std_physical = std_scaled * scaler.stds[feature_name]
    return mean_physical, std_physical


def run_prototype(config, args):
    dataset_cfg = config["dataset"]
    qlora_cfg = config["qlora"]
    training_cfg = coerce_numeric_config_fields(config["training"])

    apply_vram_cap(fraction=training_cfg["vram_fraction"], device_index=training_cfg["device_index"])

    samples = scan_core50(dataset_cfg["path"])
    stream = build_non_iid_stream(samples, session_id=args.session, frames_per_object=args.frames_per_object)
    print(f"stream length: {len(stream)} samples")

    cpu_model, cpu_processor = load_cpu_vision_encoder(MODEL_ID)

    compute_dtype = resolve_compute_dtype(qlora_cfg["compute_dtype"])
    gpu_model, gpu_processor = load_quantized_vlm_for_training(
        model_id=config["model"]["id"],
        lora_r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["alpha"],
        lora_dropout=qlora_cfg["dropout"],
        compute_dtype=compute_dtype,
        device_index=training_cfg["device_index"],
    )
    device = torch.device(f"cuda:{training_cfg['device_index']}")
    trainable_params = [p for p in gpu_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=training_cfg["learning_rate"])

    gru_model = TelemetryGRU.load_checkpoint(args.gru_checkpoint)
    scaler = TelemetryScaler.load(args.scaler_path)

    buffer = CoresetBuffer(capacity=args.capacity, window_size=100)

    logger = TelemetryLogger(output_path=args.telemetry_output, episode=EPISODE_NAME, interval_seconds=1.0)
    logger.start()

    admissions_since_check = 0
    retrain_log = []

    try:
        for index, sample in enumerate(stream):
            embedding = extract_embedding(cpu_model, cpu_processor, sample.path)
            admitted, _ = buffer.evaluate_and_add(embedding, sample)
            if admitted:
                admissions_since_check += 1

            if admissions_since_check >= args.retrain_check_interval:
                admissions_since_check = 0
                recent_telemetry = read_recent_telemetry(args.telemetry_output, EPISODE_NAME, args.seq_len)

                if recent_telemetry is None:
                    retrain_log.append({"stream_index": index, "action": "SKIPPED_INSUFFICIENT_TELEMETRY"})
                    continue

                mean_temp, std_temp = forecast_feature(gru_model, scaler, recent_telemetry, "temperature_C")
                temp_decision = decide_retrain_action(
                    mean_temperature=mean_temp, std_temperature=std_temp, temperature_limit=args.temperature_limit
                )
                decisions = [temp_decision]
                log_entry = {
                    "stream_index": index,
                    "forecast_mean_C": mean_temp,
                    "forecast_std_C": std_temp,
                    "temp_breach_probability": temp_decision.breach_probability,
                }

                if args.energy_limit_mj is not None:
                    mean_energy, std_energy = forecast_feature(gru_model, scaler, recent_telemetry, "energy_consumed_mJ")
                    energy_decision = decide_retrain_action(
                        mean_temperature=mean_energy, std_temperature=std_energy, temperature_limit=args.energy_limit_mj
                    )
                    decisions.append(energy_decision)
                    log_entry["forecast_mean_energy_mJ"] = mean_energy
                    log_entry["forecast_std_energy_mJ"] = std_energy
                    log_entry["energy_breach_probability"] = energy_decision.breach_probability

                action, limiting_decision = combine_decisions(decisions)
                log_entry["action"] = action
                log_entry["limiting_breach_probability"] = limiting_decision.breach_probability

                print(f"[{index}/{len(stream)}] temp={mean_temp:.1f}C±{std_temp:.2f}  decision={action}")
                retrain_log.append(log_entry)

                if action == "QLORA":
                    logger.set_pipeline_state(queue_size=len(buffer.items), latency_ms=0.0)
                    burst_size = min(args.burst_sample_limit, len(buffer.items))
                    burst_samples = random.sample(buffer.items, burst_size)
                    train_batches = lambda: stream_supervised_batches(
                        gpu_processor, burst_samples, device, shuffle=True, seed=dataset_cfg["seed"] + index
                    )
                    train_result = run_training_loop(gpu_model, optimizer, train_batches, epochs=1)
                    retrain_log[-1]["burst_first_step_loss"] = train_result.step_losses[0]
                    retrain_log[-1]["burst_last_step_loss"] = train_result.step_losses[-1]
                    retrain_log[-1]["burst_sample_count"] = burst_size

            if (index + 1) % 100 == 0:
                print(f"processed {index + 1}/{len(stream)}")
    finally:
        logger.stop()

    final_categories = Counter(sample.category_name for sample in buffer.items)
    summary = {
        "stream_length": len(stream),
        "buffer_capacity": args.capacity,
        "final_buffer_size": len(buffer.items),
        "final_category_counts": dict(final_categories),
        "distinct_categories": len(final_categories),
        "retrain_decisions": retrain_log,
        "qlora_bursts_run": sum(1 for entry in retrain_log if entry.get("action") == "QLORA"),
        "wait_decisions": sum(1 for entry in retrain_log if entry.get("action") == "WAIT"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)

    print(f"\ndone -- summary saved to {output_path}")
    print(f"final buffer: {dict(final_categories)}")
    print(f"QLoRA bursts: {summary['qlora_bursts_run']}, WAIT decisions: {summary['wait_decisions']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--frames-per-object", type=int, default=24)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--retrain-check-interval", type=int, default=40)
    parser.add_argument("--burst-sample-limit", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--gru-checkpoint", type=str, default="models/gru_telemetry.pt")
    parser.add_argument("--scaler-path", type=str, default="models/telemetry_scaler.json")
    parser.add_argument("--telemetry-output", type=str, default="results/telemetry/full_pipeline_prototype.csv")
    parser.add_argument("--output", type=str, default="results/prototype/full_pipeline_run.json")
    parser.add_argument("--temperature-limit", type=float, default=80.0)
    parser.add_argument("--energy-limit-mj", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_prototype(config, args)


if __name__ == "__main__":
    main()
