"""
F2 VRAM feasibility sweep.

Runs the F2 QLoRA experiment at several vram_fraction values, each as a
SEPARATE subprocess (not sequential in-process calls), because CUDA's
per-process memory fraction and the caching allocator's fragmentation
state don't cleanly reset within one Python process. A fresh process per
fraction is the only way to get a clean, independently-trustworthy
measurement at each point -- otherwise a later data point could look
better or worse than it really is just because of leftover allocator
state from the previous run.

This does NOT keep lowering/raising settings until something "passes" in
the sense the spec warns against -- it deliberately runs the FULL set of
fractions regardless of individual outcomes, to characterize the whole
feasibility boundary, and reports all of them.

Usage:
    python -m scripts.feasibility.f2_vram_sweep --config configs/feasibility.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_FRACTIONS = [0.1, 0.2, 0.3, 0.4]


def run_single_fraction(config_path: str, fraction: float, results_dir: Path) -> dict:
    output_path = results_dir / f"qlora_training_frac{fraction:.2f}.json"
    cmd = [
        sys.executable,
        "-m",
        "scripts.feasibility.f2_qlora_training",
        "--config",
        config_path,
        "--vram-fraction",
        str(fraction),
        "--output",
        str(output_path),
    ]

    print(f"\n{'=' * 70}")
    print(f"Running fraction={fraction:.2f} ...")
    print(f"{'=' * 70}")

    process = subprocess.run(cmd, capture_output=True, text=True)

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as file:
            result = json.load(file)
    else:
        result = {
            "status": "CRASHED",
            "error_message": (process.stderr.strip() or process.stdout.strip())[-2000:],
        }

    result["_vram_fraction_requested"] = fraction
    result["_subprocess_returncode"] = process.returncode

    status = result.get("status", "UNKNOWN")
    print(f"  -> status: {status}")
    if result.get("error_message"):
        print(f"  -> error: {result['error_message'][:200]}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep vram_fraction to bracket the F2 feasibility floor")
    parser.add_argument("--config", type=str, default="configs/feasibility.yaml")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=DEFAULT_FRACTIONS,
        help=f"VRAM fractions to test. Default: {DEFAULT_FRACTIONS}",
    )
    parser.add_argument("--results-dir", type=str, default="results/f2")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fraction in sorted(args.fractions):
        result = run_single_fraction(args.config, fraction, results_dir)
        all_results.append(result)

    summary_path = results_dir / "vram_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"\n\n{'=' * 70}")
    print("VRAM SWEEP SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'Fraction':<10} | {'Cap (MB)':<10} | {'Status':<25} | {'Peak VRAM (MB)':<15} | {'Loss Δ':<10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        frac = r.get("_vram_fraction_requested", "?")
        cap_mb = r.get("vram_cap_info", {}).get("effective_cap_mb", "N/A") if r.get("vram_cap_info") else "N/A"
        status = r.get("status", "UNKNOWN")
        peak_vram = r.get("peak_vram_mb", "N/A")
        loss_reduction = r.get("loss_reduction", "N/A")
        input_shapes = r.get("input_shapes", "N/A")
        print(f"{frac:<10} | {cap_mb!s:<10} | {status:<25} | {peak_vram!s:<15} | {loss_reduction!s:<10}")

    print(f"\nFull results: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
