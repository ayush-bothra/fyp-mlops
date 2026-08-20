"""
Script to download raw datasets for the FYP MLOps project.

Datasets:
1. IoT-RL: Solar, Wind & Adaptive Transmission Control (via kagglehub -> data/raw/iot_telemetry/)
2. CORe50 Visual Stream Dataset (via Official University of Bologna release -> data/raw/core50/)
"""

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile
import kagglehub

RAW_DATA_DIR = Path("data/raw")

IOT_KAGGLE_HANDLE = "wisam1985/iot-rl-solarwind-and-adaptive-transmission-control"
CORE50_URL = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"


def download_iot_telemetry() -> Path:
    """Downloads the IoT-RL telemetry dataset from Kaggle using kagglehub."""
    print(f"\n--- Downloading IoT-RL Telemetry Dataset ('{IOT_KAGGLE_HANDLE}') ---")
    cached_path = Path(kagglehub.dataset_download(IOT_KAGGLE_HANDLE))
    print(f"Cached at: {cached_path}")

    dest_path = RAW_DATA_DIR / "iot_telemetry"
    dest_path.mkdir(parents=True, exist_ok=True)

    if cached_path.is_file():
        shutil.copy2(cached_path, dest_path)
    else:
        shutil.copytree(cached_path, dest_path, dirs_exist_ok=True)

    print(f"IoT telemetry dataset ready at: {dest_path.resolve()}")
    return dest_path


def download_core50() -> Path:
    """Downloads and extracts the official CORe50 128x128 benchmark dataset."""
    dest_path = RAW_DATA_DIR / "core50"
    dest_path.mkdir(parents=True, exist_ok=True)

    zip_file_path = RAW_DATA_DIR / "core50_128x128.zip"

    print(f"\n--- Downloading CORe50 Dataset from official source ---")
    print(f"URL: {CORE50_URL}")
    print(f"Saving to: {zip_file_path}")

    def progress_bar(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB)", end="")
        else:
            print(f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB", end="")

    urllib.request.urlretrieve(CORE50_URL, zip_file_path, reporthook=progress_bar)
    print("\nDownload complete. Extracting files...")

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)

    # Clean up zip archive
    if zip_file_path.exists():
        zip_file_path.unlink()

    print(f"CORe50 dataset extracted and ready at: {dest_path.resolve()}")
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "iot", "core50"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "iot"):
        download_iot_telemetry()

    if args.dataset in ("all", "core50"):
        download_core50()

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()
