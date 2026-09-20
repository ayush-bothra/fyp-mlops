import argparse
import csv
import os
import threading
import time

import psutil
import pynvml


FIELDNAMES = [
    "episode",
    "battery_level",
    "cpu_usage",
    "memory_usage",
    "temperature_C",
    "energy_consumed_mJ",
    "queue_size",
    "latency_ms",
    "power_draw_watts",
]


class TelemetryLogger:
    def __init__(self, output_path, episode, interval_seconds=1.0, device_index=0):
        self.output_path = output_path
        self.episode = episode
        self.interval_seconds = interval_seconds
        self.device_index = device_index
        self.queue_size = 0
        self.latency_ms = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def set_pipeline_state(self, queue_size, latency_ms):
        self.queue_size = queue_size
        self.latency_ms = latency_ms

    def _read_row(self, handle, elapsed_seconds):
        temperature_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        energy_consumed_mj = power_watts * elapsed_seconds * 1000.0
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        battery_level = battery.percent if battery is not None else 100.0

        return {
            "episode": self.episode,
            "battery_level": battery_level,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "temperature_C": temperature_c,
            "energy_consumed_mJ": energy_consumed_mj,
            "queue_size": self.queue_size,
            "latency_ms": self.latency_ms,
            "power_draw_watts": power_watts,
        }

    def _run(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            file.flush()
            last_time = time.time()
            while not self._stop_event.is_set():
                now = time.time()
                elapsed = now - last_time
                last_time = now
                row = self._read_row(handle, elapsed)
                writer.writerow(row)
                file.flush()
                self._stop_event.wait(self.interval_seconds)

        pynvml.nvmlShutdown()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/telemetry/session.csv")
    parser.add_argument("--episode", default="manual_session")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    logger = TelemetryLogger(
        output_path=args.output,
        episode=args.episode,
        interval_seconds=args.interval,
        device_index=args.device_index,
    )
    logger.start()
    print(f"Logging telemetry for episode '{args.episode}' to {args.output}")
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    logger.stop()
    print("Stopped, telemetry saved")


if __name__ == "__main__":
    main()
