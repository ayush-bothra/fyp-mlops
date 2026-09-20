import csv
import sys
import time
import pynvml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.telemetry.logger import TelemetryLogger, FIELDNAMES


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def test_logging_writes_expected_schema(tmp_path):
    output_path = tmp_path / "schema_check.csv"
    logger = TelemetryLogger(output_path=str(output_path), episode="schema_check", interval_seconds=0.2)
    logger.start()
    time.sleep(1.0)
    logger.stop()

    rows = read_rows(output_path)
    if len(rows) == 0:
        return False, "no rows written"
    if list(rows[0].keys()) != FIELDNAMES:
        return False, f"schema mismatch, got {list(rows[0].keys())}"
    return True, f"{len(rows)} rows written with correct schema"


def test_pipeline_state_is_reflected(tmp_path):
    output_path = tmp_path / "pipeline_state_check.csv"
    logger = TelemetryLogger(output_path=str(output_path), episode="pipeline_state_check", interval_seconds=0.2)
    logger.set_pipeline_state(queue_size=42, latency_ms=123.5)
    logger.start()
    time.sleep(1.0)
    logger.stop()

    rows = read_rows(output_path)
    last_row = rows[-1]
    if int(last_row["queue_size"]) != 42:
        return False, f"expected queue_size 42, got {last_row['queue_size']}"
    if abs(float(last_row["latency_ms"]) - 123.5) > 1e-6:
        return False, f"expected latency_ms 123.5, got {last_row['latency_ms']}"
    return True, "pipeline state values passed through correctly"


def test_context_manager_stops_thread(tmp_path):
    output_path = tmp_path / "context_manager_check.csv"
    with TelemetryLogger(output_path=str(output_path), episode="context_manager_check", interval_seconds=0.2) as logger:
        time.sleep(0.5)
        thread_was_alive = logger._thread.is_alive()

    thread_alive_after_exit = logger._thread.is_alive()
    if not thread_was_alive:
        return False, "thread was not running inside the with block"
    if thread_alive_after_exit:
        return False, "thread is still running after the with block exited"
    return True, "context manager started and stopped the thread correctly"


def test_values_are_in_plausible_ranges(tmp_path):
    output_path = tmp_path / "range_check.csv"
    logger = TelemetryLogger(output_path=str(output_path), episode="range_check", interval_seconds=0.2)
    logger.start()
    time.sleep(1.0)
    logger.stop()

    rows = read_rows(output_path)
    for row in rows:
        temperature = float(row["temperature_C"])
        cpu_usage = float(row["cpu_usage"])
        memory_usage = float(row["memory_usage"])
        battery_level = float(row["battery_level"])
        if not (0 <= temperature <= 110):
            return False, f"temperature_C out of range: {temperature}"
        if not (0 <= cpu_usage <= 100):
            return False, f"cpu_usage out of range: {cpu_usage}"
        if not (0 <= memory_usage <= 100):
            return False, f"memory_usage out of range: {memory_usage}"
        if not (0 <= battery_level <= 100):
            return False, f"battery_level out of range: {battery_level}"
    return True, f"all {len(rows)} rows within plausible ranges"


def main():
    import tempfile

    tests = [
        test_logging_writes_expected_schema,
        test_pipeline_state_is_reflected,
        test_context_manager_stops_thread,
        test_values_are_in_plausible_ranges,
    ]

    pynvml.nvmlInit()
    results = []
    for test_func in tests:
        with tempfile.TemporaryDirectory() as tmp_dir:
            passed, message = test_func(Path(tmp_dir))
            results.append((test_func.__name__, passed, message))

    print()
    for name, passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {message}")

    failed_count = sum(1 for _, passed, _ in results if not passed)
    print(f"\n{len(results) - failed_count}/{len(results)} tests passed")
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
