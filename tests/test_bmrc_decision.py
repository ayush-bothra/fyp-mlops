import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.safety.bmrc import decide_retrain_action


def test_confident_safe_forecast_triggers_qlora():
    decision = decide_retrain_action(mean_temperature=55.0, std_temperature=3.0, temperature_limit=80.0)
    if decision.action != "QLORA":
        return False, f"expected QLORA for a forecast far below the limit, got {decision.action}"
    return True, f"correctly chose QLORA with breach_probability={decision.breach_probability:.5f}"


def test_forecast_at_limit_triggers_wait():
    decision = decide_retrain_action(mean_temperature=80.0, std_temperature=3.0, temperature_limit=80.0)
    if decision.action != "WAIT":
        return False, f"expected WAIT when the forecast mean equals the limit, got {decision.action}"
    return True, f"correctly chose WAIT with breach_probability={decision.breach_probability:.5f}"


def test_p_value_matches_hand_computed_z():
    decision = decide_retrain_action(mean_temperature=74.0, std_temperature=2.0, temperature_limit=80.0)
    expected_z = 3.0
    if abs(decision.z_score - expected_z) > 1e-9:
        return False, f"expected z_score={expected_z}, got {decision.z_score}"
    if not (0.0009 < decision.breach_probability < 0.0015):
        return False, f"breach_probability {decision.breach_probability} doesn't match the known value for z=3.0 (~0.00135)"
    return True, f"z_score and breach_probability both match hand-computed values for z={expected_z}"


def test_zero_std_does_not_crash():
    try:
        decision = decide_retrain_action(mean_temperature=70.0, std_temperature=0.0, temperature_limit=80.0)
    except ZeroDivisionError as exc:
        return False, f"raised ZeroDivisionError instead of using the epsilon guard: {exc}"
    return True, f"handled zero uncertainty without crashing, action={decision.action}"


def main():
    tests = [
        test_confident_safe_forecast_triggers_qlora,
        test_forecast_at_limit_triggers_wait,
        test_p_value_matches_hand_computed_z,
        test_zero_std_does_not_crash,
    ]

    results = [(test.__name__, *test()) for test in tests]

    print()
    for name, passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {message}")

    failed_count = sum(1 for _, passed, _ in results if not passed)
    print(f"\n{len(results) - failed_count}/{len(results)} tests passed")
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
