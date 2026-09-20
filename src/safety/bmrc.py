from dataclasses import dataclass

from scipy import stats


@dataclass
class BMRCDecision:
    action: str
    breach_probability: float
    z_score: float


def decide_retrain_action(mean_temperature, std_temperature, temperature_limit=80.0, risk_threshold=0.05):
    safe_std = max(std_temperature, 1e-6)
    z_score = (temperature_limit - mean_temperature) / safe_std
    breach_probability = stats.norm.cdf(-z_score)
    action = "QLORA" if breach_probability < risk_threshold else "WAIT"
    return BMRCDecision(action=action, breach_probability=breach_probability, z_score=z_score)

def combine_decisions(decisions):
    action = "QLORA" if all(decision.action == "QLORA" for decision in decisions) else "WAIT"
    limiting_decision = max(decisions, key=lambda decision: decision.breach_probability)
    return action, limiting_decision
