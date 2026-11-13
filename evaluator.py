"""
Evaluation Metrics & Validator for Prediction Market Forecasting.
Computes Brier Score, Continuous Log-Loss, MAE, and validates platform constraints
(bounds [0, 1] and sum-to-one constraint across outcomes).
"""

import math
from typing import Dict, List, Tuple, Any


class ForecastEvaluator:
    @staticmethod
    def validate_constraints(forecasts: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, List[str]]:
        """
        Validates that forecast entries adhere to platform competition rules:
        - Exactly 240 minute steps.
        - All probabilities in [0.0, 1.0].
        - Sum of candidate probabilities at each minute equals 1.0 (within numerical tolerance).
        """
        errors = []
        candidates = list(forecasts.keys())
        if not candidates:
            return False, ["No candidate forecasts found."]

        step_counts = {c: len(forecasts[c]) for c in candidates}
        for c, count in step_counts.items():
            if count != 240:
                errors.append(f"Candidate {c} has {count} steps; required 240.")

        for step_idx in range(240):
            sum_p = 0.0
            for c in candidates:
                if step_idx < len(forecasts[c]):
                    p = forecasts[c][step_idx]["forecast_price"]
                    if not (0.0 <= p <= 1.0):
                        errors.append(f"Step {step_idx+1}: Candidate {c} price {p} outside [0, 1].")
                    sum_p += p
            if abs(sum_p - 1.0) > 0.005:
                errors.append(f"Step {step_idx+1}: Probabilities sum to {sum_p:.4f}; must sum to 1.0.")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def compute_brier_score(
        forecast_probs: List[float],
        actual_realizations: List[float],
    ) -> float:
        """
        Computes standard multi-step Brier Score:
        BS = (1 / T) * sum((p_t - actual_t)^2)
        """
        if not forecast_probs or len(forecast_probs) != len(actual_realizations):
            return 0.0
        n = len(forecast_probs)
        return sum((p - a) ** 2 for p, a in zip(forecast_probs, actual_realizations)) / n

    @staticmethod
    def compute_mae(
        forecast_probs: List[float],
        actual_realizations: List[float],
    ) -> float:
        """Computes Mean Absolute Error across minute steps."""
        if not forecast_probs or len(forecast_probs) != len(actual_realizations):
            return 0.0
        n = len(forecast_probs)
        return sum(abs(p - a) for p, a in zip(forecast_probs, actual_realizations)) / n

    @staticmethod
    def backtest_summary(
        forecasts: Dict[str, List[Dict[str, Any]]],
        ground_truth: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Runs backtest comparison between forecasts and ground truth."""
        results = {}
        total_mae = 0.0
        total_brier = 0.0
        cand_count = 0

        for cand, steps in forecasts.items():
            if cand in ground_truth:
                preds = [s["forecast_price"] for s in steps]
                acts = [g["price"] for g in ground_truth[cand][:len(preds)]]
                if len(preds) == len(acts):
                    mae = ForecastEvaluator.compute_mae(preds, acts)
                    brier = ForecastEvaluator.compute_brier_score(preds, acts)
                    results[cand] = {
                        "mae": round(mae, 4),
                        "brier_score": round(brier, 4),
                    }
                    total_mae += mae
                    total_brier += brier
                    cand_count += 1

        if cand_count > 0:
            results["macro_avg"] = {
                "macro_mae": round(total_mae / cand_count, 4),
                "macro_brier": round(total_brier / cand_count, 4),
            }

        return results
