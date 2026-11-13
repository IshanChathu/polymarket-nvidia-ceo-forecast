"""
Polymarket Next NVIDIA CEO Minute-by-Minute Forecaster
Competition: polymarket-nvidia-ceo-forecast-01

Reads input cases from MAKEFUN_INPUT_DIR (or /input),
computes calibrated per-market volatility and decaying drift from recent prices,
and generates quantile forecasts [0.05, 0.25, 0.5, 0.75, 0.95] for 240 forward minutes.
Writes output forecasts to MAKEFUN_OUTPUT_DIR (or /output).
"""

import json
import math
import os
import sys
from typing import Dict, List, Any

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Quantile levels and corresponding standard normal critical values
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Baseline volatility prior (per square-root minute)
BASE_SIGMA = 0.0025


def clamp_price6(x: float) -> float:
    """Clamp value to [0, 1] and round to 6 decimal places (0.0, never -0.0)."""
    val = round(min(1.0, max(0.0, x)), 6)
    return 0.0 if val == -0.0 else val


def compute_market_stats(prices: List[float]) -> tuple[float, float]:
    """
    Computes per-minute volatility (sigma) and decaying drift from recent price history.
    """
    if len(prices) < 2:
        return BASE_SIGMA, 0.0

    # Use recent window (up to 120 minutes) for volatility estimation
    w_vol = min(len(prices), 120)
    recent_prices = prices[-w_vol:]

    diffs = [recent_prices[i] - recent_prices[i - 1] for i in range(1, len(recent_prices))]
    if not diffs:
        return BASE_SIGMA, 0.0

    # Mean squared difference per minute
    msd = sum(d * d for d in diffs) / len(diffs)
    empirical_sigma = math.sqrt(msd)

    # Shrinkage toward prior BASE_SIGMA to avoid overfitting to flat stretches
    # (Empirical Bayes weighting: 60% empirical, 40% prior)
    sigma = 0.6 * empirical_sigma + 0.4 * BASE_SIGMA
    # Bound sigma in sensible range [0.0005, 0.025]
    sigma = max(0.0005, min(0.025, sigma))

    # Compute short-term drift (from last 15-30 minutes)
    w_drift = min(len(prices), 30)
    p_now = prices[-1]
    p_past = prices[-w_drift]
    # Drift per minute
    raw_drift = (p_now - p_past) / float(w_drift)
    # Bound drift per minute (equivalent to max +/- 5% per hour)
    drift = max(-0.0008, min(0.0008, raw_drift))

    return sigma, drift


def forecast_market(market: Dict[str, Any], horizon: int) -> List[List[float]]:
    """
    Produces horizon rows of the 5 quantiles for a single market.
    """
    prices = market.get("prices", [])
    if not prices:
        raise ValueError(f"Market {market.get('id')} has no prices")

    x0 = prices[-1]
    sigma, drift = compute_market_stats(prices)

    # Drift half-life: 45 minutes (momentum fades into mean-reversion)
    decay_rate = math.exp(-math.log(2.0) / 45.0)

    rows: List[List[float]] = []
    current_drift_accum = 0.0
    active_drift = drift

    for h in range(1, horizon + 1):
        current_drift_accum += active_drift
        active_drift *= decay_rate

        expected_mid = x0 + current_drift_accum
        spread = sigma * math.sqrt(h)

        row = []
        for z_val in Z:
            q_val = clamp_price6(expected_mid + z_val * spread)
            row.append(q_val)

        # Ensure monotonicity: q_0 <= q_1 <= q_2 <= q_3 <= q_4
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]

        rows.append(row)

    return rows


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    input_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".in")])
    if not input_files:
        print(f"No .in files found in {INPUT_DIR}", file=sys.stderr)
        sys.exit(0)

    for file_name in input_files:
        case_id = file_name[:-3] # Strip '.in'
        input_path = os.path.join(INPUT_DIR, file_name)

        with open(input_path, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        horizon = case_data.get("horizon_minutes", 240)
        markets = case_data.get("markets", [])

        forecasts = []
        for m in markets:
            q_rows = forecast_market(m, horizon)
            forecasts.append({"id": m["id"], "q": q_rows})

        output_path = os.path.join(OUTPUT_DIR, f"{case_id}.out")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))
            f.write("\n")

    print(f"Successfully processed {len(input_files)} cases into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
