"""
Main Orchestration Script for Polymarket NVIDIA CEO Forecasting Bounty.
Executes data ingestion, feature extraction, multi-step probability forecasting,
constraint validation, and outputs ready-to-submit artifacts.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

from polymarket_client import PolymarketClient
from data_pipeline import DataPipeline
from forecasting_engine import MultiStepForecaster
from evaluator import ForecastEvaluator


def run_pipeline(output_dir: str = ".") -> int:
    print("=" * 70)
    print(" Polymarket NVIDIA CEO Forecasting Pipeline (make.fun bounty)")
    print(" Target: 240-minute (4 hours) minute-by-minute price forecast")
    print("=" * 70)

    # 1. Ingestion
    print("\n[1/5] Fetching candidate market series from Polymarket...")
    client = PolymarketClient(offline_fallback=True)
    meta = client.fetch_market_meta("nvidia-next-ceo")
    print(f"      Market: {meta.get('title')}")
    print(f"      Resolution Date: {meta.get('resolutionDate')}")
    
    raw_histories = client.fetch_all_candidate_series(lookback_minutes=1440)
    for cand, hist in raw_histories.items():
        latest_price = hist[-1]["price"] if hist else 0.0
        print(f"      - {cand:<18}: {len(hist)} minute bars | Latest: {latest_price:.4f}")

    # 2. Feature Engineering
    print("\n[2/5] Engineering rolling features (VWAP, EMA, Momentum, Realized Vol)...")
    pipeline = DataPipeline()
    features_by_candidate = {}
    for cand, hist in raw_histories.items():
        feats = pipeline.compute_features(hist)
        features_by_candidate[cand] = feats

    # 3. Multi-Step Probability Forecasting
    print("\n[3/5] Generating 240-minute forward trajectory with log-odds projection...")
    forecaster = MultiStepForecaster(horizon_minutes=240)
    forecasts = forecaster.forecast_joint_candidates(features_by_candidate)

    # 4. Constraint Validation
    print("\n[4/5] Validating competition rules & probabilistic consistency...")
    is_valid, validation_errors = ForecastEvaluator.validate_constraints(forecasts)
    if is_valid:
        print("      ✓ ALL CHECKS PASSED:")
        print("        - Exactly 240 forward minute steps generated per candidate.")
        print("        - Probability bounds [0, 1] strictly maintained.")
        print("        - Sum-to-one constraint satisfied at every minute step.")
    else:
        print("      ✗ Validation errors detected:")
        for err in validation_errors[:5]:
            print(f"        - {err}")
        return 1

    # 5. Export Submission Artifacts
    print("\n[5/5] Exporting submission artifacts...")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "submission.json")
    csv_path = os.path.join(output_dir, "submission.csv")

    # JSON Structure
    submission_payload = {
        "bounty_id": "polymarket-nvidia-ceo-forecast-01",
        "market": "nvidia-next-ceo",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_minutes": 240,
        "candidates": list(forecasts.keys()),
        "forecasts": forecasts,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission_payload, f, indent=2)
    print(f"      ✓ Saved JSON submission: {json_path}")

    # CSV Structure: minute, candidate, timestamp, forecast_price, lower_90, upper_90
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "timestamp_utc", "candidate", "forecast_probability", "lower_90", "upper_90"])
        for step_idx in range(240):
            for cand in forecasts:
                item = forecasts[cand][step_idx]
                ts_str = datetime.fromtimestamp(item["t"], tz=timezone.utc).isoformat()
                writer.writerow([
                    item["step"],
                    ts_str,
                    cand,
                    item["forecast_price"],
                    item["lower_90"],
                    item["upper_90"],
                ])
    print(f"      ✓ Saved CSV submission:  {csv_path}")

    # Preview key steps: t+15m, t+60m, t+120m, t+240m
    print("\n" + "=" * 70)
    print(" PREVIEW: Forecasted Probability Trajectory")
    print("=" * 70)
    print(f"{'Candidate':<18} | {'t+15m':<8} | {'t+60m':<8} | {'t+120m':<8} | {'t+240m':<8}")
    print("-" * 70)
    for cand in forecasts:
        p15 = forecasts[cand][14]["forecast_price"]
        p60 = forecasts[cand][59]["forecast_price"]
        p120 = forecasts[cand][119]["forecast_price"]
        p240 = forecasts[cand][239]["forecast_price"]
        print(f"{cand:<18} | {p15:.4f}   | {p60:.4f}   | {p120:.4f}   | {p240:.4f}")
    print("-" * 70)
    print("Pipeline run complete.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket NVIDIA CEO Forecaster")
    parser.add_argument("--output-dir", default=".", help="Directory to save submission files")
    args = parser.parse_args()
    sys.exit(run_pipeline(args.output_dir))
