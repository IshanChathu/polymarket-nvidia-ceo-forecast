"""
Multi-Step Forecasting Engine for Polymarket Prediction Markets.
Forecasts minute-by-minute trajectory 240 minutes (4 hours) ahead.
Uses Log-Odds (Multinomial Logit) transformation + Mean-Reverting Autoregressive Drift,
guaranteeing probability bounds [0, 1] and strict sum-to-one constraint across all outcomes.
"""

import math
from typing import Dict, List, Tuple, Any


class MultiStepForecaster:
    def __init__(
        self,
        horizon_minutes: int = 240,
        mean_reversion_half_life: float = 360.0, # minutes
        momentum_decay: float = 0.96,
    ):
        self.horizon_minutes = horizon_minutes
        self.mean_reversion_half_life = mean_reversion_half_life
        self.momentum_decay = momentum_decay

    def forecast_joint_candidates(
        self,
        candidate_features: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Produces 240-minute forward trajectories for all mutually exclusive candidates.
        Guarantees that at each future minute t+k, sum(p_i) == 1.0.
        """
        candidates = list(candidate_features.keys())
        if not candidates:
            return {}

        # 1. Extract current state & recent history for each candidate
        last_t = candidate_features[candidates[0]][-1]["t"]
        current_prices = {}
        momentum_short = {}
        momentum_long = {}
        volatilities = {}
        long_term_anchors = {}

        for c in candidates:
            feats = candidate_features[c]
            curr_row = feats[-1]
            current_prices[c] = curr_row["price"]
            momentum_short[c] = curr_row.get("mom_15", 0.0) / 15.0 # per-minute drift
            momentum_long[c] = curr_row.get("mom_60", 0.0) / 60.0
            volatilities[c] = curr_row.get("vol_30", 0.01)

            # Long-term anchor: 4-hour moving average
            lookback_anchor = min(len(feats), 240)
            avg_p = sum(f["price"] for f in feats[-lookback_anchor:]) / lookback_anchor
            long_term_anchors[c] = avg_p

        # 2. Normalize current prices to ensure valid initial probability distribution
        total_curr = sum(current_prices.values())
        norm_current = {c: current_prices[c] / total_curr for c in candidates}

        # 3. Transform to unconstrained log-odds space (z_i = ln(p_i))
        # Logit-space modeling prevents probabilities from ever leaving (0, 1)
        z_curr = {c: math.log(max(1e-5, norm_current[c])) for c in candidates}
        z_anchor = {c: math.log(max(1e-5, long_term_anchors[c])) for c in candidates}

        # Mean reversion speed kappa: ln(2) / half_life
        kappa = math.log(2.0) / self.mean_reversion_half_life

        # 4. Generate forward minute-by-minute trajectory
        forecasts_by_candidate: Dict[str, List[Dict[str, Any]]] = {c: [] for c in candidates}

        current_drift = {}
        for c in candidates:
            # Blend short and long momentum
            blended_drift = 0.7 * momentum_short[c] + 0.3 * momentum_long[c]
            # Convert drift to log-odds scale: dz/dt ~ (1/p) * dp/dt
            p_val = max(0.01, norm_current[c])
            current_drift[c] = blended_drift / p_val

        z_state = dict(z_curr)
        drift_state = dict(current_drift)

        for step in range(1, self.horizon_minutes + 1):
            future_t = last_t + (step * 60)

            # Update log-odds state with decaying momentum + mean reversion to anchor
            for c in candidates:
                drift_contrib = drift_state[c]
                mr_contrib = kappa * (z_anchor[c] - z_state[c])
                z_state[c] += (drift_contrib + mr_contrib)
                drift_state[c] *= self.momentum_decay

            # Softmax projection to guarantee probabilities strictly sum to 1.0
            max_z = max(z_state.values())
            exp_z = {c: math.exp(z_state[c] - max_z) for c in candidates}
            sum_exp = sum(exp_z.values())
            step_probs = {c: exp_z[c] / sum_exp for c in candidates}

            # Calculate confidence bounds (90% interval) based on square-root diffusion
            dt_hours = step / 60.0
            for c in candidates:
                p_hat = step_probs[c]
                vol = volatilities[c]
                # Diffusion standard deviation: sigma * sqrt(t) * p * (1 - p)
                sigma_p = vol * math.sqrt(dt_hours) * p_hat * (1.0 - p_hat)
                p_lower = max(0.001, round(p_hat - 1.645 * sigma_p, 4))
                p_upper = min(0.999, round(p_hat + 1.645 * sigma_p, 4))

                forecasts_by_candidate[c].append({
                    "step": step,
                    "t": future_t,
                    "forecast_price": round(p_hat, 4),
                    "lower_90": p_lower,
                    "upper_90": p_upper,
                })

        return forecasts_by_candidate
