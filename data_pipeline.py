"""
Feature Engineering & Data Pipeline for Minute-by-Minute Prediction Market Forecasting.
Processes raw market candles into clean feature representations including
rolling VWAP, EMA, realized volatility, spread, and multi-asset sum-to-one constraints.
"""

import math
from typing import Dict, List, Tuple, Any


class DataPipeline:
    def __init__(self, ema_spans: Tuple[int, ...] = (10, 30, 120), vwap_windows: Tuple[int, ...] = (5, 15, 60)):
        self.ema_spans = ema_spans
        self.vwap_windows = vwap_windows

    def compute_features(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms a single candidate's price/volume history into feature dicts.
        """
        if not history:
            return []

        features_list = []
        prices = [h["price"] for h in history]
        volumes = [h.get("volume", 1.0) for h in history]
        bids = [h.get("bid", h["price"]) for h in history]
        asks = [h.get("ask", h["price"]) for h in history]
        timestamps = [h["t"] for h in history]

        n = len(prices)

        # Precompute EMAs
        emas: Dict[int, List[float]] = {}
        for span in self.ema_spans:
            alpha = 2.0 / (span + 1.0)
            ema_vals = [prices[0]]
            for i in range(1, n):
                ema_vals.append(alpha * prices[i] + (1 - alpha) * ema_vals[-1])
            emas[span] = ema_vals

        for i in range(n):
            curr_p = prices[i]
            curr_vol = volumes[i]
            spread = max(0.0001, asks[i] - bids[i])

            # Rolling VWAP
            vwap_feats = {}
            for w in self.vwap_windows:
                start_idx = max(0, i - w + 1)
                sub_p = prices[start_idx : i + 1]
                sub_v = volumes[start_idx : i + 1]
                total_vol = sum(sub_v)
                if total_vol > 0:
                    vwap = sum(p * v for p, v in zip(sub_p, sub_v)) / total_vol
                else:
                    vwap = curr_p
                vwap_feats[f"vwap_{w}"] = round(vwap, 5)
                vwap_feats[f"p_to_vwap_{w}"] = round(curr_p - vwap, 5)

            # Momentum / Price differences
            mom_5 = curr_p - prices[max(0, i - 5)]
            mom_15 = curr_p - prices[max(0, i - 15)]
            mom_60 = curr_p - prices[max(0, i - 60)]

            # Realized volatility (rolling 30-min log returns)
            w_vol = 30
            start_vol = max(1, i - w_vol + 1)
            log_rets = []
            for j in range(start_vol, i + 1):
                if prices[j - 1] > 0 and prices[j] > 0:
                    log_rets.append(math.log(prices[j] / prices[j - 1]))
            if len(log_rets) > 1:
                mean_ret = sum(log_rets) / len(log_rets)
                var_ret = sum((r - mean_ret) ** 2 for r in log_rets) / (len(log_rets) - 1)
                vol_30 = math.sqrt(max(0.0, var_ret)) * math.sqrt(1440) # annualized minute vol
            else:
                vol_30 = 0.01

            row = {
                "t": timestamps[i],
                "price": curr_p,
                "volume": curr_vol,
                "spread": round(spread, 5),
                "mom_5": round(mom_5, 5),
                "mom_15": round(mom_15, 5),
                "mom_60": round(mom_60, 5),
                "vol_30": round(vol_30, 5),
            }
            row.update(vwap_feats)
            for span in self.ema_spans:
                row[f"ema_{span}"] = round(emas[span][i], 5)
                row[f"p_to_ema_{span}"] = round(curr_p - emas[span][i], 5)

            features_list.append(row)

        return features_list

    def align_and_normalize_multivariate(
        self,
        candidate_histories: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Aligns timestamps across all candidates and applies softmax/Dirichlet projection
        so probabilities sum exactly to 1.0 at each step.
        """
        candidates = list(candidate_histories.keys())
        if not candidates:
            return {}

        # Build map by timestamp
        time_map: Dict[int, Dict[str, float]] = {}
        for cand, history in candidate_histories.items():
            for entry in history:
                t = entry["t"]
                if t not in time_map:
                    time_map[t] = {}
                time_map[t][cand] = entry["price"]

        # Filter timestamps where all candidates are present
        common_timestamps = sorted(t for t, p_dict in time_map.items() if len(p_dict) == len(candidates))

        # Normalized histories
        normalized_histories: Dict[str, List[Dict[str, Any]]] = {c: [] for c in candidates}

        for t in common_timestamps:
            raw_prices = [time_map[t][c] for c in candidates]
            # Clean non-negative clamp
            clamped = [max(0.001, p) for p in raw_prices]
            sum_p = sum(clamped)
            norm_p = [p / sum_p for p in clamped]

            for idx, c in enumerate(candidates):
                normalized_histories[c].append({
                    "t": t,
                    "price": round(norm_p[idx], 5),
                })

        return normalized_histories
