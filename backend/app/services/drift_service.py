"""Data Drift and Model Health Monitoring Service.

Computes statistical distribution shifts between reference (training) and current data:
- Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests for numerical features
- Categorical distribution shift and new-category detection
- Missing-value and schema-change anomaly checks
- Prediction distribution drift
- Model Health Status (HEALTHY, WARNING, DRIFT_DETECTED, CRITICAL)
- Evidence-based retraining trigger evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between reference and actual distributions."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # Determine quantiles on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges = np.unique(bin_edges)  # handle discrete duplicates

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractions with Laplace smoothing
    expected_frac = (expected_counts + 1e-4) / (len(expected) + 1e-4 * len(expected_counts))
    actual_frac = (actual_counts + 1e-4) / (len(actual) + 1e-4 * len(actual_counts))

    psi_val = np.sum((actual_frac - expected_frac) * np.log(actual_frac / expected_frac))
    return float(max(0.0, psi_val))


class DriftMonitoringService:
    """Monitors feature drift, prediction drift, and schema health."""

    def __init__(
        self,
        psi_warning_threshold: float = 0.10,
        psi_drift_threshold: float = 0.25,
        ks_alpha: float = 0.01,
        null_spike_threshold: float = 0.10,
    ):
        self.psi_warning_threshold = float(psi_warning_threshold)
        self.psi_drift_threshold = float(psi_drift_threshold)
        self.ks_alpha = float(ks_alpha)
        self.null_spike_threshold = float(null_spike_threshold)

    def analyze_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_predictions: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None,
        feature_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Compute full statistical drift report comparing reference vs current data."""
        cols = feature_columns or [
            c for c in reference_df.columns if c in current_df.columns and not c.lower().endswith("id")
        ]

        feature_reports: list[dict[str, Any]] = []
        alerts: list[dict[str, Any]] = []

        total_features = len(cols)
        drifted_count = 0
        warning_count = 0

        for col in cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref_s = reference_df[col]
            cur_s = current_df[col]

            # 1. Null / Missingness check
            ref_null_rate = float(ref_s.isnull().mean())
            cur_null_rate = float(cur_s.isnull().mean())
            null_diff = cur_null_rate - ref_null_rate

            if null_diff >= self.null_spike_threshold:
                alerts.append(
                    {
                        "feature": col,
                        "type": "missingness_spike",
                        "severity": "high",
                        "message": f"Null rate spiked from {ref_null_rate*100:.1f}% to {cur_null_rate*100:.1f}%.",
                    }
                )

            # 2. Check if numerical
            is_numeric = pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(cur_s)

            if is_numeric:
                ref_clean = pd.to_numeric(ref_s, errors="coerce").dropna().values
                cur_clean = pd.to_numeric(cur_s, errors="coerce").dropna().values

                if len(ref_clean) >= 10 and len(cur_clean) >= 10:
                    psi_score = calculate_psi(ref_clean, cur_clean)
                    ks_stat, ks_pval = ks_2samp(ref_clean, cur_clean)

                    status = "normal"
                    if psi_score >= self.psi_drift_threshold or ks_pval < self.ks_alpha:
                        status = "drift_detected"
                        drifted_count += 1
                        alerts.append(
                            {
                                "feature": col,
                                "type": "numerical_drift",
                                "severity": "high",
                                "message": f"Significant distribution drift: PSI={psi_score:.3f}, KS p-val={ks_pval:.4f}.",
                            }
                        )
                    elif psi_score >= self.psi_warning_threshold:
                        status = "warning"
                        warning_count += 1

                    feature_reports.append(
                        {
                            "feature": col,
                            "type": "numeric",
                            "status": status,
                            "psi": round(psi_score, 4),
                            "ks_statistic": round(float(ks_stat), 4),
                            "ks_p_value": round(float(ks_pval), 5),
                            "reference_mean": round(float(np.mean(ref_clean)), 2),
                            "current_mean": round(float(np.mean(cur_clean)), 2),
                            "reference_std": round(float(np.std(ref_clean)), 2),
                            "current_std": round(float(np.std(cur_clean)), 2),
                            "null_rate_change": round(null_diff, 3),
                        }
                    )
            else:
                # Categorical drift
                ref_cats = set(ref_s.dropna().astype(str).unique())
                cur_cats = set(cur_s.dropna().astype(str).unique())
                new_cats = list(cur_cats - ref_cats)

                ref_counts = ref_s.value_counts(normalize=True).to_dict()
                cur_counts = cur_s.value_counts(normalize=True).to_dict()

                # Simple variation distance
                all_cats = list(ref_cats | cur_cats)
                tvd = 0.5 * sum(abs(ref_counts.get(c, 0.0) - cur_counts.get(c, 0.0)) for c in all_cats)

                status = "normal"
                if tvd >= 0.20 or bool(new_cats):
                    status = "drift_detected" if tvd >= 0.25 else "warning"
                    if status == "drift_detected":
                        drifted_count += 1
                    else:
                        warning_count += 1

                    if new_cats:
                        alerts.append(
                            {
                                "feature": col,
                                "type": "new_categories",
                                "severity": "medium",
                                "message": f"Unobserved categories in incoming data: {new_cats[:3]}.",
                            }
                        )

                feature_reports.append(
                    {
                        "feature": col,
                        "type": "categorical",
                        "status": status,
                        "total_variation_distance": round(float(tvd), 3),
                        "new_categories_detected": new_cats,
                        "null_rate_change": round(null_diff, 3),
                    }
                )

        # 3. Prediction Distribution Drift
        pred_drift: dict[str, Any] = {"available": False}
        if reference_predictions is not None and current_predictions is not None:
            ref_p = np.asarray(reference_predictions).ravel()
            cur_p = np.asarray(current_predictions).ravel()
            if len(ref_p) >= 10 and len(cur_p) >= 10:
                p_psi = calculate_psi(ref_p, cur_p)
                p_ks_stat, p_ks_pval = ks_2samp(ref_p, cur_p)
                pred_drift = {
                    "available": True,
                    "psi": round(p_psi, 4),
                    "ks_p_value": round(float(p_ks_pval), 5),
                    "reference_mean_probability": round(float(np.mean(ref_p)), 3),
                    "current_mean_probability": round(float(np.mean(cur_p)), 3),
                    "reference_std": round(float(np.std(ref_p)), 3),
                    "current_std": round(float(np.std(cur_p)), 3),
                    "drift_status": "drift_detected" if p_psi >= 0.20 else "normal",
                }

        # 4. Overall Model Health & Retraining Recommendation
        if drifted_count >= 2 or (pred_drift.get("available") and pred_drift.get("psi", 0) >= 0.25):
            overall_status = "DRIFT_DETECTED"
            retrain_recommended = True
            retrain_reason = (
                f"{drifted_count} features exhibit severe distribution shift (PSI >= {self.psi_drift_threshold}). "
                "Retraining candidate model recommended."
            )
        elif drifted_count >= 1 or warning_count >= 3:
            overall_status = "WARNING"
            retrain_recommended = False
            retrain_reason = "Moderate feature drift observed. Monitor closely before triggering retraining."
        else:
            overall_status = "HEALTHY"
            retrain_recommended = False
            retrain_reason = "Distributions match reference baseline within expected statistical bounds."

        return {
            "health_status": overall_status,
            "retrain_recommended": retrain_recommended,
            "retrain_reason": retrain_reason,
            "summary": {
                "total_features_monitored": total_features,
                "features_with_drift": drifted_count,
                "features_with_warnings": warning_count,
                "features_normal": total_features - drifted_count - warning_count,
                "alerts_count": len(alerts),
            },
            "alerts": alerts,
            "features": feature_reports,
            "prediction_drift": pred_drift,
        }
