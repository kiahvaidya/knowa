"""Tests for Actionable Counterfactual Explanation Engine."""

import numpy as np
import pandas as pd
import pytest

from app.ml.explainers.counterfactual import (
    CounterfactualExplainer,
    IMMUTABLE_FEATURES,
)


def _dummy_predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Predict higher risk if monthly charges are high or contract is month-to-month."""
    probs = []
    for _, row in df.iterrows():
        p = 0.3
        # Age should not affect dial if immutable, but let's say model sees it
        if float(row.get("age", 30)) > 60:
            p += 0.1
        # Controllable features
        mc = float(row.get("monthly_charges", 50))
        if mc > 80:
            p += 0.3
        elif mc < 40:
            p -= 0.15

        contract = str(row.get("contract_type", "month-to-month")).lower()
        if contract in ("month-to-month", "month_to_month"):
            p += 0.25
        elif contract in ("one_year", "two_year"):
            p -= 0.2

        tech = str(row.get("tech_support", "no")).lower()
        if tech == "yes":
            p -= 0.15

        probs.append(float(np.clip(p, 0.05, 0.95)))
    return np.array(probs)


class TestCounterfactualExplainer:
    """Test suite for Feature 1: Actionable Counterfactual Explanations."""

    def test_immutable_features_are_never_modified(self):
        """Verify immutable features such as age, gender, customer_id, tenure are not modified."""
        features = [
            "customer_id",
            "age",
            "gender",
            "tenure",
            "monthly_charges",
            "contract_type",
            "tech_support",
        ]
        ref_df = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "age": [25, 45, 65],
                "gender": ["M", "F", "F"],
                "tenure": [2, 12, 36],
                "monthly_charges": [40.0, 75.0, 95.0],
                "contract_type": ["month-to-month", "one_year", "two_year"],
                "tech_support": ["no", "yes", "no"],
            }
        )

        explainer = CounterfactualExplainer(
            predict_fn=_dummy_predict_proba,
            feature_names=features,
            reference_data=ref_df,
            risk_threshold=0.5,
        )

        test_row = pd.DataFrame(
            [
                {
                    "customer_id": "C999",
                    "age": 70,
                    "gender": "F",
                    "tenure": 3,
                    "monthly_charges": 90.0,
                    "contract_type": "month-to-month",
                    "tech_support": "no",
                }
            ]
        )

        res = explainer.generate_counterfactuals(test_row)
        assert res["success"] is True
        assert len(res["scenarios"]) > 0

        # Verify no scenario touches age, gender, customer_id, or tenure
        for sc in res["scenarios"]:
            for mod_k in sc["modified_features"].keys():
                assert mod_k not in IMMUTABLE_FEATURES, f"Immutable feature {mod_k} was modified!"
                assert mod_k not in ("age", "gender", "tenure", "customer_id")

    def test_validity_and_risk_reduction(self):
        """Verify counterfactual successfully reduces prediction below threshold."""
        features = ["monthly_charges", "contract_type", "tech_support"]
        ref_df = pd.DataFrame(
            {
                "monthly_charges": [30.0, 50.0, 90.0],
                "contract_type": ["month-to-month", "one_year", "two_year"],
                "tech_support": ["no", "yes", "yes"],
            }
        )

        explainer = CounterfactualExplainer(
            predict_fn=_dummy_predict_proba,
            feature_names=features,
            reference_data=ref_df,
            risk_threshold=0.50,
        )

        test_row = pd.DataFrame(
            [
                {
                    "monthly_charges": 92.0,
                    "contract_type": "month-to-month",
                    "tech_support": "no",
                }
            ]
        )

        res = explainer.generate_counterfactuals(test_row)
        assert res["has_feasible_counterfactual"] is True
        assert res["is_high_risk"] is True

        best = res["scenarios"][0]
        assert best["new_prediction"] < best["original_prediction"]
        assert best["probability_delta"] < 0
        assert best["sparsity"] in (1, 2, 3)
        assert best["feasibility_score"] > 0.8
        assert best["estimated_cost"] > 0
        assert "causal_disclaimer" in best

    def test_infeasible_counterfactual_communicated_gracefully(self):
        """When only immutable features drive risk, explainer communicates infeasibility."""
        def immutable_only_model(df: pd.DataFrame) -> np.ndarray:
            # Model prediction strictly depends on immutable age and tenure
            probs = []
            for _, r in df.iterrows():
                p = 0.85 if float(r.get("age", 30)) > 60 else 0.2
                probs.append(p)
            return np.array(probs)

        features = ["customer_id", "age", "tenure"]
        explainer = CounterfactualExplainer(
            predict_fn=immutable_only_model,
            feature_names=features,
            risk_threshold=0.50,
        )

        test_row = pd.DataFrame([{"customer_id": "C1", "age": 75, "tenure": 5}])
        res = explainer.generate_counterfactuals(test_row)
        assert res["success"] is True
        assert res["has_feasible_counterfactual"] is False
        assert "No feasible counterfactual found" in res["message"]
