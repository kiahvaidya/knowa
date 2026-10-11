"""Actionable Counterfactual Explanation Engine.

Identifies minimal, feasible, constraint-respecting interventions on controllable features
that alter the model's prediction, avoiding immutable attributes and respecting valid domains.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from app.insights.feature_mapping import (
    FEATURE_MAPPING,
    get_feature_info,
    resolve_feature_key,
)
from app.ml.blindspot import _HARD_LOW_CANONICAL, _CONTEXT_CANONICAL, _intervenability
from app.recommendations.action_catalog import ACTION_CATALOG

logger = logging.getLogger(__name__)

# Absolute immutable columns that should never be proposed as counterfactual levers
IMMUTABLE_FEATURES = {
    *_HARD_LOW_CANONICAL,
    *_CONTEXT_CANONICAL,
    "customerid",
    "customer_id",
    "entity_id",
    "id",
    "account_id",
    "user_id",
    "churn",
    "target",
    "label",
}


@dataclass
class FeatureConstraint:
    """Constraints for a single feature during counterfactual search."""

    name: str
    is_controllable: bool = True
    data_type: str = "numeric"  # "numeric" | "categorical" | "ordinal"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_categories: list[str] = field(default_factory=list)
    step_size: Optional[float] = None
    cost_weight: float = 1.0  # difficulty / cost of changing this feature
    preferred_direction: Optional[str] = None  # "increase" | "decrease" | None


@dataclass
class CounterfactualScenario:
    """A single proposed counterfactual intervention scenario."""

    scenario_id: str
    title: str
    description: str
    modified_features: dict[str, Any]
    feature_changes: list[dict[str, Any]]
    original_prediction: float
    new_prediction: float
    probability_delta: float
    sparsity: int
    normalized_distance: float
    estimated_cost: float
    estimated_benefit: float
    feasibility_score: float
    is_valid: bool
    assumptions: list[str]
    limitations: list[str]
    causal_disclaimer: str


class CounterfactualExplainer:
    """Finds actionable, constraint-respecting counterfactual explanations.

    Given a trained model and a high-risk instance, discovers minimal changes to
    controllable features that lower predicted risk while strictly adhering to:
      1. Immutability constraints (age, gender, tenure, IDs cannot change)
      2. Valid numerical bounds and categorical value spaces
      3. Actionable business directions (e.g. discounts lower monthly charges)
    """

    def __init__(
        self,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        feature_names: Sequence[str],
        reference_data: Optional[pd.DataFrame] = None,
        feature_config: Optional[dict[str, Any]] = None,
        risk_threshold: float = 0.50,
        target_risk: float = 0.35,
    ):
        """Initialize explainer.

        Args:
            predict_fn: Callable that accepts a DataFrame and returns 1D probability array.
            feature_names: List of model feature columns.
            reference_data: Optional background/training dataset to infer min/max/categories.
            feature_config: Optional schema definitions from project metadata.
            risk_threshold: Decision boundary above which counterfactuals are sought.
            target_risk: Target probability for a successful counterfactual.
        """
        self.predict_fn = predict_fn
        self.feature_names = list(feature_names)
        self.reference_data = reference_data
        self.feature_config = feature_config or {}
        self.risk_threshold = float(risk_threshold)
        self.target_risk = float(target_risk)
        self.constraints: dict[str, FeatureConstraint] = self._build_constraints()

    def _build_constraints(self) -> dict[str, FeatureConstraint]:
        """Derive constraints from data, schema config, and domain blindspots."""
        constraints: dict[str, FeatureConstraint] = {}

        for feat in self.feature_names:
            norm_name = feat.lower().strip()
            level, reason = _intervenability(feat, self.feature_config)
            is_controllable = (
                level in ("high", "medium")
                and norm_name not in IMMUTABLE_FEATURES
                and not norm_name.endswith("_missing")
                and not norm_name.endswith("_is_missing")
            )

            # Determine type, bounds, and categories
            data_type = "numeric"
            min_val = None
            max_val = None
            allowed_cats: list[str] = []
            step = None

            if self.reference_data is not None and feat in self.reference_data.columns:
                series = self.reference_data[feat].dropna()
                if pd.api.types.is_numeric_dtype(series):
                    data_type = "numeric"
                    min_val = float(series.min()) if not series.empty else 0.0
                    max_val = float(series.max()) if not series.empty else 100.0
                    # Check if integer-like
                    if (series % 1 == 0).all():
                        step = 1.0
                else:
                    data_type = "categorical"
                    allowed_cats = [str(c) for c in series.unique()]
            else:
                # Fallback inferences from feature name
                if any(k in norm_name for k in ("contract", "payment", "internet", "security", "support", "streaming")):
                    data_type = "categorical"
                    if "contract" in norm_name:
                        allowed_cats = ["month-to-month", "one_year", "two_year"]
                    elif "security" in norm_name or "support" in norm_name:
                        allowed_cats = ["no", "yes"]
                else:
                    data_type = "numeric"
                    if "charge" in norm_name or "mrr" in norm_name:
                        min_val, max_val = 15.0, 150.0
                    elif "score" in norm_name or "satisfaction" in norm_name:
                        min_val, max_val = 1.0, 5.0
                        step = 1.0
                    elif "complaint" in norm_name or "ticket" in norm_name:
                        min_val, max_val = 0.0, 20.0
                        step = 1.0

            # Override from explicit feature_config if supplied
            cfg = self.feature_config.get(feat, {})
            if isinstance(cfg, dict):
                if cfg.get("is_controllable") is not None:
                    is_controllable = bool(cfg["is_controllable"])
                if cfg.get("min") is not None:
                    min_val = float(cfg["min"])
                if cfg.get("max") is not None:
                    max_val = float(cfg["max"])
                if cfg.get("categories"):
                    allowed_cats = [str(c) for c in cfg["categories"]]
                    data_type = "categorical"

            # Cost weights
            cost_weight = 1.0
            if "contract" in norm_name:
                cost_weight = 1.2
            elif "charge" in norm_name:
                cost_weight = 1.5
            elif "support" in norm_name or "security" in norm_name:
                cost_weight = 0.8

            constraints[feat] = FeatureConstraint(
                name=feat,
                is_controllable=is_controllable,
                data_type=data_type,
                min_value=min_val,
                max_value=max_val,
                allowed_categories=allowed_cats,
                step_size=step,
                cost_weight=cost_weight,
            )

        return constraints

    def generate_counterfactuals(
        self,
        instance_df: pd.DataFrame,
        raw_features: Optional[dict[str, Any]] = None,
        top_drivers: Optional[list[dict[str, Any]]] = None,
        max_scenarios: int = 4,
    ) -> dict[str, Any]:
        """Find feasible counterfactual recommendations for instance.

        Returns:
            Dictionary containing list of CounterfactualScenarios, feature controllability
            summary, feasibility checks, and assumptions.
        """
        # Baseline prediction
        try:
            preds = self.predict_fn(instance_df)
            orig_prob = float(preds[0])
        except Exception as e:
            logger.error("Error predicting baseline in counterfactual search: %s", e)
            return {
                "success": False,
                "error": f"Failed to compute baseline prediction: {e}",
                "scenarios": [],
            }

        # If already low risk, no urgent counterfactual needed, but provide proactive maintenance
        is_high_risk = orig_prob >= self.risk_threshold

        controllable_features = [
            f for f, c in self.constraints.items() if c.is_controllable and f in instance_df.columns
        ]
        immutable_features = [
            f for f in self.feature_names if f not in controllable_features
        ]

        # Prioritize search using top drivers if provided
        search_order: list[str] = []
        if top_drivers:
            for d in top_drivers:
                fn = d.get("feature")
                if fn in controllable_features and fn not in search_order:
                    search_order.append(fn)
        for f in controllable_features:
            if f not in search_order:
                search_order.append(f)

        scenarios: list[CounterfactualScenario] = []

        # 1. Single-lever candidate search (Sparsity = 1)
        for feat in search_order:
            candidate = self._search_single_feature(instance_df, feat, orig_prob)
            if candidate and candidate.is_valid:
                scenarios.append(candidate)
            if len(scenarios) >= max_scenarios:
                break

        # 2. Multi-lever combined bundle (Sparsity = 2 or 3) if single didn't find enough
        if len(scenarios) < max_scenarios and len(search_order) >= 2:
            bundle_scenarios = self._search_bundle_features(
                instance_df, search_order[:4], orig_prob
            )
            for bs in bundle_scenarios:
                if bs.is_valid and not any(
                    s.modified_features == bs.modified_features for s in scenarios
                ):
                    scenarios.append(bs)
                if len(scenarios) >= max_scenarios:
                    break

        # Sort scenarios by: validity first, then lowest new_prediction, then lowest cost
        scenarios.sort(key=lambda s: (not s.is_valid, s.new_prediction, s.estimated_cost))

        formatted_scenarios = [self._format_scenario(s) for s in scenarios[:max_scenarios]]

        has_feasible = len(formatted_scenarios) > 0 and any(
            s["is_valid"] for s in formatted_scenarios
        )

        message = (
            f"Generated {len(formatted_scenarios)} actionable counterfactual scenario(s)."
            if has_feasible
            else (
                "No feasible counterfactual found within allowable business bounds. "
                "Risk is primarily driven by immutable context (such as tenure or history) "
                "that cannot be altered through near-term operational dials."
            )
        )

        return {
            "success": True,
            "has_feasible_counterfactual": has_feasible,
            "message": message,
            "original_probability": orig_prob,
            "risk_threshold": self.risk_threshold,
            "target_risk": self.target_risk,
            "is_high_risk": is_high_risk,
            "controllable_features": [
                {
                    "name": f,
                    "display_name": get_feature_info(f).get("display_name", f.replace("_", " ").title()),
                    "type": self.constraints[f].data_type,
                    "bounds": {
                        "min": self.constraints[f].min_value,
                        "max": self.constraints[f].max_value,
                    } if self.constraints[f].data_type == "numeric" else None,
                    "categories": self.constraints[f].allowed_categories or None,
                }
                for f in controllable_features[:8]
            ],
            "immutable_features": [
                {
                    "name": f,
                    "display_name": get_feature_info(f).get("display_name", f.replace("_", " ").title()),
                    "reason": "Context / immutable attribute that cannot be altered as an action dial",
                }
                for f in immutable_features[:8]
            ],
            "scenarios": formatted_scenarios,
            "governance": {
                "immutability_enforced": True,
                "domain_bounds_enforced": True,
                "causal_disclaimer": (
                    "Counterfactual recommendations represent mathematical sensitivity on the "
                    "trained predictive model surface. They reflect associative patterns in "
                    "historical data and do not guarantee causal impact without controlled trials."
                ),
            },
        }

    def _search_single_feature(
        self,
        instance_df: pd.DataFrame,
        feature: str,
        orig_prob: float,
    ) -> Optional[CounterfactualScenario]:
        """Test candidate interventions on a single feature."""
        constraint = self.constraints[feature]
        orig_val = instance_df[feature].iloc[0]

        best_scenario: Optional[CounterfactualScenario] = None
        min_p = orig_prob

        # Categorical choices
        if constraint.data_type == "categorical":
            candidates = [c for c in constraint.allowed_categories if str(c) != str(orig_val)]
            if not candidates:
                # Default binary toggles
                if str(orig_val).lower() in ("0", "no", "false"):
                    candidates = ["1", "yes", "true"]
                elif str(orig_val).lower() in ("1", "yes", "true"):
                    candidates = ["0", "no", "false"]

            for val in candidates:
                test_df = instance_df.copy()
                test_df[feature] = val
                try:
                    p = float(self.predict_fn(test_df)[0])
                except Exception:
                    continue

                if p < min_p:
                    min_p = p
                    delta = p - orig_prob
                    cost = self._estimate_cost({feature: val}, orig_val)
                    benefit = max(0.0, -delta) * 1000.0  # illustrative $ value
                    is_valid = p <= self.risk_threshold or delta <= -0.10

                    best_scenario = CounterfactualScenario(
                        scenario_id=f"cf_single_{feature}",
                        title=f"Adjust {get_feature_info(feature).get('display_name', feature)}",
                        description=f"Change {feature} from {orig_val} to {val}.",
                        modified_features={feature: val},
                        feature_changes=[
                            {
                                "feature": feature,
                                "original_value": orig_val,
                                "proposed_value": val,
                                "direction": "switch",
                            }
                        ],
                        original_prediction=orig_prob,
                        new_prediction=p,
                        probability_delta=delta,
                        sparsity=1,
                        normalized_distance=0.25,
                        estimated_cost=cost,
                        estimated_benefit=benefit,
                        feasibility_score=1.0,
                        is_valid=is_valid,
                        assumptions=[
                            f"Customer is eligible and receptive to changing {feature} to {val}.",
                            "Other account usage habits remain constant during transition.",
                        ],
                        limitations=[
                            "Model prediction relies on observational correlation.",
                        ],
                        causal_disclaimer=(
                            "Sensitivity projection only; does not guarantee customer retention."
                        ),
                    )

        # Numeric choices
        elif constraint.data_type == "numeric":
            curr = float(orig_val) if not pd.isna(orig_val) else 0.0
            min_v = constraint.min_value if constraint.min_value is not None else curr * 0.5
            max_v = constraint.max_value if constraint.max_value is not None else curr * 1.5

            # Test candidate adjustments: -10%, -20%, -30% for charges/complaints, or +1/+2 for satisfaction
            norm_name = feature.lower()
            if any(k in norm_name for k in ("charge", "mrr", "cost", "complaint", "ticket", "day")):
                # Lowering is favorable
                factors = [0.85, 0.75, 0.65, 0.50]
                candidates = [max(min_v, curr * f) for f in factors if curr * f < curr]
            else:
                # Increasing is favorable
                factors = [1.15, 1.30, 1.50]
                candidates = [min(max_v, curr * f) for f in factors if curr * f > curr]

            if constraint.step_size and constraint.step_size >= 1.0:
                candidates = [round(c) for c in candidates]

            for val in set(candidates):
                if val == curr:
                    continue
                test_df = instance_df.copy()
                test_df[feature] = val
                try:
                    p = float(self.predict_fn(test_df)[0])
                except Exception:
                    continue

                if p < min_p:
                    min_p = p
                    delta = p - orig_prob
                    cost = self._estimate_cost({feature: val}, curr)
                    benefit = max(0.0, -delta) * 1000.0
                    is_valid = p <= self.risk_threshold or delta <= -0.08

                    change_dir = "decrease" if val < curr else "increase"
                    best_scenario = CounterfactualScenario(
                        scenario_id=f"cf_single_{feature}",
                        title=f"{'Reduce' if val < curr else 'Increase'} {get_feature_info(feature).get('display_name', feature)}",
                        description=f"Modify {feature} from {curr:.2f} to {val:.2f}.",
                        modified_features={feature: val},
                        feature_changes=[
                            {
                                "feature": feature,
                                "original_value": round(curr, 2),
                                "proposed_value": round(val, 2),
                                "direction": change_dir,
                            }
                        ],
                        original_prediction=orig_prob,
                        new_prediction=p,
                        probability_delta=delta,
                        sparsity=1,
                        normalized_distance=abs(val - curr) / max(1.0, max_v - min_v),
                        estimated_cost=cost,
                        estimated_benefit=benefit,
                        feasibility_score=1.0,
                        is_valid=is_valid,
                        assumptions=[
                            f"Proposed change of {abs(val - curr):.2f} in {feature} is commercially practical.",
                        ],
                        limitations=[
                            "Elasticity may vary across customer segments.",
                        ],
                        causal_disclaimer=(
                            "Estimated based on sensitivity in training data."
                        ),
                    )

        return best_scenario

    def _search_bundle_features(
        self,
        instance_df: pd.DataFrame,
        features: list[str],
        orig_prob: float,
    ) -> list[CounterfactualScenario]:
        """Search multi-lever combinations (bundle of 2-3 features)."""
        scenarios: list[CounterfactualScenario] = []
        if len(features) < 2:
            return scenarios

        # Take pair of top two features
        pair = features[:2]
        mods: dict[str, Any] = {}
        changes: list[dict[str, Any]] = []

        test_df = instance_df.copy()
        for f in pair:
            c = self.constraints[f]
            orig_val = instance_df[f].iloc[0]
            if c.data_type == "categorical":
                target_val = (
                    c.allowed_categories[1]
                    if len(c.allowed_categories) > 1 and str(orig_val) == str(c.allowed_categories[0])
                    else (c.allowed_categories[0] if c.allowed_categories else "yes")
                )
                test_df[f] = target_val
                mods[f] = target_val
                changes.append(
                    {
                        "feature": f,
                        "original_value": orig_val,
                        "proposed_value": target_val,
                        "direction": "switch",
                    }
                )
            else:
                curr = float(orig_val) if not pd.isna(orig_val) else 0.0
                min_v = c.min_value if c.min_value is not None else curr * 0.5
                target_val = max(min_v, round(curr * 0.8, 2))
                test_df[f] = target_val
                mods[f] = target_val
                changes.append(
                    {
                        "feature": f,
                        "original_value": round(curr, 2),
                        "proposed_value": target_val,
                        "direction": "decrease",
                    }
                )

        try:
            p = float(self.predict_fn(test_df)[0])
        except Exception:
            return scenarios

        if p < orig_prob - 0.05:
            delta = p - orig_prob
            cost = sum(self._estimate_cost({f: mods[f]}, instance_df[f].iloc[0]) for f in pair)
            benefit = max(0.0, -delta) * 1200.0

            scenarios.append(
                CounterfactualScenario(
                    scenario_id="cf_bundle_strategic",
                    title="Comprehensive Retention Bundle",
                    description=f"Simultaneously address {', '.join(pair)}.",
                    modified_features=mods,
                    feature_changes=changes,
                    original_prediction=orig_prob,
                    new_prediction=p,
                    probability_delta=delta,
                    sparsity=len(pair),
                    normalized_distance=0.45,
                    estimated_cost=round(cost, 2),
                    estimated_benefit=round(benefit, 2),
                    feasibility_score=0.92,
                    is_valid=True,
                    assumptions=[
                        "Joint intervention has compounding positive effect.",
                        "Team has operational capacity to deliver both interventions.",
                    ],
                    limitations=[
                        "Higher execution friction when applying multi-lever packages.",
                    ],
                    causal_disclaimer=(
                        "Non-causal predictive estimate based on multivariate joint model response."
                    ),
                )
            )

        return scenarios

    def _estimate_cost(self, modifications: dict[str, Any], orig_val: Any) -> float:
        """Estimate representative intervention cost in illustrative currency ($)."""
        cost = 50.0  # base operational outreach cost
        for feat, val in modifications.items():
            norm_name = feat.lower()
            if "charge" in norm_name or "mrr" in norm_name:
                try:
                    diff = float(orig_val) - float(val)
                    cost += max(0.0, diff * 6.0)  # 6-month discount allowance
                except Exception:
                    cost += 80.0
            elif "contract" in norm_name:
                cost += 60.0  # retention incentive bonus
            elif "support" in norm_name or "security" in norm_name:
                cost += 40.0  # setup & onboarding allowance
            else:
                cost += 35.0
        return round(cost, 2)

    def _format_scenario(self, scenario: CounterfactualScenario) -> dict[str, Any]:
        """Serialize scenario for JSON API consumption."""
        return {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "description": scenario.description,
            "modified_features": scenario.modified_features,
            "feature_changes": scenario.feature_changes,
            "original_prediction": round(scenario.original_prediction, 4),
            "new_prediction": round(scenario.new_prediction, 4),
            "probability_delta": round(scenario.probability_delta, 4),
            "risk_reduction_pct": round(
                (
                    (scenario.original_prediction - scenario.new_prediction)
                    / max(0.01, scenario.original_prediction)
                )
                * 100,
                1,
            ),
            "sparsity": scenario.sparsity,
            "normalized_distance": round(scenario.normalized_distance, 3),
            "estimated_cost": scenario.estimated_cost,
            "estimated_benefit": scenario.estimated_benefit,
            "net_benefit": round(scenario.estimated_benefit - scenario.estimated_cost, 2),
            "feasibility_score": scenario.feasibility_score,
            "is_valid": scenario.is_valid,
            "assumptions": scenario.assumptions,
            "limitations": scenario.limitations,
            "causal_disclaimer": scenario.causal_disclaimer,
        }
