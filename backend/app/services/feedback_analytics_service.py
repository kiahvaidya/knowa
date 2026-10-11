"""End-to-End Decision Feedback Loop and Traceability Analytics Service.

Provides complete audit lineage from prediction and explanation to action commitment,
implementation status, observed outcome, and empirical effectiveness analysis.
Explicitly distinguishes observational correlation from causal impact.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Decision, DecisionActivity, Project, ProjectPrediction, TrainedModel

logger = logging.getLogger(__name__)


class FeedbackAnalyticsService:
    """Analyzes recommendation lifecycle, outcome attribution, and feedback loop lineage."""

    def __init__(self, db: Session, org_id: str):
        self.db = db
        self.org_id = org_id

    def get_decision_lineage(self, decision_id: str) -> dict[str, Any]:
        """Retrieve complete end-to-end audit trace for a specific decision."""
        decision = (
            self.db.query(Decision)
            .filter(
                Decision.id == decision_id,
                Decision.organization_id == self.org_id,
            )
            .first()
        )
        if not decision:
            raise ValueError(f"Decision '{decision_id}' not found")

        project = self.db.query(Project).filter(Project.id == decision.project_id).first()
        prediction = None
        if decision.prediction_id:
            prediction = (
                self.db.query(ProjectPrediction)
                .filter(ProjectPrediction.id == decision.prediction_id)
                .first()
            )

        model_version = prediction.model_version if prediction else "Unknown"
        dataset_name = project.dataset.name if (project and project.dataset) else "Default Dataset"

        # Check difference between expected and observed results
        has_outcome = bool(decision.actual_outcome)
        observed_favorable = None
        lift_discrepancy = None

        if has_outcome:
            out_str = (decision.actual_outcome or "").lower()
            # Favorable if churn avoided (0, false, no, retained)
            favorable_indicators = {"0", "false", "no", "retained", "stayed", "active", "non-churn"}
            observed_favorable = any(ind in out_str for ind in favorable_indicators)

            # Compare expected lift (typically negative for risk reduction, e.g. -0.25)
            if decision.expected_lift is not None and decision.probability_at_commit is not None:
                exp_after = (
                    decision.expected_probability_after
                    if decision.expected_probability_after is not None
                    else max(0.0, decision.probability_at_commit + decision.expected_lift)
                )
                obs_val = 0.0 if observed_favorable else 1.0
                lift_discrepancy = round(obs_val - exp_after, 4)

        activities = (
            self.db.query(DecisionActivity)
            .filter(DecisionActivity.decision_id == decision.id)
            .order_by(DecisionActivity.created_at.asc())
            .all()
        )

        return {
            "decision_id": decision.id,
            "entity_id": decision.entity_id or "N/A",
            "project": {
                "id": project.id if project else "N/A",
                "name": project.name if project else "N/A",
                "target_column": project.target_column if project else "N/A",
            },
            "lineage": {
                "model_version": model_version,
                "dataset_name": dataset_name,
                "prediction_id": decision.prediction_id,
                "prediction_date": prediction.created_at.isoformat() if prediction else None,
            },
            "initial_prediction_and_explanation": {
                "probability_at_commit": decision.probability_at_commit,
                "risk_level_at_commit": decision.risk_level_at_commit,
                "top_factors": prediction.top_factors if prediction else (
                    decision.case_snapshot.get("drivers") if decision.case_snapshot else []
                ),
                "features_snapshot": prediction.features if prediction else (
                    decision.case_snapshot.get("features") if decision.case_snapshot else {}
                ),
            },
            "recommendation_details": {
                "action_code": decision.action_code,
                "action_name": decision.action_name,
                "action_description": decision.action_description,
                "expected_probability_after": decision.expected_probability_after,
                "expected_lift": decision.expected_lift,
                "decision_summary": decision.decision_summary,
                "assumptions": (
                    "Assumes average intervention adherence under business-as-usual conditions. "
                    "Expected impact is an observational estimate, not an experimentally proven causal guarantee."
                ),
            },
            "implementation_status": {
                "status": decision.status,  # proposed, committed, checking, closed, cancelled
                "committed_at": decision.committed_at.isoformat() if decision.committed_at else None,
                "recheck_interval_days": decision.recheck_interval_days,
                "recheck_scheduled_at": decision.recheck_at.isoformat() if decision.recheck_at else None,
                "last_checkin_at": decision.last_checkin_at.isoformat() if decision.last_checkin_at else None,
                "checkin_count": decision.checkin_count,
            },
            "observed_outcome": {
                "recorded": has_outcome,
                "actual_outcome": decision.actual_outcome,
                "observed_favorable": observed_favorable,
                "outcome_notes": decision.outcome_notes,
                "autopsy_narrative": decision.autopsy_narrative,
                "lift_discrepancy": lift_discrepancy,
                "discrepancy_interpretation": (
                    "Observed outcome matched or exceeded expected risk reduction"
                    if lift_discrepancy is not None and lift_discrepancy <= 0
                    else "Observed outcome fell short of predicted risk reduction or event still occurred"
                    if lift_discrepancy is not None
                    else "Pending follow-up observation"
                ),
            },
            "audit_trail": [
                {
                    "event": a.event,
                    "payload": a.payload,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in activities
            ],
            "causal_attribution_disclaimer": (
                "CRITICAL NOTE: Observed post-intervention outcomes reflect real-world status but cannot be "
                "conclusively attributed to the intervention without a randomized A/B control or formal causal "
                "identification. External market trends, customer self-selection, and delayed effects may be confounding."
            ),
        }

    def generate_effectiveness_report(self, project_id: Optional[str] = None) -> dict[str, Any]:
        """Aggregate analysis of recommendation adoption, outcome accuracy, and empirical lift."""
        q = self.db.query(Decision).filter(Decision.organization_id == self.org_id)
        if project_id:
            q = q.filter(Decision.project_id == project_id)

        decisions = q.all()

        total_decisions = len(decisions)
        if total_decisions == 0:
            return {
                "total_decisions": 0,
                "summary": "No decisions recorded in ledger yet.",
                "action_performance": [],
                "discrepancy_summary": {},
            }

        status_counts = {"committed": 0, "checking": 0, "closed": 0, "cancelled": 0, "proposed": 0}
        actions_map: dict[str, dict[str, Any]] = {}

        for d in decisions:
            st = d.status if d.status in status_counts else "committed"
            status_counts[st] += 1

            code = d.action_code or "unknown"
            if code not in actions_map:
                actions_map[code] = {
                    "action_code": code,
                    "action_name": d.action_name or code,
                    "total_count": 0,
                    "closed_with_outcome": 0,
                    "favorable_outcomes": 0,
                    "adverse_outcomes": 0,
                    "expected_lifts": [],
                    "observed_favorable_rate": 0.0,
                    "average_expected_lift": 0.0,
                    "empirical_vs_expected": "Pending Data",
                }

            entry = actions_map[code]
            entry["total_count"] += 1
            if d.expected_lift is not None:
                entry["expected_lifts"].append(float(d.expected_lift))

            if d.actual_outcome:
                entry["closed_with_outcome"] += 1
                out_str = str(d.actual_outcome).lower()
                favorable_indicators = {"0", "false", "no", "retained", "stayed", "active", "non-churn"}
                if any(ind in out_str for ind in favorable_indicators):
                    entry["favorable_outcomes"] += 1
                else:
                    entry["adverse_outcomes"] += 1

        action_performance = []
        for code, entry in actions_map.items():
            n_outcomes = entry["closed_with_outcome"]
            fav_rate = (entry["favorable_outcomes"] / n_outcomes) if n_outcomes > 0 else 0.0
            avg_exp_lift = float(np.mean(entry["expected_lifts"])) if entry["expected_lifts"] else 0.0

            # Compare empirical favorable rate
            if n_outcomes >= 3:
                # If negative lift = risk down, favorable rate should be high (> 0.6)
                if fav_rate >= 0.70:
                    status_tag = "HIGH_PERFORMING"
                elif fav_rate >= 0.40:
                    status_tag = "MODERATE_IMPACT"
                else:
                    status_tag = "UNDERPERFORMING"
            else:
                status_tag = "INSUFFICIENT_OUTCOMES"

            action_performance.append({
                "action_code": code,
                "action_name": entry["action_name"],
                "total_decisions": entry["total_count"],
                "completed_with_outcome": n_outcomes,
                "favorable_outcomes": entry["favorable_outcomes"],
                "favorable_rate": round(fav_rate, 3),
                "avg_expected_lift": round(avg_exp_lift, 3),
                "status_tag": status_tag,
            })

        completed_total = sum(a["completed_with_outcome"] for a in action_performance)
        favorable_total = sum(a["favorable_outcomes"] for a in action_performance)
        overall_fav_rate = round(favorable_total / completed_total, 3) if completed_total > 0 else 0.0

        return {
            "total_decisions": total_decisions,
            "status_breakdown": status_counts,
            "overall_outcome_summary": {
                "total_completed_with_outcomes": completed_total,
                "total_favorable_outcomes": favorable_total,
                "overall_favorable_rate": overall_fav_rate,
                "pending_follow_ups": status_counts["checking"] + status_counts["committed"],
            },
            "action_performance": action_performance,
            "attribution_guidance": (
                "Action performance metrics illustrate empirical outcome distributions. "
                "Because business environments are dynamic, high favorable rates may be partly "
                "attributable to general retention baseline, seasonality, or customer self-selection."
            ),
        }

    def export_verified_feedback_dataset(self, project_id: str) -> dict[str, Any]:
        """Export verified outcome cases as a clean labeled dataset for model retraining and validation."""
        decisions = (
            self.db.query(Decision)
            .filter(
                Decision.project_id == project_id,
                Decision.organization_id == self.org_id,
                Decision.actual_outcome.isnot(None),
            )
            .all()
        )

        records = []
        for d in decisions:
            pred = (
                self.db.query(ProjectPrediction)
                .filter(ProjectPrediction.id == d.prediction_id)
                .first()
                if d.prediction_id
                else None
            )
            feats = pred.features if pred else (d.case_snapshot.get("features") if d.case_snapshot else {})

            records.append({
                "decision_id": d.id,
                "entity_id": d.entity_id,
                "action_taken": d.action_code,
                "actual_outcome": d.actual_outcome,
                "recheck_date": d.last_checkin_at.isoformat() if d.last_checkin_at else d.committed_at.isoformat(),
                **feats,
            })

        return {
            "project_id": project_id,
            "verified_records_count": len(records),
            "records": records,
            "ready_for_retraining": len(records) >= 50,
            "guidance": (
                f"{len(records)} verified feedback examples available. "
                "When >= 50 verified examples are collected, they can be folded into candidate model retraining datasets."
            ),
        }
