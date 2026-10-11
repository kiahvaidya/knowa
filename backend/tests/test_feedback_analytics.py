"""Tests for Feature 5: End-to-End Decision Feedback Loop & Traceability."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.db.models import (
    Dataset,
    Decision,
    DecisionActivity,
    Organization,
    Project,
    ProjectPrediction,
)
from app.services.feedback_analytics_service import FeedbackAnalyticsService


class TestFeedbackAnalytics:
    """Test suite for decision lineage, outcome attribution, and verified feedback export."""

    @pytest.fixture
    def test_db(self):
        """Create an in-memory SQLite database populated with full decision lifecycle data."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

        org = Organization(id="org_feedback_test", name="Feedback Org", slug="fb-org")
        db.add(org)

        ds = Dataset(
            id="ds_fb_test",
            organization_id="org_feedback_test",
            name="Telco Base Dataset",
            file_path="/fake/telco.csv",
            file_size=2048,
            row_count=200,
            columns=[{"name": "tenure", "dtype": "int"}, {"name": "churn", "dtype": "int"}],
        )
        db.add(ds)

        proj = Project(
            id="proj_fb_test",
            organization_id="org_feedback_test",
            dataset_id="ds_fb_test",
            name="Feedback Test Project",
            target_column="churn",
            feature_columns=["tenure", "MonthlyCharges"],
            status="active",
        )
        db.add(proj)

        pred = ProjectPrediction(
            id="pred_fb_1",
            project_id="proj_fb_test",
            model_version="v1.0",
            entity_id="cust_101",
            features={"tenure": 3, "MonthlyCharges": 89.5},
            probability=0.78,
            risk_level="high",
            top_factors=[{"feature": "tenure", "impact": 0.35, "direction": "increases_risk"}],
        )
        db.add(pred)

        dec1 = Decision(
            id="dec_fb_1",
            organization_id="org_feedback_test",
            project_id="proj_fb_test",
            prediction_id="pred_fb_1",
            entity_id="cust_101",
            status="closed",
            action_code="annual_discount_15",
            action_name="15% Annual Retention Discount",
            probability_at_commit=0.78,
            risk_level_at_commit="high",
            expected_probability_after=0.45,
            expected_lift=-0.33,
            actual_outcome="retained",
            outcome_notes="Customer accepted 1-year contract extension.",
            checkin_count=2,
        )
        db.add(dec1)

        dec2 = Decision(
            id="dec_fb_2",
            organization_id="org_feedback_test",
            project_id="proj_fb_test",
            prediction_id="pred_fb_1",
            entity_id="cust_102",
            status="committed",
            action_code="annual_discount_15",
            action_name="15% Annual Retention Discount",
            probability_at_commit=0.65,
            expected_lift=-0.25,
            actual_outcome=None,  # still open
        )
        db.add(dec2)

        dec3 = Decision(
            id="dec_fb_3",
            organization_id="org_feedback_test",
            project_id="proj_fb_test",
            prediction_id="pred_fb_1",
            entity_id="cust_103",
            status="closed",
            action_code="courtesy_call",
            action_name="VIP Courtesy Outreach",
            probability_at_commit=0.82,
            expected_lift=-0.15,
            actual_outcome="churned",
            outcome_notes="Customer churned despite phone outreach.",
        )
        db.add(dec3)

        act = DecisionActivity(
            organization_id="org_feedback_test",
            decision_id="dec_fb_1",
            event="checked_in",
            payload={"note": "Recorded retention outcome"},
        )
        db.add(act)

        db.commit()
        yield db
        db.close()

    def test_decision_lineage_traceability(self, test_db):
        """Retrieve complete lineage and verify model version, predictions, and attribution caveat."""
        service = FeedbackAnalyticsService(test_db, "org_feedback_test")

        trace = service.get_decision_lineage("dec_fb_1")

        assert trace["decision_id"] == "dec_fb_1"
        assert trace["entity_id"] == "cust_101"
        assert trace["lineage"]["model_version"] == "v1.0"
        assert trace["lineage"]["dataset_name"] == "Telco Base Dataset"
        assert trace["initial_prediction_and_explanation"]["probability_at_commit"] == 0.78
        assert len(trace["initial_prediction_and_explanation"]["top_factors"]) > 0

        # Recommendation and outcome details
        rec = trace["recommendation_details"]
        assert rec["action_code"] == "annual_discount_15"
        assert rec["expected_lift"] == -0.33

        obs = trace["observed_outcome"]
        assert obs["recorded"] is True
        assert obs["actual_outcome"] == "retained"
        assert obs["observed_favorable"] is True
        assert obs["lift_discrepancy"] is not None

        # Causal disclaimer must be explicitly present
        assert "causal_attribution_disclaimer" in trace
        assert "cannot be conclusively attributed" in trace["causal_attribution_disclaimer"].lower()

    def test_action_effectiveness_aggregation(self, test_db):
        """Verify aggregated empirical lift, favorable rates, and status tags."""
        service = FeedbackAnalyticsService(test_db, "org_feedback_test")

        report = service.generate_effectiveness_report("proj_fb_test")

        assert report["total_decisions"] == 3
        assert report["overall_outcome_summary"]["total_completed_with_outcomes"] == 2
        assert report["overall_outcome_summary"]["total_favorable_outcomes"] == 1
        assert report["overall_outcome_summary"]["overall_favorable_rate"] == 0.5

        # Check action breakdown
        actions = {a["action_code"]: a for a in report["action_performance"]}
        assert "annual_discount_15" in actions
        assert "courtesy_call" in actions

        disc = actions["annual_discount_15"]
        assert disc["total_decisions"] == 2
        assert disc["completed_with_outcome"] == 1
        assert disc["favorable_outcomes"] == 1
        assert disc["favorable_rate"] == 1.0

        call = actions["courtesy_call"]
        assert call["total_decisions"] == 1
        assert call["completed_with_outcome"] == 1
        assert call["favorable_outcomes"] == 0
        assert call["favorable_rate"] == 0.0

    def test_export_verified_feedback_dataset(self, test_db):
        """Exporting feedback dataset should only include records with observed outcomes."""
        service = FeedbackAnalyticsService(test_db, "org_feedback_test")

        export_res = service.export_verified_feedback_dataset("proj_fb_test")

        assert export_res["project_id"] == "proj_fb_test"
        assert export_res["verified_records_count"] == 2  # dec1 and dec3 have actual_outcome

        record_ids = {r["decision_id"] for r in export_res["records"]}
        assert "dec_fb_1" in record_ids
        assert "dec_fb_3" in record_ids
        assert "dec_fb_2" not in record_ids  # dec2 was pending outcome

        # Verify feature columns attached
        rec1 = next(r for r in export_res["records"] if r["decision_id"] == "dec_fb_1")
        assert rec1["tenure"] == 3
        assert rec1["actual_outcome"] == "retained"
