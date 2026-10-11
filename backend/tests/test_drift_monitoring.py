"""Tests for Feature 4: Drift Monitoring & Safe Model Updates."""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.db.models import Dataset, Organization, Project, TrainedModel
from app.services.drift_service import DriftMonitoringService, calculate_psi
from app.services.model_lifecycle_service import ModelLifecycleService


class TestDriftMonitoring:
    """Test suite for statistical drift detection and health scoring."""

    def test_psi_identical_distribution(self):
        """PSI on identical distributions should be near 0 (< 0.10)."""
        np.random.seed(42)
        ref = np.random.normal(loc=100.0, scale=15.0, size=1000)
        curr = np.random.normal(loc=100.0, scale=15.0, size=1000)

        psi = calculate_psi(ref, curr, num_bins=10)
        assert psi < 0.10

    def test_psi_shifted_distribution(self):
        """PSI on substantially shifted distributions should detect drift (> 0.25)."""
        np.random.seed(42)
        ref = np.random.normal(loc=100.0, scale=15.0, size=1000)
        curr = np.random.normal(loc=140.0, scale=15.0, size=1000)

        psi = calculate_psi(ref, curr, num_bins=10)
        assert psi > 0.25

    def test_drift_analysis_comprehensive(self):
        """Analyze multi-feature drift across numerical and categorical features."""
        np.random.seed(42)
        n = 500

        ref_df = pd.DataFrame({
            "age": np.random.randint(20, 60, size=n),
            "monthly_charges": np.random.normal(70, 10, size=n),
            "contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2]),
        })

        # Current df has drifted monthly_charges and shifted contract distribution
        curr_df = pd.DataFrame({
            "age": np.random.randint(20, 60, size=n),
            "monthly_charges": np.random.normal(110, 10, size=n),  # Significant drift
            "contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.9, 0.05, 0.05]),
        })

        service = DriftMonitoringService()
        report = service.analyze_drift(
            reference_df=ref_df,
            current_df=curr_df,
            feature_columns=["age", "monthly_charges", "contract"],
        )

        assert "health_status" in report
        assert "features" in report
        assert "retrain_recommended" in report

        # Map features list by feature name
        features_by_name = {f["feature"]: f for f in report["features"]}

        # Monthly charges should be flagged as drifted
        mc_report = features_by_name["monthly_charges"]
        assert mc_report["status"] == "drift_detected"
        assert mc_report["psi"] > 0.25

        # Retraining trigger check
        assert report["retrain_recommended"] is True
        assert len(report["retrain_reason"]) > 0

    def test_missingness_spike_detection(self):
        """Missing value spike should trigger anomaly alerts."""
        n = 200
        ref_df = pd.DataFrame({"score": np.random.normal(50, 5, size=n)})

        # Inject 30% NaNs in current
        curr_vals = np.random.normal(50, 5, size=n).astype(float)
        curr_vals[:60] = np.nan
        curr_df = pd.DataFrame({"score": curr_vals})

        service = DriftMonitoringService(null_spike_threshold=0.10)
        report = service.analyze_drift(
            reference_df=ref_df,
            current_df=curr_df,
            feature_columns=["score"],
        )

        features_by_name = {f["feature"]: f for f in report["features"]}
        score_res = features_by_name["score"]
        assert score_res["null_rate_change"] >= 0.25
        assert len(report["alerts"]) > 0


class TestModelLifecycle:
    """Test candidate evaluation, atomic promotion, rollback, and safety gating."""

    @pytest.fixture
    def test_db(self):
        """Create an isolated in-memory SQLite database with required foreign keys."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

        org = Organization(
            id="org_test",
            name="Test Org",
            slug="test-org",
        )
        db.add(org)

        dataset = Dataset(
            id="ds_test",
            organization_id="org_test",
            name="Test Dataset",
            file_path="/fake/path.csv",
            file_size=1024,
            row_count=100,
            columns=[{"name": "age", "dtype": "int"}, {"name": "churn", "dtype": "int"}],
        )
        db.add(dataset)

        proj = Project(
            id="proj_lifecycle_test",
            organization_id="org_test",
            dataset_id="ds_test",
            name="Lifecycle Test Project",
            problem_type="binary_classification",
            target_column="churn",
            target_positive_label="1",
            feature_columns=["age"],
            status="active",
        )
        db.add(proj)

        v1 = TrainedModel(
            id="mod_v1",
            project_id="proj_lifecycle_test",
            version="v1",
            model_path="/fake/path/v1",
            accuracy=0.82,
            auc_roc=0.85,
            f1_score=0.78,
            training_samples=1000,
            is_active=True,
        )
        db.add(v1)
        db.commit()

        yield db
        db.close()

    def test_candidate_acceptance_and_promotion(self, test_db):
        """Candidate satisfying acceptance criteria should promote atomically."""
        service = ModelLifecycleService(db=test_db, org_id="org_test")

        # Create a candidate model with good metrics (auc=0.86 vs active=0.85)
        cand = TrainedModel(
            id="mod_v2_cand",
            project_id="proj_lifecycle_test",
            version="v2-candidate",
            model_path="/fake/path/v2",
            accuracy=0.83,
            auc_roc=0.86,
            f1_score=0.80,
            training_samples=1200,
            is_active=False,
        )
        test_db.add(cand)
        test_db.commit()

        # Evaluate candidate
        eval_res = service.evaluate_candidate("proj_lifecycle_test", cand.id)
        assert eval_res["passes_acceptance_criteria"] is True
        assert eval_res["recommendation"] == "PROMOTE_TO_PRODUCTION"

        # Promote candidate
        promo_res = service.promote_candidate("proj_lifecycle_test", cand.id)
        assert promo_res["success"] is True
        assert promo_res["active_version"] == "v2"

        # Verify old active is deactivated and candidate is now active
        v1 = test_db.query(TrainedModel).filter(TrainedModel.id == "mod_v1").first()
        v2 = test_db.query(TrainedModel).filter(TrainedModel.id == "mod_v2_cand").first()
        assert v1.is_active is False
        assert v2.is_active is True
        assert v2.version == "v2"

    def test_candidate_rejection_on_degradation(self, test_db):
        """Candidate with degraded performance must be blocked unless force=True."""
        service = ModelLifecycleService(db=test_db, org_id="org_test")

        # Create a degraded candidate model (auc=0.70 vs active=0.85)
        degraded = TrainedModel(
            id="mod_bad_cand",
            project_id="proj_lifecycle_test",
            version="v2-candidate",
            model_path="/fake/path/bad",
            accuracy=0.70,
            auc_roc=0.70,
            f1_score=0.60,
            training_samples=1200,
            is_active=False,
        )
        test_db.add(degraded)
        test_db.commit()

        eval_res = service.evaluate_candidate("proj_lifecycle_test", degraded.id)
        assert eval_res["passes_acceptance_criteria"] is False
        assert eval_res["recommendation"] == "REJECT_CANDIDATE"

        # Promoting without force should raise ValueError
        with pytest.raises(ValueError, match="failed acceptance criteria"):
            service.promote_candidate("proj_lifecycle_test", degraded.id, force=False)

        # Active model remains v1
        v1 = test_db.query(TrainedModel).filter(TrainedModel.id == "mod_v1").first()
        assert v1.is_active is True

    def test_rollback_to_previous_model(self, test_db):
        """Rollback restores the previous active model."""
        service = ModelLifecycleService(db=test_db, org_id="org_test")

        # Add v2 as current active, v1 as previous inactive
        v1 = test_db.query(TrainedModel).filter(TrainedModel.id == "mod_v1").first()
        v1.is_active = False

        v2 = TrainedModel(
            id="mod_v2",
            project_id="proj_lifecycle_test",
            version="v2",
            model_path="/fake/path/v2",
            accuracy=0.83,
            auc_roc=0.86,
            training_samples=1200,
            is_active=True,
        )
        test_db.add(v2)
        test_db.commit()

        # Perform rollback
        rollback_res = service.rollback_model("proj_lifecycle_test")
        assert rollback_res["success"] is True
        assert rollback_res["active_version"] == "v1"

        # Verify database state
        test_db.refresh(v1)
        test_db.refresh(v2)
        assert v1.is_active is True
        assert v2.is_active is False
