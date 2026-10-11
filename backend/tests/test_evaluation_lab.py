"""Tests for Feature 3: Model Trust and Evaluation Lab."""

import os
import pytest

from app.services.evaluation_service import EvaluationLabService


class TestEvaluationLab:
    """Test suite for Model Trust & Evaluation Lab."""

    def test_benchmark_execution_and_metrics_integrity(self, tmp_path):
        """Run benchmark on benchmark dataset and verify mathematical metric ranges."""
        service = EvaluationLabService(reports_dir=str(tmp_path))

        report = service.run_benchmark(
            random_seed=42,
            experiment_name="test_experiment",
        )

        assert report["experiment_id"].startswith("eval_")
        assert report["dataset"]["name"] == "IBM Telco Customer Churn"
        assert report["dataset"]["test_samples"] > 0

        # Check models comparison
        models = {m["model_name"]: m for m in report["models"]}
        assert "Simple Heuristic Baseline (Class Prior)" in models
        assert "Logistic Regression (L2 Regularized)" in models
        assert "Knowa Stacked Ensemble (XGB + LGBM + RF)" in models

        # Check metric bounds for Ensemble
        ens = models["Knowa Stacked Ensemble (XGB + LGBM + RF)"]
        assert 0.0 <= ens["accuracy"] <= 1.0
        assert 0.0 <= ens["precision"] <= 1.0
        assert 0.0 <= ens["recall"] <= 1.0
        assert 0.0 <= ens["f1_score"] <= 1.0
        assert 0.5 <= ens["auc_roc"] <= 1.0  # trained model outperforms random chance
        assert 0.0 <= ens["pr_auc"] <= 1.0
        assert 0.0 <= ens["brier_score"] <= 1.0
        assert 0.0 <= ens["ece"] <= 0.5

        # Check uncertainty metrics
        unc = report["uncertainty"]
        assert unc["target_coverage"] == 0.90
        assert 0.70 <= unc["empirical_coverage"] <= 1.00
        assert unc["mean_interval_width"] > 0.0

        # Check calibration curve bins
        curves = report["calibration_curves"]["ensemble"]
        assert len(curves) == 10
        total_binned = sum(b["sample_count"] for b in curves)
        assert total_binned == report["dataset"]["test_samples"]

        # Check operational latency
        op = report["operational"]
        assert op["latency_p50_ms"] >= 0.0
        assert op["latency_p95_ms"] >= op["latency_p50_ms"]

    def test_report_listing_and_retrieval(self, tmp_path):
        """Verify reports are properly persisted and retrievable."""
        service = EvaluationLabService(reports_dir=str(tmp_path))
        report = service.run_benchmark(random_seed=42)

        reports_list = service.list_reports()
        assert len(reports_list) >= 1
        assert reports_list[0]["experiment_id"] == report["experiment_id"]
