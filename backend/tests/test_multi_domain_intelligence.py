"""Tests for Multi-Domain Intelligence in Decision Studio and Evaluation Lab."""

import numpy as np
import pandas as pd
import pytest

from app.recommendations.domains import (
    DOMAIN_ECOMMERCE,
    DOMAIN_FINANCE,
    DOMAIN_HEALTHCARE,
    DOMAIN_HR_ATTRITION,
    DOMAIN_SAAS,
    DOMAIN_TELCO,
    detect_domain,
)
from app.services.evaluation_service import EvaluationLabService
from app.services.optimizer_service import CandidateCase, DecisionOptimizer


def test_domain_detection_rules():
    """Verify domain detection across industry, text, and column feature markers."""
    # Organization industry
    assert detect_domain(organization_industry="Banking") == DOMAIN_FINANCE
    assert detect_domain(organization_industry="hospital") == DOMAIN_HEALTHCARE
    assert detect_domain(organization_industry="Workforce") == DOMAIN_HR_ATTRITION
    assert detect_domain(organization_industry="E-Commerce") == DOMAIN_ECOMMERCE
    assert detect_domain(organization_industry="Cloud Tech") == DOMAIN_SAAS

    # Metadata text keywords
    assert detect_domain(metadata_text="Employee turnover and resignation risk model") == DOMAIN_HR_ATTRITION
    assert detect_domain(metadata_text="Loan default probability scoring") == DOMAIN_FINANCE
    assert detect_domain(metadata_text="30-day diabetic readmission forecast") == DOMAIN_HEALTHCARE
    assert detect_domain(metadata_text="Shopping cart abandonment reduction") == DOMAIN_ECOMMERCE

    # Column feature markers
    assert detect_domain(features=["overtime", "yearsatcompany", "worklifebalance"]) == DOMAIN_HR_ATTRITION
    assert detect_domain(features={"creditscore": 720, "loanamount": 15000, "debttoincome": 0.28}) == DOMAIN_FINANCE
    assert detect_domain(features=["nummedications", "timeinhospital", "diagnoses"]) == DOMAIN_HEALTHCARE
    assert detect_domain(features=["cartvalue", "sessionduration", "itemsincart"]) == DOMAIN_ECOMMERCE
    assert detect_domain(features=["seatsused", "dailylogins", "mrr"]) == DOMAIN_SAAS


def test_decision_optimizer_for_hr_domain():
    """Verify optimizer dynamically selects HR actions and scales benefits appropriately."""
    optimizer = DecisionOptimizer(
        budget=10000.0,
        max_capacity=5,
        currency_symbol="₹",
        domain=DOMAIN_HR_ATTRITION,
    )

    candidates = [
        CandidateCase(
            entity_id=f"EMP_{i+1:03d}",
            features={
                "overtime": "Yes" if i % 2 == 0 else "No",
                "years_at_company": i + 2,
                "monthly_income": 4500.0 + (i * 300),
            },
            baseline_probability=0.45 + (i * 0.05),
            customer_value=50000.0,  # Value of retaining an employee
        )
        for i in range(10)
    ]

    res = optimizer.optimize(candidates, strategy="all")
    assert res["success"] is True
    assert res["domain"] == DOMAIN_HR_ATTRITION
    assert "Retained Employee" in res["unit_label"]

    knapsack = res["strategies"]["knapsack_optimal"]
    assert knapsack["total_cost"] <= 10000.0
    assert knapsack["interventions_count"] <= 5
    assert len(knapsack["allocated_actions"]) > 0

    # Ensure actions come from HR catalog
    from app.recommendations.action_catalog import HR_ATTRITION_CATALOG
    for a in knapsack["allocated_actions"]:
        assert a["action_code"] in HR_ATTRITION_CATALOG


def test_decision_optimizer_for_finance_domain():
    """Verify optimizer selects Finance actions for loan/account protection."""
    optimizer = DecisionOptimizer(
        budget=2000.0,
        max_capacity=5,
        currency_symbol="$",
        domain=DOMAIN_FINANCE,
    )

    candidates = [
        CandidateCase(
            entity_id=f"ACCT_{i+1:03d}",
            features={
                "credit_score": 580 + (i * 15),
                "loan_amount": 10000.0 + (i * 1000),
                "debt_to_income": 0.35 + (i * 0.02),
            },
            baseline_probability=0.35 + (i * 0.04),
            customer_value=12000.0,
        )
        for i in range(8)
    ]

    res = optimizer.optimize(candidates, strategy="knapsack")
    assert res["success"] is True
    assert res["domain"] == DOMAIN_FINANCE
    assert "Protected Account" in res["unit_label"]

    knapsack = res["strategies"]["knapsack_optimal"]
    assert knapsack["total_cost"] <= 2000.0
    assert len(knapsack["allocated_actions"]) > 0


def test_evaluation_lab_on_generic_dataset(tmp_path):
    """Verify EvaluationLab benchmarks arbitrary datasets with numeric 0/1 target without crashing."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "emp_id": [f"ID_{i}" for i in range(n)],
        "overtime": np.random.choice(["Yes", "No"], size=n),
        "job_satisfaction": np.random.randint(1, 5, size=n),
        "monthly_income": np.random.uniform(3000, 15000, size=n),
        "years_at_company": np.random.randint(1, 15, size=n),
        "attrition_numeric": np.random.choice([0, 1], size=n, p=[0.70, 0.30]),
    })

    service = EvaluationLabService(reports_dir=str(tmp_path))
    report = service.run_benchmark(
        df=df,
        target_column="attrition_numeric",
        positive_label="1",
        dataset_name="HR Workforce Retention Test",
        domain=DOMAIN_HR_ATTRITION,
        experiment_name="test_hr_bench",
    )

    assert report["dataset"]["name"] == "HR Workforce Retention Test"
    assert report["dataset"]["domain"] == DOMAIN_HR_ATTRITION
    assert report["dataset"]["target_column"] == "attrition_numeric"
    assert report["dataset"]["total_samples"] == n

    # Check models ran
    assert len(report["models"]) == 3
    for m in report["models"]:
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["brier_score"] <= 1.0

    # Uncertainty & calibration
    assert report["uncertainty"]["empirical_coverage"] > 0.0
    assert len(report["calibration_curves"]["ensemble"]) == 10
