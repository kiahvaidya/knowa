"""Tests for Feature 2: Constraint-Aware Business Decision Optimizer."""

import pytest

from app.services.optimizer_service import (
    CandidateCase,
    DecisionOptimizer,
)


class TestDecisionOptimizer:
    """Test suite for Decision Optimizer under budget and capacity constraints."""

    @pytest.fixture
    def sample_candidates(self):
        """Create a diverse candidate cohort."""
        candidates = []
        for i in range(20):
            # Varying risk, charges, confidence
            prob = 0.30 + (i * 0.03)  # 0.30 to 0.87
            mc = 40.0 + (i * 4.0)     # 40 to 116
            low_conf = (i % 5 == 0)
            candidates.append(
                CandidateCase(
                    entity_id=f"CUST_{i+1:03d}",
                    features={
                        "monthly_charges": mc,
                        "contract_type": "month-to-month" if i % 2 == 0 else "one_year",
                        "tenure": i + 2,
                    },
                    baseline_probability=prob,
                    risk_level="high" if prob >= 0.6 else "medium",
                    low_confidence=low_conf,
                    customer_value=720.0,
                )
            )
        return candidates

    def test_budget_constraint_never_exceeded(self, sample_candidates):
        """Verify total cost never exceeds configured budget across all strategies."""
        budget_limit = 500.0
        capacity_limit = 15

        optimizer = DecisionOptimizer(budget=budget_limit, max_capacity=capacity_limit)
        res = optimizer.optimize(sample_candidates, strategy="all")

        assert res["success"] is True
        for name, strat in res["strategies"].items():
            cost = strat["total_cost"]
            assert cost <= budget_limit, f"Strategy {name} exceeded budget: {cost} > {budget_limit}"

    def test_capacity_constraint_never_exceeded(self, sample_candidates):
        """Verify number of interventions never exceeds configured maximum capacity."""
        budget_limit = 5000.0  # high budget so capacity binds
        capacity_limit = 5

        optimizer = DecisionOptimizer(budget=budget_limit, max_capacity=capacity_limit)
        res = optimizer.optimize(sample_candidates, strategy="all")

        assert res["success"] is True
        for name, strat in res["strategies"].items():
            count = strat["interventions_count"]
            assert count <= capacity_limit, f"Strategy {name} exceeded capacity: {count} > {capacity_limit}"

    def test_unique_action_per_customer(self, sample_candidates):
        """Ensure no individual customer receives multiple conflicting actions."""
        optimizer = DecisionOptimizer(budget=2000.0, max_capacity=20)
        res = optimizer.optimize(sample_candidates, strategy="all")

        for name, strat in res["strategies"].items():
            if name == "no_action":
                continue
            entity_ids = [a["entity_id"] for a in strat["allocated_actions"]]
            assert len(entity_ids) == len(set(entity_ids)), f"Duplicate customer in {name}"

    def test_knapsack_optimal_yields_maximal_or_equal_net_benefit(self, sample_candidates):
        """Verify integer programming finds high net benefit."""
        optimizer = DecisionOptimizer(budget=1000.0, max_capacity=10)
        res = optimizer.optimize(sample_candidates, strategy="all")

        knapsack_net = res["strategies"]["knapsack_optimal"]["total_net_benefit"]
        greedy_net = res["strategies"]["greedy_roi"]["total_net_benefit"]
        no_action_net = res["strategies"]["no_action"]["total_net_benefit"]

        assert knapsack_net >= 0.0
        assert knapsack_net >= no_action_net
        # Knapsack should find at least as good as greedy or within rounding
        assert knapsack_net >= greedy_net - 1e-4

    def test_binding_constraints_attribution(self, sample_candidates):
        """Verify optimizer explains when budget or capacity constraints are binding."""
        # Tight budget
        opt_tight_budget = DecisionOptimizer(budget=120.0, max_capacity=20)
        res_budget = opt_tight_budget.optimize(sample_candidates, strategy="knapsack")
        knap = res_budget["strategies"]["knapsack_optimal"]
        assert any("Budget constraint binding" in b for b in knap["binding_constraints"])

        # Tight capacity
        opt_tight_cap = DecisionOptimizer(budget=10000.0, max_capacity=2)
        res_cap = opt_tight_cap.optimize(sample_candidates, strategy="knapsack")
        knap_cap = res_cap["strategies"]["knapsack_optimal"]
        assert any("Intervention capacity limit reached" in b for b in knap_cap["binding_constraints"])

    def test_no_action_baseline_structure(self, sample_candidates):
        """Check no action baseline structure."""
        optimizer = DecisionOptimizer(budget=500.0, max_capacity=10)
        res = optimizer.optimize(sample_candidates)
        no_action = res["strategies"]["no_action"]
        assert no_action["total_cost"] == 0.0
        assert no_action["interventions_count"] == 0
        assert "zero intervention" in no_action["rationale"].lower()
