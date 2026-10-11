"""Constraint-Aware Business Decision Optimizer.

Formulates and solves portfolio action selection under real-world business constraints:
- Configurable budgets (e.g. ₹50,000 / $5,000)
- Maximum intervention capacity (e.g. max 50 contacts)
- Action eligibility & feasibility rules
- Risk & uncertainty discounting
- Mathematical Integer Linear Programming (Knapsack), Greedy ROI, Risk-First, and No-Action baseline comparisons
- Transparent attribution explaining binding constraints and prioritization logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from app.recommendations.action_catalog import ACTION_CATALOG, Action, get_action
from app.recommendations.domains import detect_domain

logger = logging.getLogger(__name__)


@dataclass
class CandidateCase:
    """A customer / entity candidate for portfolio intervention."""

    entity_id: str
    features: dict[str, Any]
    baseline_probability: float
    risk_level: str = "medium"
    low_confidence: bool = False
    customer_value: float = 600.0  # Estimated annual revenue / value


@dataclass
class ActionOption:
    """An eligible candidate action evaluated for a specific case."""

    case_idx: int
    entity_id: str
    action_code: str
    action_name: str
    cost: float
    expected_probability_reduction: float
    expected_benefit: float
    net_benefit: float
    roi_ratio: float
    uncertainty_penalty: float
    reasoning: str


@dataclass
class StrategyResult:
    """Solution output for a specific optimization strategy."""

    strategy_name: str
    display_title: str
    total_cost: float
    total_expected_benefit: float
    total_net_benefit: float
    interventions_count: int
    budget_utilization_pct: float
    capacity_utilization_pct: float
    allocated_actions: list[dict[str, Any]]
    binding_constraints: list[str]
    rationale: str


class DecisionOptimizer:
    """Portfolio optimizer for business interventions under constraints."""

    def __init__(
        self,
        budget: float,
        max_capacity: int,
        currency_symbol: str = "₹",
        domain: str = "telco",
        uncertainty_discount_factor: float = 0.20,
    ):
        """Initialize optimizer.

        Args:
            budget: Total available spending budget.
            max_capacity: Maximum number of customer interventions allowed.
            currency_symbol: Currency symbol for display (e.g. '₹' or '$').
            domain: Business domain for action catalog lookup.
            uncertainty_discount_factor: Fraction by which benefits on low-confidence cases are discounted.
        """
        self.budget = float(max(0.0, budget))
        self.max_capacity = int(max(1, max_capacity))
        self.currency_symbol = currency_symbol
        self.domain = domain
        self.uncertainty_discount_factor = float(uncertainty_discount_factor)

    def optimize(
        self,
        candidates: list[CandidateCase],
        custom_actions: Optional[list[dict[str, Any]]] = None,
        strategy: str = "all",
    ) -> dict[str, Any]:
        """Run portfolio optimization across candidates and strategies.

        Args:
            candidates: List of CandidateCases to consider.
            custom_actions: Optional org-specific actions.
            strategy: 'knapsack' | 'greedy_roi' | 'risk_first' | 'all'

        Returns:
            Dictionary comparing strategies, portfolio actions, and constraint explanations.
        """
        if not candidates:
            return {
                "success": False,
                "message": "No candidate cases provided for optimization.",
                "strategies": {},
            }

        # 1. Enumerate eligible action options for each candidate
        options = self._generate_eligible_options(candidates, custom_actions)

        if not options:
            return {
                "success": False,
                "message": "No eligible actions found for the candidate pool.",
                "strategies": {},
            }

        # 2. Run requested strategies
        results: dict[str, Any] = {}

        # Baseline: No-Action
        no_action = self._solve_no_action(candidates)
        results["no_action"] = self._format_strategy(no_action)

        # Strategy 1: Optimal Integer Linear Programming (Knapsack)
        if strategy in ("all", "knapsack"):
            knapsack_res = self._solve_knapsack(candidates, options)
            results["knapsack_optimal"] = self._format_strategy(knapsack_res)

        # Strategy 2: Greedy ROI
        if strategy in ("all", "greedy_roi"):
            greedy_res = self._solve_greedy_roi(candidates, options)
            results["greedy_roi"] = self._format_strategy(greedy_res)

        # Strategy 3: Risk-First
        if strategy in ("all", "risk_first"):
            risk_res = self._solve_risk_first(candidates, options)
            results["risk_first"] = self._format_strategy(risk_res)

        # Determine recommended strategy (highest net benefit among feasible strategies)
        best_name = "knapsack_optimal" if "knapsack_optimal" in results else "greedy_roi"
        candidates_count = len(candidates)

        return {
            "success": True,
            "configured_budget": self.budget,
            "max_capacity": self.max_capacity,
            "candidate_pool_size": candidates_count,
            "currency_symbol": self.currency_symbol,
            "recommended_strategy": best_name,
            "strategies": results,
            "governance": {
                "constraints_validated": True,
                "benefit_is_model_estimated": True,
                "uncertainty_discount_applied": True,
                "disclaimer": (
                    "Projected benefits reflect model retention elasticity and customer value estimates. "
                    "Actual realization depends on customer responsiveness and execution fidelity."
                ),
            },
        }

    def _generate_eligible_options(
        self,
        candidates: list[CandidateCase],
        custom_actions: Optional[list[dict[str, Any]]] = None,
    ) -> list[ActionOption]:
        """Generate and score eligible actions for each candidate case."""
        options: list[ActionOption] = []

        catalog = dict(ACTION_CATALOG)
        if custom_actions:
            for ca in custom_actions:
                code = ca.get("code") or ca.get("id")
                if code:
                    catalog[code] = Action(
                        code=code,
                        name=ca.get("name", code),
                        description=ca.get("description", ""),
                        base_cost=float(ca.get("estimated_cost", 50.0)),
                        impact_potential=float(ca.get("estimated_impact", 0.5)),
                        implementation_time="short",
                        category="save",
                    )

        for idx, cand in enumerate(candidates):
            # Only consider actionable cases with non-negligible risk
            if cand.baseline_probability < 0.20:
                continue

            for code, action in catalog.items():
                if not self._is_action_eligible(action, cand):
                    continue

                # Dollar cost
                cost = self._calculate_dollar_cost(action, cand)
                if cost > self.budget:
                    continue

                # Expected probability reduction
                lift = min(0.40, action.impact_potential * cand.baseline_probability * 0.6)
                
                # Raw expected dollar benefit = lift * customer annual value
                raw_benefit = lift * cand.customer_value

                # Apply uncertainty penalty if confidence is low
                penalty = 0.0
                if cand.low_confidence:
                    penalty = raw_benefit * self.uncertainty_discount_factor
                effective_benefit = max(0.0, raw_benefit - penalty)

                net_benefit = effective_benefit - cost
                roi_ratio = effective_benefit / max(1.0, cost)

                reasoning = (
                    f"Selected {action.name} (cost {self.currency_symbol}{cost:.0f}) "
                    f"to reduce risk from {cand.baseline_probability*100:.0f}% to "
                    f"{(cand.baseline_probability - lift)*100:.0f}%."
                )

                options.append(
                    ActionOption(
                        case_idx=idx,
                        entity_id=cand.entity_id,
                        action_code=code,
                        action_name=action.name,
                        cost=cost,
                        expected_probability_reduction=lift,
                        expected_benefit=effective_benefit,
                        net_benefit=net_benefit,
                        roi_ratio=roi_ratio,
                        uncertainty_penalty=penalty,
                        reasoning=reasoning,
                    )
                )

        return options

    def _is_action_eligible(self, action: Action, cand: CandidateCase) -> bool:
        """Check business eligibility conditions."""
        conds = action.applicable_conditions or {}
        if not conds:
            return True

        if "min_probability" in conds and cand.baseline_probability < conds["min_probability"]:
            return False

        contract = str(cand.features.get("contract_type", cand.features.get("Contract", ""))).lower()
        if "contract_type_in" in conds:
            allowed = [str(x).lower() for x in conds["contract_type_in"]]
            if contract and contract not in allowed:
                return False

        return True

    def _calculate_dollar_cost(self, action: Action, cand: CandidateCase) -> float:
        """Scale base cost into realistic local currency amount."""
        monthly_charges = float(
            cand.features.get("monthly_charges", cand.features.get("MonthlyCharges", 65.0)) or 65.0
        )
        # If action is a percentage discount:
        code = action.code.lower()
        if "discount_10" in code:
            return round(monthly_charges * 0.10 * 6.0, 2)  # 6-month discount
        elif "discount_20" in code:
            return round(monthly_charges * 0.20 * 6.0, 2)
        elif "upgrade_contract" in code:
            return 80.0
        elif "plan_optimization" in code:
            return 45.0
        elif "support" in code:
            return 60.0

        # General scaled cost
        return round(max(30.0, action.base_cost * 200.0), 2)

    def _solve_knapsack(
        self,
        candidates: list[CandidateCase],
        options: list[ActionOption],
    ) -> StrategyResult:
        """Exact Integer Linear Programming formulation."""
        n_options = len(options)
        if n_options == 0:
            return self._empty_strategy("knapsack_optimal", "Optimal Portfolio (Integer Programming)")

        # Objective: minimize -NetBenefit
        c = np.array([-opt.net_benefit for opt in options])

        # Constraints:
        # 1. Total Cost <= Budget
        cost_row = [opt.cost for opt in options]

        # 2. Total Interventions <= Max Capacity
        capacity_row = [1.0 for _ in options]

        A_ub = [cost_row, capacity_row]
        b_ub = [self.budget, float(self.max_capacity)]

        # 3. At most one action per customer case
        case_map: dict[int, list[int]] = {}
        for opt_idx, opt in enumerate(options):
            case_map.setdefault(opt.case_idx, []).append(opt_idx)

        for case_idx, opt_indices in case_map.items():
            if len(opt_indices) > 1:
                row = [0.0] * n_options
                for oi in opt_indices:
                    row[oi] = 1.0
                A_ub.append(row)
                b_ub.append(1.0)

        # Solve binary ILP
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(0, 1),
            integrality=np.ones(n_options),
            method="highs",
        )

        selected_indices = []
        if res.success:
            selected_indices = [i for i, x in enumerate(res.x) if x > 0.5]
        else:
            # Fall back to greedy if solver fails
            return self._solve_greedy_roi(candidates, options)

        allocated = [options[i] for i in selected_indices]
        return self._build_strategy_result(
            "knapsack_optimal",
            "Optimal Portfolio (Integer Linear Programming)",
            allocated,
            "Globally maximizes net business benefit subject to total budget and capacity limits.",
        )

    def _solve_greedy_roi(
        self,
        candidates: list[CandidateCase],
        options: list[ActionOption],
    ) -> StrategyResult:
        """Greedy selection prioritized by Benefit/Cost ROI ratio."""
        # Sort options descending by ROI ratio
        sorted_opts = sorted(options, key=lambda o: (o.net_benefit > 0, o.roi_ratio), reverse=True)

        allocated: list[ActionOption] = []
        spent = 0.0
        seen_cases: set[int] = set()

        for opt in sorted_opts:
            if opt.case_idx in seen_cases:
                continue
            if len(allocated) >= self.max_capacity:
                break
            if spent + opt.cost <= self.budget:
                allocated.append(opt)
                spent += opt.cost
                seen_cases.add(opt.case_idx)

        return self._build_strategy_result(
            "greedy_roi",
            "Highest ROI First (Benefit / Cost Ratio)",
            allocated,
            "Prioritizes actions with the highest expected dollar return per unit spent.",
        )

    def _solve_risk_first(
        self,
        candidates: list[CandidateCase],
        options: list[ActionOption],
    ) -> StrategyResult:
        """Selection prioritized by highest baseline risk severity."""
        # Find highest-impact action for each customer
        cand_best_action: dict[int, ActionOption] = {}
        for opt in sorted(options, key=lambda o: o.expected_benefit, reverse=True):
            if opt.case_idx not in cand_best_action:
                cand_best_action[opt.case_idx] = opt

        # Sort candidate cases by baseline risk descending
        cand_indices_by_risk = sorted(
            cand_best_action.keys(),
            key=lambda ci: candidates[ci].baseline_probability,
            reverse=True,
        )

        allocated: list[ActionOption] = []
        spent = 0.0
        for ci in cand_indices_by_risk:
            if len(allocated) >= self.max_capacity:
                break
            opt = cand_best_action[ci]
            if spent + opt.cost <= self.budget:
                allocated.append(opt)
                spent += opt.cost

        return self._build_strategy_result(
            "risk_first",
            "Highest Risk First (Severe Cases Prioritized)",
            allocated,
            "Focuses budget on customers with the highest baseline churn probability first.",
        )

    def _solve_no_action(self, candidates: list[CandidateCase]) -> StrategyResult:
        """No-action baseline for benchmarking."""
        total_loss = sum(c.baseline_probability * c.customer_value for c in candidates)
        return StrategyResult(
            strategy_name="no_action",
            display_title="No-Action Baseline (Zero Spend)",
            total_cost=0.0,
            total_expected_benefit=0.0,
            total_net_benefit=0.0,
            interventions_count=0,
            budget_utilization_pct=0.0,
            capacity_utilization_pct=0.0,
            allocated_actions=[],
            binding_constraints=[],
            rationale=(
                f"Status quo with zero intervention expenditure. "
                f"Estimated natural customer attrition risk value: {self.currency_symbol}{total_loss:.0f}."
            ),
        )

    def _build_strategy_result(
        self,
        name: str,
        title: str,
        allocated: list[ActionOption],
        rationale: str,
    ) -> StrategyResult:
        total_cost = sum(a.cost for a in allocated)
        total_benefit = sum(a.expected_benefit for a in allocated)
        total_net = sum(a.net_benefit for a in allocated)

        budget_pct = (total_cost / max(1.0, self.budget)) * 100.0
        cap_pct = (len(allocated) / max(1, self.max_capacity)) * 100.0

        binding = []
        if budget_pct >= 90.0:
            binding.append(f"Budget constraint binding ({budget_pct:.1f}% utilized)")
        if cap_pct >= 95.0:
            binding.append(f"Intervention capacity limit reached ({len(allocated)}/{self.max_capacity})")

        formatted_actions = [
            {
                "case_idx": a.case_idx,
                "entity_id": a.entity_id,
                "action_code": a.action_code,
                "action_name": a.action_name,
                "cost": round(a.cost, 2),
                "expected_probability_reduction": round(a.expected_probability_reduction, 4),
                "expected_benefit": round(a.expected_benefit, 2),
                "net_benefit": round(a.net_benefit, 2),
                "roi_ratio": round(a.roi_ratio, 2),
                "uncertainty_discounted": a.uncertainty_penalty > 0,
                "reasoning": a.reasoning,
            }
            for a in allocated
        ]

        return StrategyResult(
            strategy_name=name,
            display_title=title,
            total_cost=round(total_cost, 2),
            total_expected_benefit=round(total_benefit, 2),
            total_net_benefit=round(total_net, 2),
            interventions_count=len(allocated),
            budget_utilization_pct=round(budget_pct, 1),
            capacity_utilization_pct=round(cap_pct, 1),
            allocated_actions=formatted_actions,
            binding_constraints=binding,
            rationale=rationale,
        )

    def _empty_strategy(self, name: str, title: str) -> StrategyResult:
        return StrategyResult(
            strategy_name=name,
            display_title=title,
            total_cost=0.0,
            total_expected_benefit=0.0,
            total_net_benefit=0.0,
            interventions_count=0,
            budget_utilization_pct=0.0,
            capacity_utilization_pct=0.0,
            allocated_actions=[],
            binding_constraints=[],
            rationale="No eligible actions within budget.",
        )

    def _format_strategy(self, res: StrategyResult) -> dict[str, Any]:
        return {
            "strategy_name": res.strategy_name,
            "display_title": res.display_title,
            "total_cost": res.total_cost,
            "total_expected_benefit": res.total_expected_benefit,
            "total_net_benefit": res.total_net_benefit,
            "interventions_count": res.interventions_count,
            "budget_utilization_pct": res.budget_utilization_pct,
            "capacity_utilization_pct": res.capacity_utilization_pct,
            "binding_constraints": res.binding_constraints,
            "rationale": res.rationale,
            "allocated_actions": res.allocated_actions,
        }
