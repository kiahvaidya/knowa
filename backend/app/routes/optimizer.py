"""Decision Optimizer API routes."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.db.models import Customer, Project, ProjectPrediction
from app.routes.auth import AuthContext, get_auth_context
from app.services.decision_service import DecisionService
from app.services.optimizer_service import CandidateCase, DecisionOptimizer
from app.services.project_service import ProjectService

router = APIRouter()


class OptimizeCohortRequest(BaseModel):
    budget: float = Field(default=50000.0, description="Total spend budget")
    max_capacity: int = Field(default=25, description="Maximum intervention capacity")
    currency_symbol: str = Field(default="₹", description="Currency symbol for display")
    strategy: str = Field(default="all", description="Optimization strategy to run")
    value_multiplier: float = Field(default=600.0, description="Annual customer value per retained customer")
    custom_candidates: Optional[list[dict[str, Any]]] = None


class CommitOptimizedPortfolioRequest(BaseModel):
    strategy_name: str = Field(default="knapsack_optimal")
    recheck_interval_days: int = Field(default=30)
    allocated_actions: list[dict[str, Any]]


@router.post("/demo-optimize")
async def demo_optimize(
    request: OptimizeCohortRequest,
    db: Session = Depends(get_db),
):
    """Run constraint-aware decision optimization on demo customer cohort."""
    optimizer = DecisionOptimizer(
        budget=request.budget,
        max_capacity=request.max_capacity,
        currency_symbol=request.currency_symbol,
    )

    candidates: list[CandidateCase] = []
    if request.custom_candidates:
        for idx, c in enumerate(request.custom_candidates):
            candidates.append(
                CandidateCase(
                    entity_id=str(c.get("entity_id", f"case_{idx+1}")),
                    features=c.get("features", {}),
                    baseline_probability=float(c.get("probability", 0.65)),
                    customer_value=float(c.get("customer_value", request.value_multiplier)),
                )
            )
    else:
        # Load from seeded demo customers
        customers = db.query(Customer).limit(50).all()
        for idx, cust in enumerate(customers):
            # Deterministic churn probability proxy based on monthly charges / tenure
            feats = cust.features or {}
            mc = float(feats.get("monthly_charges", 70.0))
            prob = min(0.92, max(0.18, (mc / 120.0) * 0.85))
            candidates.append(
                CandidateCase(
                    entity_id=cust.external_id or f"CUST_{idx+1}",
                    features=feats,
                    baseline_probability=prob,
                    customer_value=request.value_multiplier,
                )
            )

    return optimizer.optimize(candidates, strategy=request.strategy)


@router.post("/projects/{project_id}/optimize")
async def project_optimize(
    project_id: str,
    request: OptimizeCohortRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Run decision optimization on a project's active cohort under budget constraints."""
    proj_service = ProjectService(db, auth.org_id)
    project = proj_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    candidates: list[CandidateCase] = []

    # If custom candidates supplied, use them
    if request.custom_candidates:
        for idx, c in enumerate(request.custom_candidates):
            candidates.append(
                CandidateCase(
                    entity_id=str(c.get("entity_id", f"case_{idx+1}")),
                    features=c.get("features", {}),
                    baseline_probability=float(c.get("probability", 0.60)),
                    customer_value=float(c.get("customer_value", request.value_multiplier)),
                )
            )
    else:
        # Try loading test-data or recent predictions
        recent_preds = (
            db.query(ProjectPrediction)
            .filter(ProjectPrediction.project_id == project_id)
            .order_by(ProjectPrediction.created_at.desc())
            .limit(80)
            .all()
        )
        if recent_preds:
            for p in recent_preds:
                prob = float(p.probability) if p.probability is not None else 0.5
                candidates.append(
                    CandidateCase(
                        entity_id=p.entity_id or f"pred_{p.id[:8]}",
                        features=p.features,
                        baseline_probability=prob,
                        risk_level=p.risk_level or "medium",
                        low_confidence=bool(p.low_confidence),
                        customer_value=request.value_multiplier,
                    )
                )
        else:
            # Fallback to test data scored on the fly
            test_data_resp = proj_service.get_test_data(project_id, limit=40)
            for idx, r in enumerate(test_data_resp):
                mc = float(r.get("monthly_charges", r.get("MonthlyCharges", 65.0)) or 65.0)
                prob = min(0.90, max(0.20, (mc / 110.0) * 0.75))
                candidates.append(
                    CandidateCase(
                        entity_id=f"case_{idx+1}",
                        features=r,
                        baseline_probability=prob,
                        customer_value=request.value_multiplier,
                    )
                )

    optimizer = DecisionOptimizer(
        budget=request.budget,
        max_capacity=request.max_capacity,
        currency_symbol=request.currency_symbol,
    )

    custom_actions = [
        {"code": a.code, "name": a.name, "estimated_cost": a.estimated_cost, "estimated_impact": a.estimated_impact}
        for a in project.organization.custom_actions
    ] if project.organization and project.organization.custom_actions else None

    return optimizer.optimize(candidates, custom_actions=custom_actions, strategy=request.strategy)


@router.post("/projects/{project_id}/optimize/commit")
async def commit_optimized_portfolio(
    project_id: str,
    request: CommitOptimizedPortfolioRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Commit an entire optimized action portfolio into the B3 Decision Ledger."""
    decision_service = DecisionService(db, auth.org_id)
    committed: list[dict[str, Any]] = []

    for action in request.allocated_actions:
        try:
            dec = decision_service.create_from_case(
                project_id=project_id,
                action_code=action["action_code"],
                action_name=action.get("action_name"),
                entity_id=action.get("entity_id"),
                decision_summary=action.get("reasoning"),
                expected_lift=-float(action.get("expected_probability_reduction", 0.15)),
                recheck_interval_days=request.recheck_interval_days,
                actor_user_id=auth.user_id,
            )
            committed.append(dec)
        except Exception as e:
            logger.warning("Failed committing portfolio action: %s", e)

    return {
        "success": True,
        "committed_count": len(committed),
        "committed_decisions": committed,
        "message": f"Successfully committed {len(committed)} decisions from {request.strategy_name} into ledger.",
    }
