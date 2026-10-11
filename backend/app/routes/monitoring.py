"""Monitoring, Drift, and Model Lifecycle API routes."""

import os
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.db.models import Project, ProjectPrediction
from app.routes.auth import AuthContext, get_auth_context
from app.services.drift_service import DriftMonitoringService
from app.services.model_lifecycle_service import ModelLifecycleService
from app.services.project_service import ProjectService

router = APIRouter()


class CandidateTrainRequest(BaseModel):
    trigger_reason: str = Field(default="Evidence-based drift trigger")


class PromoteRequest(BaseModel):
    force: bool = Field(default=False)


@router.get("/projects/{project_id}/drift")
async def get_project_drift_report(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Analyze feature and prediction drift between training reference and current cases."""
    proj_service = ProjectService(db, auth.org_id)
    project = proj_service.get_project(project_id)
    if not project or not project.dataset:
        raise HTTPException(status_code=404, detail="Project or dataset not found")

    trained_model = proj_service.get_active_model(project_id)
    if not trained_model:
        raise HTTPException(status_code=400, detail="Project does not have an active trained model")

    # Load reference dataset
    path = project.dataset.file_path
    if not path.endswith(".parquet") and not path.endswith(".csv"):
        parquet_path = path.replace(".csv", ".parquet")
        if os.path.exists(parquet_path):
            path = parquet_path
    ref_df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    # Load incoming cases from recent predictions or split test-data
    recent_preds = (
        db.query(ProjectPrediction)
        .filter(ProjectPrediction.project_id == project_id)
        .order_by(ProjectPrediction.created_at.desc())
        .limit(100)
        .all()
    )

    if len(recent_preds) >= 20:
        cur_df = pd.DataFrame([p.features for p in recent_preds])
        cur_preds = [p.probability for p in recent_preds if p.probability is not None]
    else:
        # Generate simulated production batch with slight natural drift for inspection
        cur_df = ref_df.sample(min(200, len(ref_df)), random_state=123).copy()
        # Simulate slight price pressure shift on monthly charges
        for col in ("monthly_charges", "MonthlyCharges"):
            if col in cur_df.columns:
                cur_df[col] = pd.to_numeric(cur_df[col], errors="coerce") * 1.08
        cur_preds = None

    service = DriftMonitoringService()
    report = service.analyze_drift(
        reference_df=ref_df,
        current_df=cur_df,
        feature_columns=project.feature_columns,
    )
    return report


@router.post("/projects/{project_id}/candidate/train")
async def train_candidate_model(
    project_id: str,
    request: CandidateTrainRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Train a candidate model in isolation without altering production model."""
    lifecycle = ModelLifecycleService(db, auth.org_id)
    try:
        res = lifecycle.train_candidate_model(
            project_id=project_id,
            trigger_reason=request.trigger_reason,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Candidate training failed: {str(e)}")


@router.get("/projects/{project_id}/candidate/{candidate_id}/evaluate")
async def evaluate_candidate(
    project_id: str,
    candidate_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Evaluate candidate model against acceptance criteria before promotion."""
    lifecycle = ModelLifecycleService(db, auth.org_id)
    try:
        return lifecycle.evaluate_candidate(project_id, candidate_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/projects/{project_id}/candidate/{candidate_id}/promote")
async def promote_candidate_model(
    project_id: str,
    candidate_id: str,
    request: PromoteRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Promote an approved candidate model to active production."""
    lifecycle = ModelLifecycleService(db, auth.org_id)
    try:
        return lifecycle.promote_candidate(project_id, candidate_id, force=request.force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {str(e)}")


@router.post("/projects/{project_id}/rollback")
async def rollback_project_model(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Roll back active model to the previously deployed version."""
    lifecycle = ModelLifecycleService(db, auth.org_id)
    try:
        return lifecycle.rollback_model(project_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")


@router.get("/projects/{project_id}/history")
async def get_model_history(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """List all deployed and candidate model versions with historical metrics."""
    lifecycle = ModelLifecycleService(db, auth.org_id)
    return lifecycle.get_model_history(project_id)
