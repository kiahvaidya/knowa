"""Evaluation Lab API routes."""

import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.recommendations.domains import detect_domain
from app.routes.auth import AuthContext, get_auth_context
from app.services.dataset_service import DatasetService
from app.services.evaluation_service import EvaluationLabService
from app.services.project_service import ProjectService

router = APIRouter()


class RunBenchmarkRequest(BaseModel):
    dataset_path: Optional[str] = None
    target_column: str = Field(default="Churn")
    positive_label: str = Field(default="Yes")
    random_seed: int = Field(default=42)
    experiment_name: str = Field(default="benchmark_telco_churn")


@router.get("/latest")
async def get_latest_evaluation(project_id: Optional[str] = Query(default=None)):
    """Get latest model trust and evaluation report, optionally filtered by project_id."""
    service = EvaluationLabService()
    report = service.get_latest_report(project_id=project_id)
    if not report:
        raise HTTPException(status_code=404, detail="No evaluation reports available")
    return report


@router.get("/history")
async def list_evaluation_history():
    """List historical benchmark experiment runs."""
    service = EvaluationLabService()
    return service.list_reports()


@router.post("/run")
async def run_benchmark_experiment(request: RunBenchmarkRequest):
    """Run full evaluation benchmark on real-world dataset."""
    service = EvaluationLabService()
    try:
        report = service.run_benchmark(
            dataset_path=request.dataset_path,
            target_column=request.target_column,
            positive_label=request.positive_label,
            random_seed=request.random_seed,
            experiment_name=request.experiment_name,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


@router.post("/projects/{project_id}/evaluate")
async def evaluate_project_dataset(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Run full trust evaluation on a SaaS project's dataset across its registered domain."""
    proj_service = ProjectService(db, auth.org_id)
    project = proj_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    df = None
    if project.dataset_id:
        try:
            ds_service = DatasetService(db, auth.org_id)
            df = ds_service.load_dataframe(project.dataset_id)
        except Exception:
            pass

    file_path = None
    if df is None and project.dataset:
        file_path = project.dataset.file_path
        if not os.path.exists(file_path):
            if file_path.endswith(".csv"):
                parquet_path = file_path.replace(".csv", ".parquet")
                if os.path.exists(parquet_path):
                    file_path = parquet_path
            elif file_path.endswith(".parquet"):
                csv_path = file_path.replace(".parquet", ".csv")
                if os.path.exists(csv_path):
                    file_path = csv_path

    if df is None and (not file_path or not os.path.exists(file_path)):
        raise HTTPException(status_code=404, detail="Project dataset file could not be located")

    # Detect domain dynamically
    sample_cols = list(df.columns) if df is not None else ([f.name for f in project.features] if project.features else None)
    org_ind = project.organization.industry if project.organization else None
    domain = detect_domain(
        organization_industry=org_ind,
        feature_columns=sample_cols,
        project_name=project.name,
        target_column=project.target_column,
        target_description=project.target_description,
        metadata_text=f"{project.name} {project.description or ''} {project.target_description or ''}",
    )

    service = EvaluationLabService()
    try:
        report = service.run_benchmark(
            df=df,
            dataset_path=file_path,
            target_column=project.target_column,
            positive_label=project.target_positive_label or "1",
            experiment_name=f"project_{project.name[:25]}",
            dataset_name=project.name,
            domain=domain,
            project_id=project_id,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project evaluation failed: {str(e)}")
