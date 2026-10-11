"""API routes for Feature 5: End-to-End Decision Feedback Loop and Traceability Analytics."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.routes.auth import AuthContext, get_auth_context
from app.services.feedback_analytics_service import FeedbackAnalyticsService

router = APIRouter()


@router.get("/decisions/{decision_id}/lineage")
async def get_decision_lineage(
    decision_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Retrieve full end-to-end lineage for a recommendation/decision."""
    service = FeedbackAnalyticsService(db, auth.org_id)
    try:
        return service.get_decision_lineage(decision_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lineage lookup failed: {str(e)}")


@router.get("/effectiveness")
async def get_feedback_effectiveness(
    project_id: Optional[str] = Query(None, description="Optional project ID filter"),
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Analyze recommendation effectiveness, empirical lift vs predicted lift, and completion rates."""
    service = FeedbackAnalyticsService(db, auth.org_id)
    try:
        return service.generate_effectiveness_report(project_id=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Effectiveness calculation failed: {str(e)}")


@router.get("/projects/{project_id}/export-dataset")
async def export_feedback_dataset(
    project_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    """Export verified ground-truth feedback as a labeled dataset for retraining/validation."""
    service = FeedbackAnalyticsService(db, auth.org_id)
    try:
        return service.export_verified_feedback_dataset(project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback export failed: {str(e)}")
