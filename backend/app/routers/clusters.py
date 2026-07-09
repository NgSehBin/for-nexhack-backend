"""
app/routers/clusters.py
GET  /api/clusters              — list all clusters ordered by risk score
GET  /api/cluster/{id}          — full cluster detail
GET  /api/cluster/{id}/report   — get (or generate) AI case brief
POST /api/cluster/{id}/decision — analyst decision with audit log
"""
import json
import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_analyst
from app.models.db import Account, AuditLog, Cluster, ClusterStatus, Transaction
from app.models.schemas import (
    ClusterDetail,
    ClusterSummary,
    DecisionRequest,
)
from app.services.agent import get_agent

router = APIRouter(prefix="/api", tags=["clusters"])
logger = logging.getLogger(__name__)


def _check_cluster_access(cluster: Cluster, user: dict):
    """IDOR guard — analyst can only access clusters in their tenant."""
    if user["role"] == "admin":
        return
    if cluster.tenant_id and cluster.tenant_id != user.get("tenant_id"):
        raise HTTPException(status_code=403, detail="Access denied")


def _cluster_to_summary(c: Cluster) -> dict:
    return {
        "id": c.id,
        "risk_score": c.risk_score,
        "member_count": len(c.account_ids or []),
        "total_flow": c.total_flow,
        "known_mule_count": c.known_mule_count,
        "pattern_flags": c.pattern_flags,
        "status": c.status.value if c.status else "pending",
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


def _cluster_to_detail(c: Cluster, db: Session) -> dict:
    member_ids = c.account_ids or []

    # Fetch account details for members
    accounts = db.query(Account).filter(Account.id.in_(member_ids)).all()
    member_accounts = [
        {
            "id": a.id,
            "label": a.label,
            "account_type": a.account_type,
            "known_mule": a.known_mule,
            "status": a.status.value if a.status else "clean",
        }
        for a in accounts
    ]

    # Fetch intra-cluster transactions for timeline
    txns = (
        db.query(Transaction)
        .filter(
            Transaction.sender_id.in_(member_ids),
            Transaction.receiver_id.in_(member_ids),
        )
        .order_by(Transaction.timestamp)
        .limit(100)
        .all()
    )
    transactions = [
        {
            "id": t.id,
            "sender_id": t.sender_id,
            "receiver_id": t.receiver_id,
            "amount": t.amount,
            "currency": t.currency,
            "timestamp": t.timestamp.isoformat(),
            "channel": t.channel,
        }
        for t in txns
    ]

    report = None
    if c.report_text:
        try:
            report = json.loads(c.report_text)
        except json.JSONDecodeError:
            report = {"raw": c.report_text}

    return {
        "id": c.id,
        "risk_score": c.risk_score,
        "account_ids": member_ids,
        "member_accounts": member_accounts,
        "member_count": len(member_ids),
        "total_flow": c.total_flow,
        "known_mule_count": c.known_mule_count,
        "pattern_flags": c.pattern_flags,
        "node_scores": c.node_scores,
        "shap_values": c.shap_values,
        "status": c.status.value if c.status else "pending",
        "report": report,
        "transactions": transactions,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


@router.get("/clusters")
async def list_clusters(
    status: str | None = None,
    min_score: float = 0,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """
    List clusters ordered by risk_score descending.
    Optional filters: status (pending/escalated/monitoring/dismissed), min_score.
    """
    query = db.query(Cluster)

    # Tenant filter for non-admins
    if user["role"] != "admin":
        query = query.filter(
            (Cluster.tenant_id == user.get("tenant_id")) |
            (Cluster.tenant_id.is_(None))
        )

    if status:
        try:
            query = query.filter(Cluster.status == ClusterStatus(status))
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    if min_score > 0:
        query = query.filter(Cluster.risk_score >= min_score)

    total = query.count()
    clusters = (
        query
        .order_by(Cluster.risk_score.desc())
        .offset(offset)
        .limit(min(limit, 200))
        .all()
    )

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "clusters": [_cluster_to_summary(c) for c in clusters],
    }


@router.get("/cluster/{cluster_id}")
async def get_cluster(
    cluster_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        raise HTTPException(404, "Cluster not found")
    _check_cluster_access(cluster, user)
    return _cluster_to_detail(cluster, db)


@router.get("/cluster/{cluster_id}/report")
async def get_cluster_report(
    cluster_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """
    Returns the AI case brief. If not yet generated, triggers generation
    as a background task and returns 202 so the frontend can poll.
    """
    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        raise HTTPException(404, "Cluster not found")
    _check_cluster_access(cluster, user)

    if cluster.report_text:
        try:
            return {"status": "ready", "report": json.loads(cluster.report_text)}
        except json.JSONDecodeError:
            pass

    # Trigger generation in background
    background_tasks.add_task(_generate_and_store_report, cluster_id)
    return {
        "status": "generating",
        "message": "Report is being generated. Poll this endpoint again in 10 seconds.",
    }


async def _generate_and_store_report(cluster_id: str):
    """Background task: generate case brief and store in DB."""
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        cluster = db.query(Cluster).filter_by(id=cluster_id).first()
        if not cluster:
            return

        agent = get_agent()
        report = agent.generate_case_brief(cluster, db)
        cluster.report_text = json.dumps(report)
        db.commit()
        logger.info(f"Case brief generated for cluster {cluster_id}")
    except Exception as e:
        logger.error(f"Case brief generation failed for {cluster_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


@router.post("/cluster/{cluster_id}/decision")
async def submit_decision(
    cluster_id: str,
    request: Request,
    body: DecisionRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """
    Analyst decision: escalate | monitor | dismiss.
    Writes an immutable audit log row. Returns updated cluster status.
    """
    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        raise HTTPException(404, "Cluster not found")
    _check_cluster_access(cluster, user)

    previous_status = cluster.status.value if cluster.status else "pending"

    # Map decision to ClusterStatus
    status_map = {
        "escalate": ClusterStatus.escalated,
        "monitor": ClusterStatus.monitoring,
        "dismiss": ClusterStatus.dismissed,
    }
    new_status = status_map[body.action]
    cluster.status = new_status
    cluster.assigned_analyst_id = user["user_id"]
    cluster.updated_at = datetime.utcnow()

    # Append-only audit log (REVOKE UPDATE DELETE is set at DB level)
    audit = AuditLog(
        cluster_id=cluster_id,
        analyst_id=user["user_id"],
        action=body.action,
        previous_status=previous_status,
        new_status=new_status.value,
        notes=body.notes,
        ip_address=request.client.host if request.client else None,
    )
    db.add(audit)
    db.commit()

    return {
        "cluster_id": cluster_id,
        "previous_status": previous_status,
        "new_status": new_status.value,
        "analyst_id": user["user_id"],
    }


@router.get("/cluster/{cluster_id}/audit")
async def get_audit_log(
    cluster_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """Returns the full audit trail for a cluster."""
    cluster = db.query(Cluster).filter_by(id=cluster_id).first()
    if not cluster:
        raise HTTPException(404, "Cluster not found")
    _check_cluster_access(cluster, user)

    logs = (
        db.query(AuditLog)
        .filter_by(cluster_id=cluster_id)
        .order_by(AuditLog.timestamp.asc())
        .all()
    )
    return {
        "cluster_id": cluster_id,
        "entries": [
            {
                "id": log.id,
                "analyst_id": log.analyst_id,
                "action": log.action,
                "previous_status": log.previous_status,
                "new_status": log.new_status,
                "notes": log.notes,
                "timestamp": log.timestamp.isoformat(),
            }
            for log in logs
        ],
    }