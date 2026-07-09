"""
app/routers/graph.py
GET /api/graph  — nodes + edges for the frontend canvas
GET /api/stats  — topbar stat strip numbers
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_analyst
from app.models.db import Account, Cluster, ClusterStatus, Transaction
from app.services.graph_engine import get_engine

router = APIRouter(prefix="/api", tags=["graph"])
logger = logging.getLogger(__name__)


@router.get("/graph")
async def get_graph(
    window_minutes: int = 300,
    cluster_id: str | None = None,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """
    Returns {nodes, edges} for the frontend graph canvas.
    Nodes include id, label, score, tier (high/elevated/clean), known_mule, cluster_id.
    window_minutes: how far back to include edges (default 300 = 5h, matching prototype).
    cluster_id: if provided, only return nodes/edges in that cluster.
    """
    engine = get_engine()
    data = engine.get_graph_data(window_minutes=window_minutes)

    # Attach cluster_id to each node
    clusters = db.query(Cluster).filter(
        Cluster.status != ClusterStatus.dismissed
    ).all()
    node_to_cluster: dict[str, str] = {}
    for c in clusters:
        for acc_id in (c.account_ids or []):
            node_to_cluster[acc_id] = c.id

    nodes = []
    for n in data["nodes"]:
        cid = node_to_cluster.get(n["id"])
        if cluster_id and cid != cluster_id:
            continue
        nodes.append({**n, "cluster_id": cid})

    edges = data["edges"]
    if cluster_id:
        node_ids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["from"] in node_ids and e["to"] in node_ids]

    return {"nodes": nodes, "edges": edges}


@router.get("/stats")
async def get_stats(
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """
    Topbar stat strip — matches the prototype's displayed numbers.
    """
    total_accounts = db.query(Account).count()
    total_transactions = db.query(Transaction).count()

    active_clusters = db.query(Cluster).filter(
        Cluster.status.in_([ClusterStatus.pending, ClusterStatus.escalated, ClusterStatus.monitoring])
    ).count()

    pending_clusters = db.query(Cluster).filter(
        Cluster.status == ClusterStatus.pending
    ).count()

    escalated_clusters = db.query(Cluster).filter(
        Cluster.status == ClusterStatus.escalated
    ).count()

    return {
        "total_accounts": total_accounts,
        "total_transactions": total_transactions,
        "active_clusters": active_clusters,
        "pending_clusters": pending_clusters,
        "escalated_clusters": escalated_clusters,
        "window_hours": 5,
    }


@router.get("/accounts")
async def list_accounts(
    known_mule: bool | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """List accounts with optional known_mule filter."""
    query = db.query(Account)
    if known_mule is not None:
        query = query.filter(Account.known_mule == known_mule)
    accounts = query.limit(min(limit, 500)).all()
    return {
        "accounts": [
            {
                "id": a.id,
                "label": a.label,
                "account_type": a.account_type,
                "known_mule": a.known_mule,
                "status": a.status.value if a.status else "clean",
            }
            for a in accounts
        ]
    }