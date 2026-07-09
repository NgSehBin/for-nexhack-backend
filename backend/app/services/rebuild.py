"""
Background task: runs after every ingest batch.
1. Adds transactions to graph
2. Recomputes all node scores
3. Detects clusters via Louvain
4. Upserts Cluster rows in DB
"""
import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.db import Account, Transaction, Cluster, ClusterStatus
from app.services.graph_engine import get_engine

logger = logging.getLogger(__name__)


async def rebuild_graph_and_score(txn_ids: list[str]):
    """Called as a FastAPI BackgroundTask after each ingest batch."""
    db: Session = SessionLocal()
    try:
        engine = get_engine()

        # Fetch full transaction + account data from DB
        txns = db.query(Transaction).filter(Transaction.id.in_(txn_ids)).all()
        all_account_ids = list({t.sender_id for t in txns} | {t.receiver_id for t in txns})
        accounts = db.query(Account).filter(Account.id.in_(all_account_ids)).all()

        txn_dicts = [
            {
                "sender_id": t.sender_id,
                "receiver_id": t.receiver_id,
                "amount": t.amount,
                "timestamp": t.timestamp,
            }
            for t in txns
        ]
        acc_dicts = [
            {
                "id": a.id,
                "label": a.label,
                "known_mule": a.known_mule,
                "account_type": a.account_type,
            }
            for a in accounts
        ]

        await engine.add_transactions(txn_dicts, acc_dicts)

        # Recompute scores for all nodes
        node_scores = engine.compute_all_scores()

        # Detect clusters
        clusters = engine.detect_clusters()

        # Upsert clusters into DB
        # Only match against clusters touched recently — without this, every
        # ingest batch re-scans every non-dismissed cluster ever created,
        # which gets slower as the table grows.
        match_cutoff = datetime.utcnow() - timedelta(days=7)
        existing = (
            db.query(Cluster)
            .filter(
                Cluster.status.notin_([ClusterStatus.dismissed]),
                Cluster.updated_at >= match_cutoff,
            )
            .all()
        )

        for cluster_data in clusters:
            member_ids = cluster_data["account_ids"]
            matched = next(
                (c for c in existing
                 if set(c.account_ids or []) == set(member_ids)),
                None
            )

            if matched:
                # Update score and flags
                matched.risk_score = cluster_data["risk_score"]
                matched.total_flow = cluster_data["total_flow"]
                matched.known_mule_count = cluster_data["known_mule_count"]
                matched.pattern_flags = cluster_data["pattern_flags"]
                matched.node_scores = cluster_data["node_scores"]
                matched.updated_at = datetime.utcnow()
            else:
                new_cluster = Cluster(
                    account_ids=member_ids,
                    risk_score=cluster_data["risk_score"],
                    total_flow=cluster_data["total_flow"],
                    known_mule_count=cluster_data["known_mule_count"],
                    pattern_flags=cluster_data["pattern_flags"],
                    node_scores=cluster_data["node_scores"],
                    status=ClusterStatus.pending,
                )
                db.add(new_cluster)

        db.commit()
        logger.info(f"Rebuild complete: {len(clusters)} clusters, {len(node_scores)} nodes scored")

    except Exception as e:
        logger.error(f"Rebuild failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()
