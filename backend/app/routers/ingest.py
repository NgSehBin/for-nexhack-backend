"""
app/routers/ingest.py
POST /api/transactions — accepts a batch of transactions, persists to DB,
fires graph rebuild + scoring as a background task.
POST /api/accounts    — upsert account metadata (label, known_mule flag)
"""
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_analyst
from app.models.db import Account, AccountStatus, Transaction
from app.models.schemas import TransactionBatch
from app.services.rebuild import rebuild_graph_and_score

router = APIRouter(prefix="/api", tags=["ingest"])
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)


class AccountUpsert(BaseModel):
    id: str = Field(..., min_length=36, max_length=36)
    label: str = Field(..., min_length=1, max_length=120)
    account_type: str = Field(..., pattern=r"^(mule|source|victim|benign|unknown)$")
    known_mule: bool = False
    bank_code: str | None = Field(None, max_length=20)
    tenant_id: str = Field(default="default", max_length=64)


class AccountBatch(BaseModel):
    accounts: list[AccountUpsert] = Field(..., min_length=1, max_length=500)


@router.post("/accounts", status_code=202)
@limiter.limit("50/minute")
async def upsert_accounts(
    request: Request,
    payload: AccountBatch,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """
    Upsert account metadata. Call this before or alongside /transactions
    so the graph engine has labels and known_mule flags.
    """
    upserted = 0
    for a in payload.accounts:
        existing = db.query(Account).filter_by(id=a.id).first()
        if existing:
            existing.label = a.label
            existing.account_type = a.account_type
            existing.known_mule = a.known_mule
            if a.bank_code:
                existing.bank_code = a.bank_code
        else:
            db.add(Account(
                id=a.id,
                label=a.label,
                account_type=a.account_type,
                known_mule=a.known_mule,
                bank_code=a.bank_code,
                status=AccountStatus.confirmed_mule if a.known_mule else AccountStatus.clean,
                tenant_id=a.tenant_id or user.get("tenant_id", "default"),
            ))
        upserted += 1

    db.commit()
    return {"upserted": upserted}


@router.post("/transactions", status_code=202)
@limiter.limit("100/minute")
async def ingest_transactions(
    request: Request,
    payload: TransactionBatch,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: dict = Depends(require_analyst),
):
    """
    Ingest a batch of transactions. Returns immediately (202 Accepted).
    Graph rebuild + cluster scoring runs as a background task.
    """
    # Auto-create any accounts not yet in DB (minimal record, label = id prefix)
    all_account_ids = set()
    for t in payload.transactions:
        all_account_ids.add(t.sender_id)
        all_account_ids.add(t.receiver_id)

    existing_ids = {
        row.id for row in
        db.query(Account.id).filter(Account.id.in_(all_account_ids)).all()
    }
    for acc_id in all_account_ids - existing_ids:
        db.add(Account(
            id=acc_id,
            label=acc_id[:8],          # placeholder label until upserted
            account_type="unknown",
            tenant_id=user.get("tenant_id", "default"),
        ))

    # Persist transactions
    txn_ids = []
    for t in payload.transactions:
        existing = db.query(Transaction).filter_by(id=t.id).first()
        if existing:
            continue   # idempotent — skip duplicates
        db.add(Transaction(
            id=t.id,
            sender_id=t.sender_id,
            receiver_id=t.receiver_id,
            amount=t.amount,
            currency=t.currency,
            timestamp=t.timestamp,
            channel=t.channel,
            reference=t.reference,
            tenant_id=user.get("tenant_id", "default"),
        ))
        txn_ids.append(t.id)

    db.commit()

    if txn_ids:
        background_tasks.add_task(rebuild_graph_and_score, txn_ids)

    return {
        "accepted": len(txn_ids),
        "skipped_duplicates": len(payload.transactions) - len(txn_ids),
        "rebuild": "queued" if txn_ids else "skipped",
    }