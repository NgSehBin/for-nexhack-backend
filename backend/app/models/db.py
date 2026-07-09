import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, DateTime, Text, Integer,
    Enum as SAEnum, ForeignKey, Boolean
)
from sqlalchemy.dialects.postgresql import JSONB
from app.core.database import Base


def new_uuid() -> str:
    return str(uuid.uuid4())


# ── Enums ──────────────────────────────────────────────────────────────────

class AccountStatus(str, enum.Enum):
    clean = "clean"
    suspected = "suspected"
    confirmed_mule = "confirmed_mule"


class ClusterStatus(str, enum.Enum):
    pending = "pending"
    escalated = "escalated"
    monitoring = "monitoring"
    dismissed = "dismissed"


class AnalystRole(str, enum.Enum):
    analyst = "analyst"
    admin = "admin"


# ── Tables ─────────────────────────────────────────────────────────────────

class Account(Base):
    __tablename__ = "accounts"

    id = Column(String, primary_key=True, default=new_uuid)
    label = Column(String, nullable=False)             # human-readable name (matches prototype)
    account_type = Column(String, nullable=False)      # mule / source / victim / benign
    bank_code = Column(String, nullable=True)
    registration_date = Column(DateTime, nullable=True)
    status = Column(SAEnum(AccountStatus), default=AccountStatus.clean)
    known_mule = Column(Boolean, default=False)        # matches prototype knownMule flag
    device_fingerprint = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    tenant_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=new_uuid)
    sender_id = Column(String, ForeignKey("accounts.id"), nullable=False)
    receiver_id = Column(String, ForeignKey("accounts.id"), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String, default="MYR")
    timestamp = Column(DateTime, nullable=False)
    channel = Column(String, nullable=False)           # DuitNow / IBG / FPX / SWIFT
    reference = Column(String, nullable=True)          # NEVER passed to Gemini
    tenant_id = Column(String, nullable=True)


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(String, primary_key=True, default=new_uuid)
    account_ids = Column(JSONB, nullable=False)        # list[str]
    risk_score = Column(Float, nullable=False)         # 0-100, matches prototype clusterScore
    total_flow = Column(Float, default=0.0)            # MYR moved within cluster
    known_mule_count = Column(Integer, default=0)
    pattern_flags = Column(JSONB, nullable=True)       # {fan_in, fan_out, pass_through, proximity}
    node_scores = Column(JSONB, nullable=True)         # {account_id: score} per-node scores
    shap_values = Column(JSONB, nullable=True)         # top SHAP features
    status = Column(SAEnum(ClusterStatus), default=ClusterStatus.pending)
    report_text = Column(Text, nullable=True)          # JSON string of AI case brief
    assigned_analyst_id = Column(String, nullable=True)
    tenant_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Analyst(Base):
    __tablename__ = "analysts"

    id = Column(String, primary_key=True, default=new_uuid)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SAEnum(AnalystRole), default=AnalystRole.analyst)
    tenant_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """Append-only. REVOKE UPDATE, DELETE on this table at DB level."""
    __tablename__ = "audit_log"

    id = Column(String, primary_key=True, default=new_uuid)
    cluster_id = Column(String, nullable=False)        # no cascade FK — must outlive cluster
    analyst_id = Column(String, nullable=False)
    action = Column(String, nullable=False)            # approve / reject / modify
    previous_status = Column(String, nullable=True)
    new_status = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
