import re
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

# Patterns that suggest prompt injection attempts in free-text fields
_INJECTION_PATTERNS = [
    "ignore previous", "ignore all", "system:", "assistant:", "user:",
    "<|", "{{", "}}", "[inst]", "[/inst]", "you are now", "disregard",
    "override", "new instruction",
]

VALID_CHANNELS = {"DuitNow", "IBG", "FPX", "SWIFT"}


def _validate_uuid(v: str) -> str:
    if not UUID4_RE.match(v.lower()):
        raise ValueError("Must be a valid UUID4")
    return v


def _sanitize_text(v: str | None) -> str | None:
    if v is None:
        return None
    lower = v.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            raise ValueError(f"Field contains disallowed content: '{pattern}'")
    return v.strip()[:200]


# ── Ingest ─────────────────────────────────────────────────────────────────

class TransactionCreate(BaseModel):
    id: str = Field(..., min_length=36, max_length=36)
    sender_id: str = Field(..., min_length=36, max_length=36)
    receiver_id: str = Field(..., min_length=36, max_length=36)
    amount: float = Field(..., gt=0, lt=10_000_000)
    currency: str = Field(default="MYR", pattern=r"^(MYR|USD|SGD|EUR)$")
    timestamp: datetime
    channel: str
    reference: str | None = Field(None, max_length=200)

    @field_validator("id", "sender_id", "receiver_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        return _validate_uuid(v)

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        if v not in VALID_CHANNELS:
            raise ValueError(f"channel must be one of {VALID_CHANNELS}")
        return v

    @field_validator("reference")
    @classmethod
    def sanitize_reference(cls, v: str | None) -> str | None:
        return _sanitize_text(v)

    @model_validator(mode="after")
    def sender_ne_receiver(self) -> "TransactionCreate":
        if self.sender_id == self.receiver_id:
            raise ValueError("sender_id and receiver_id must differ")
        return self


class TransactionBatch(BaseModel):
    transactions: list[TransactionCreate] = Field(..., min_length=1, max_length=500)


# ── Auth ───────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Cluster responses ──────────────────────────────────────────────────────

class ClusterSummary(BaseModel):
    id: str
    risk_score: float
    member_count: int
    total_flow: float
    known_mule_count: int
    pattern_flags: dict[str, Any] | None
    status: str
    created_at: datetime


class ClusterDetail(BaseModel):
    id: str
    risk_score: float
    account_ids: list[str]
    member_count: int
    total_flow: float
    known_mule_count: int
    pattern_flags: dict[str, Any] | None
    node_scores: dict[str, float] | None
    shap_values: list[dict] | None
    status: str
    report_text: str | None
    created_at: datetime


class NodeResponse(BaseModel):
    id: str
    label: str
    score: float
    tier: str              # high / elevated / clean
    known_mule: bool
    cluster_id: str | None


class GraphResponse(BaseModel):
    nodes: list[NodeResponse]
    edges: list[dict]      # [{from, to, amount}]


class StatsResponse(BaseModel):
    total_accounts: int
    total_transactions: int
    active_clusters: int
    pending_clusters: int
    window_hours: int = 5


# ── Decision ───────────────────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    action: str = Field(..., pattern=r"^(escalate|monitor|dismiss)$")
    notes: str | None = Field(None, max_length=1000)

    @field_validator("notes")
    @classmethod
    def sanitize_notes(cls, v: str | None) -> str | None:
        return _sanitize_text(v)
