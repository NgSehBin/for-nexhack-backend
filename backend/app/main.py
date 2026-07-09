"""
app/main.py — FastAPI application entry point.
Wires together: CORS, rate limiting, all routers, lifespan startup.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.database import engine
from app.models.db import Base
from app.services.scorer import get_scorer
from app.services.graph_engine import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan — runs on startup and shutdown ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all DB tables (Alembic handles migrations in prod;
    # this is a safe fallback for fresh hackathon environments)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified")

    # Pre-load ML scorer so first request isn't slow
    scorer = get_scorer()
    if scorer.model is not None:
        logger.info("XGBoost model loaded at startup")
    else:
        logger.info("No model artifact found — using graph-only scoring")

    # Initialise graph engine singleton
    get_engine()
    logger.info("Graph engine initialised")

    # Seed a default admin analyst if none exists
    _seed_default_analyst()

    yield
    logger.info("Shutdown complete")


def _seed_default_analyst():
    """Create a default admin account on first run so you can log in immediately."""
    from app.core.database import SessionLocal
    from app.core.security import hash_password
    from app.models.db import Analyst, AnalystRole

    db = SessionLocal()
    try:
        if db.query(Analyst).count() == 0:
            # bcrypt hard limit is 72 bytes — keep password short
            password = "hackathon2026"[:72]
            admin = Analyst(
                email="admin@muledetect.local",
                hashed_password=hash_password(password),
                role=AnalystRole.admin,
                tenant_id="default",
            )
            db.add(admin)
            db.commit()
            logger.info("Seeded default admin: admin@muledetect.local / hackathon2026")
    except Exception as e:
        logger.warning(f"Could not seed analyst: {e}")
        db.rollback()
    finally:
        db.close()


# ── App factory ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mule Network Detection API",
    version="1.0.0",
    description="AI-powered mule account detection — NexHack 2026",
    lifespan=lifespan,
)

# SlowAPI state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow the React/HTML frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",   # VS Code Live Server
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security response headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ── Routers ────────────────────────────────────────────────────────────────
from app.routers import auth, ingest, clusters, graph as graph_router

app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(clusters.router)
app.include_router(graph_router.router)


# ── Global error handlers (always return JSON, never HTML) ─────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.exception_handler(422)
async def validation_error(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors() if hasattr(exc, "errors") else str(exc)},
    )


@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}