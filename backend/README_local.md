# Mule Detection — Backend

Money mule detection system using XGBoost + FastAPI + AI-generated risk reports.

## Quick Start

### 1. Create secrets
See `secrets/IMPORTANT.md` for what to put in each file.

### 2. Train the ML model (and download requirements :)
```bash
pip install -r requirements.txt
python ml/data_gen.py
python ml/feature_pipeline.py
python ml/train.py
```

### 3. Start the backend
```bash
docker compose up --build
```

Swagger docs at http://localhost:8000/docs

### 4. Seed test data
```bash
python ml/data_gen.py --seed-api
```

### 5. Login
Use these credentials in `POST /auth/token`:
- Email: `admin@muledetect.local`
- Password: `hackathon2026`

## API Endpoints
| Endpoint | Purpose |
|---|---|
| POST /auth/token | Login, get JWT |
| GET /api/clusters | List all clusters with risk scores |
| GET /api/cluster/{id} | Single cluster detail |
| GET /api/cluster/{id}/report | AI-generated risk report |
| GET /api/graph | Transaction graph (nodes + edges) |

## ML Artifacts
After training, these are saved to `ml/artifacts/`:
- `mule_scorer.pkl` — trained model
- `feature_names.pkl` — feature list
- `shap_summary.png` — SHAP feature importance chart

## Stack
- FastAPI + PostgreSQL + Alembic
- XGBoost + SHAP
- NetworkX + Louvain (graph clustering)
- OpenRouter (AI reports)
- - PRIMARY_MODEL: "openai/gpt-oss-120b:free"
- - FALLBACK_MODEL: "z-ai/glm-4.5-air:free"
