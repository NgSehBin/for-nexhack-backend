from pathlib import Path
from pydantic_settings import BaseSettings


def _read_secret(env_var: str, fallback: str | None = None) -> str:
    import os
    file_path = os.getenv(env_var)
    if file_path and Path(file_path).exists():
        return Path(file_path).read_text().strip()
    if fallback is not None:
        return fallback
    raise RuntimeError(
        f"Secret not found: set {env_var} to a path containing the secret value."
    )


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    PRIMARY_MODEL: str = "openai/gpt-oss-120b:free"
    FALLBACK_MODEL: str = "z-ai/glm-4.5-air:free"
    DATABASE_URL: str = ""
    JWT_SECRET: str = ""

    DB_HOST: str = "db"
    DB_NAME: str = "muledetect"
    DB_USER: str = "mule"
    ENVIRONMENT: str = "development"

    # Graph engine limits
    GRAPH_MAX_NODES: int = 50_000
    GRAPH_MAX_EDGES: int = 200_000
    GRAPH_WINDOW_MINUTES: int = 300   # 5h — matches prototype

    # ML artifacts
    MODEL_PATH: str = "ml/artifacts/mule_scorer.pkl"
    FEATURES_PATH: str = "ml/artifacts/feature_names.pkl"

    # Score thresholds — match prototype
    SCORE_HIGH: int = 50
    SCORE_ELEVATED: int = 28
    CLUSTER_ESCALATE: int = 55
    CLUSTER_MONITOR: int = 35

    model_config = {"env_file": ".env", "extra": "ignore"}


def load_settings() -> Settings:
    s = Settings()

    try:
        s.OPENROUTER_API_KEY = _read_secret("OPENROUTER_KEY_FILE", s.OPENROUTER_API_KEY)
    except RuntimeError:
        pass

    try:
        db_password = _read_secret("DB_PASSWORD_FILE", "mule_secret")
        s.DATABASE_URL = (
            f"postgresql://{s.DB_USER}:{db_password}@{s.DB_HOST}/{s.DB_NAME}"
        )
    except RuntimeError:
        pass

    try:
        s.JWT_SECRET = _read_secret("JWT_SECRET_FILE", "dev_jwt_secret_change_me")
    except RuntimeError:
        s.JWT_SECRET = "dev_jwt_secret_change_me"

    return s


settings = load_settings()
