# Run it with:

# pip install alembic psycopg2-binary --break-system-packages
# alembic revision --autogenerate -m "initial schema: accounts, transactions, clusters, audit_log, analysts"
# alembic upgrade head
#

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Make project root importable so `app...` resolves regardless of where
# `alembic upgrade head` is run from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.db import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_database_url() -> str:
    """
    Build the Postgres URL the same way app/core/config.py does for the
    running app — read the password from the mounted secret file, never
    from an env var or hardcoded string.
    """
    password_file = PROJECT_ROOT / "secrets" / "db_password.txt"
    try:
        db_password = password_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing {password_file}. Create it with the Postgres password "
            "for the 'mule' user before running migrations."
        ) from exc

    db_user = os.getenv("DB_USER", "mule")
    # Default to localhost: migrations are normally generated/run from the
    # host. Override with DB_HOST=db if running `alembic upgrade head`
    # inside the api container, where the docker network resolves the name.
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "muledetect")

    return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


config.set_main_option("sqlalchemy.url", get_database_url())

# app/models/db.py defines Base.metadata with Account, Transaction,
# Cluster, AuditLog, Analyst — autogenerate compares against all of them.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()