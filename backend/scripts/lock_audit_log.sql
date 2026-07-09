-- Run this AFTER `alembic upgrade head` — it requires audit_log to already
-- exist, which db_init.sql cannot guarantee since it runs before migrations.
--
-- Usage:
--   alembic upgrade head
--   docker compose exec -T db psql -U mule -d muledetect -f - < scripts/lock_audit_log.sql
--
-- (or, if running Postgres without Docker:)
--   psql -U mule -d muledetect -f scripts/lock_audit_log.sql

REVOKE UPDATE, DELETE ON audit_log FROM mule;
