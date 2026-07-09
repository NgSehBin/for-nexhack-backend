-- This runs automatically when the PostgreSQL container first starts.
-- Creates a read-only user and makes audit_log append-only.

-- Read-only user for future reporting use
CREATE USER mule_readonly WITH PASSWORD 'readonly_strong_pw';
GRANT CONNECT ON DATABASE muledetect TO mule_readonly;
GRANT USAGE ON SCHEMA public TO mule_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mule_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mule_readonly;

-- Make audit_log append-only: the app's own DB user ('mule') can INSERT and
-- SELECT, but never UPDATE or DELETE rows once written.
--
-- NOTE: this script runs once, when the Postgres container is first
-- initialized (docker-entrypoint-initdb.d) — BEFORE Alembic has created any
-- tables. audit_log doesn't exist yet at this point, so a bare REVOKE here
-- would fail. The DO block below makes it a no-op if the table is missing,
-- so this script stays safe to keep around — but the REVOKE will not
-- actually take effect until you run scripts/lock_audit_log.sql (or an
-- Alembic migration) AFTER `alembic upgrade head` has created the table.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'audit_log') THEN
        REVOKE UPDATE, DELETE ON audit_log FROM mule;
    END IF;
END $$;

