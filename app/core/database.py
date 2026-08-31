import logging
import asyncpg
from app.core.config import settings

logger = logging.getLogger("inferroute")


SCHEMA_SQL = """
-- Feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(64) NOT NULL,
    thumbs VARCHAR(10) NOT NULL,
    comment TEXT DEFAULT '',
    model VARCHAR(128) DEFAULT '',
    provider VARCHAR(64) DEFAULT '',
    variant VARCHAR(10) DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Usage tracking table
CREATE TABLE IF NOT EXISTS usage_log (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(128) NOT NULL,
    provider VARCHAR(64) NOT NULL,
    model VARCHAR(128) NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    routing_decision VARCHAR(128) DEFAULT '',
    trace_id VARCHAR(64) DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RAG documents table with pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_documents (
    id VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(128) NOT NULL,
    namespace VARCHAR(256) NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Routing configs table (Portkey-style per-tenant config)
CREATE TABLE IF NOT EXISTS routing_configs (
    id SERIAL PRIMARY KEY,
    config_id VARCHAR(128) NOT NULL,
    tenant_id VARCHAR(128) NOT NULL,
    name VARCHAR(256) DEFAULT '',
    targets JSONB NOT NULL DEFAULT '[]',
    latency_budget_ms INTEGER DEFAULT NULL,
    cost_ceiling DECIMAL(10, 6) DEFAULT NULL,
    fallback_targets JSONB DEFAULT '[]',
    retry_attempts INTEGER DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, config_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_feedback_trace_id ON feedback(trace_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_usage_log_tenant_id ON usage_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_usage_log_created_at ON usage_log(created_at);
CREATE INDEX IF NOT EXISTS idx_rag_documents_tenant_namespace ON rag_documents(tenant_id, namespace);
CREATE INDEX IF NOT EXISTS idx_routing_configs_tenant ON routing_configs(tenant_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_routing_configs_lookup ON routing_configs(tenant_id, config_id);
"""


class Database:
    """Async Postgres connection pool with schema auto-initialization."""

    _pool: asyncpg.Pool | None = None

    async def init(self, dsn: str = None, min_size: int = 2, max_size: int = 10):
        dsn = dsn or settings.postgres_dsn
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=30,
        )

        # Initialize schema
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

        logger.info("Postgres initialized: %s", dsn)

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            await self.init()
        return self._pool

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Postgres connection pool closed")


db = Database()
