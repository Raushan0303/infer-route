"""Per-tenant routing config registry — Redis-cached, Postgres-backed.

Developer creates configs (like Portkey), each config lists acceptable providers.
Gateway looks up config by (tenant_id, config_id), caches in Redis (TTL 5 min).
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.database import db

logger = logging.getLogger("inferroute")

CACHE_PREFIX = "routing:config"
CACHE_TTL = 300  # 5 minutes — configs rarely change


@dataclass
class RoutingTarget:
    """A single provider/model target in a routing config."""
    provider: str
    model: str
    tier: str = "standard"  # cheap | standard | premium
    weight: float = 1.0


@dataclass
class RoutingConfig:
    """A per-tenant routing configuration."""
    config_id: str
    tenant_id: str
    name: str = ""
    targets: list[RoutingTarget] = field(default_factory=list)
    fallback_targets: list[RoutingTarget] = field(default_factory=list)
    latency_budget_ms: int | None = None
    cost_ceiling: float | None = None
    retry_attempts: int = 3

    def to_json(self) -> str:
        return json.dumps({
            "config_id": self.config_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "targets": [{"provider": t.provider, "model": t.model, "tier": t.tier, "weight": t.weight} for t in self.targets],
            "fallback_targets": [{"provider": t.provider, "model": t.model, "tier": t.tier, "weight": t.weight} for t in self.fallback_targets],
            "latency_budget_ms": self.latency_budget_ms,
            "cost_ceiling": self.cost_ceiling,
            "retry_attempts": self.retry_attempts,
        })

    @classmethod
    def from_json(cls, data: str) -> "RoutingConfig":
        d = json.loads(data)
        return cls(
            config_id=d["config_id"],
            tenant_id=d["tenant_id"],
            name=d.get("name", ""),
            targets=[RoutingTarget(**t) for t in d.get("targets", [])],
            fallback_targets=[RoutingTarget(**t) for t in d.get("fallback_targets", [])],
            latency_budget_ms=d.get("latency_budget_ms"),
            cost_ceiling=d.get("cost_ceiling"),
            retry_attempts=d.get("retry_attempts", 3),
        )


class ConfigRegistry:
    """Redis-cached, Postgres-backed routing config lookup.

    Lookup path: Redis (1ms) → Postgres (3-5ms) → cache miss.
    Cache hit rate ~99.9% (configs rarely change).
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local_cache: dict[str, RoutingConfig] = {}  # fallback if Redis is down

    def _cache_key(self, tenant_id: str, config_id: str) -> str:
        return f"{CACHE_PREFIX}:{tenant_id}:{config_id}"

    async def get(self, tenant_id: str, config_id: str) -> RoutingConfig | None:
        """Look up a routing config. Returns None if not found."""
        cache_key = self._cache_key(tenant_id, config_id)

        # 1. Try Redis cache (~1ms)
        if self._redis:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    return RoutingConfig.from_json(cached)
            except Exception as e:
                logger.debug("Redis config cache miss: %s", e)

        # 2. Try local in-memory cache (fallback if Redis is down)
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]

        # 3. Query Postgres (~3-5ms)
        try:
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT config_id, tenant_id, name, targets, fallback_targets,
                              latency_budget_ms, cost_ceiling, retry_attempts
                       FROM routing_configs
                       WHERE tenant_id = $1 AND config_id = $2""",
                    tenant_id, config_id,
                )
                if row is None:
                    return None

                config = RoutingConfig(
                    config_id=row["config_id"],
                    tenant_id=row["tenant_id"],
                    name=row["name"] or "",
                    targets=[RoutingTarget(**t) for t in json.loads(row["targets"])],
                    fallback_targets=[RoutingTarget(**t) for t in json.loads(row["fallback_targets"])] if row["fallback_targets"] else [],
                    latency_budget_ms=row["latency_budget_ms"],
                    cost_ceiling=float(row["cost_ceiling"]) if row["cost_ceiling"] else None,
                    retry_attempts=row["retry_attempts"] or 3,
                )

                # Cache in Redis (TTL 5 min)
                if self._redis:
                    try:
                        await self._redis.setex(cache_key, CACHE_TTL, config.to_json())
                    except Exception:
                        pass

                # Cache locally too
                self._local_cache[cache_key] = config
                return config

        except Exception as e:
            logger.error("Failed to lookup routing config: %s", e)
            return None

    async def create(self, config: RoutingConfig) -> bool:
        """Create or update a routing config."""
        try:
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO routing_configs (config_id, tenant_id, name, targets, fallback_targets, latency_budget_ms, cost_ceiling, retry_attempts)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                       ON CONFLICT (tenant_id, config_id) DO UPDATE SET
                           name = EXCLUDED.name,
                           targets = EXCLUDED.targets,
                           fallback_targets = EXCLUDED.fallback_targets,
                           latency_budget_ms = EXCLUDED.latency_budget_ms,
                           cost_ceiling = EXCLUDED.cost_ceiling,
                           retry_attempts = EXCLUDED.retry_attempts,
                           updated_at = NOW()""",
                    config.config_id, config.tenant_id, config.name,
                    json.dumps([{"provider": t.provider, "model": t.model, "tier": t.tier, "weight": t.weight} for t in config.targets]),
                    json.dumps([{"provider": t.provider, "model": t.model, "tier": t.tier, "weight": t.weight} for t in config.fallback_targets]) if config.fallback_targets else "[]",
                    config.latency_budget_ms,
                    config.cost_ceiling,
                    config.retry_attempts,
                )

            # Invalidate cache
            cache_key = self._cache_key(config.tenant_id, config.config_id)
            if self._redis:
                try:
                    await self._redis.delete(cache_key)
                except Exception:
                    pass
            self._local_cache.pop(cache_key, None)

            logger.info("Routing config saved: tenant=%s config=%s", config.tenant_id, config.config_id)
            return True
        except Exception as e:
            logger.error("Failed to save routing config: %s", e)
            return False

    async def list_configs(self, tenant_id: str) -> list[dict]:
        """List all configs for a tenant."""
        try:
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT config_id, name, targets, latency_budget_ms, cost_ceiling, created_at, updated_at
                       FROM routing_configs WHERE tenant_id = $1 ORDER BY created_at DESC""",
                    tenant_id,
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to list routing configs: %s", e)
            return []

    async def delete(self, tenant_id: str, config_id: str) -> bool:
        """Delete a routing config."""
        try:
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM routing_configs WHERE tenant_id = $1 AND config_id = $2",
                    tenant_id, config_id,
                )
            # Invalidate cache
            cache_key = self._cache_key(tenant_id, config_id)
            if self._redis:
                try:
                    await self._redis.delete(cache_key)
                except Exception:
                    pass
            self._local_cache.pop(cache_key, None)
            return result.endswith("1") or "DELETE 1" in result
        except Exception as e:
            logger.error("Failed to delete routing config: %s", e)
            return False
