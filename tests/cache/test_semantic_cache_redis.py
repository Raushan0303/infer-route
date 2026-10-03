"""SemanticCache against a real Redis: lookups must not evict entries.

Before the fix, XFetch early expiry was rolled for EVERY scanned entry and
deleted the ones that "expired", so each lookup randomly evicted entries
(1000 → 822 after 30 lookups in benchmarks/cache_latency.py).
"""
import os
import uuid

import pytest
import redis.asyncio as aioredis

from app.cache import semantic_cache as sc
from app.cache.semantic_cache import SemanticCache
from app.gateway.models import InferRouteMessage, InferRouteRequest, InferRouteResponse, InferRouteUsage

REDIS_URL = os.environ.get("INFERROUTE_TEST_REDIS_URL", "redis://localhost:6390/0")


class Emb:
    async def embed(self, text):
        return [float(ord(c)) for c in text.ljust(8)[:8]]


def req(t):
    return InferRouteRequest(model="m", messages=[InferRouteMessage(role="user", content=t)])


RESP = InferRouteResponse(content="x", model="m", provider="p", usage=InferRouteUsage(input_tokens=1, output_tokens=1))


@pytest.mark.asyncio
async def test_lookup_does_not_evict_unmatched_entries(monkeypatch):
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await r.ping()
    except Exception:
        pytest.skip("redis not reachable")
    cache, tenant = SemanticCache(r, Emb()), f"t-{uuid.uuid4().hex[:6]}"
    for i in range(20):
        await cache.set(tenant, req(f"prompt{i:02d}"), RESP)
    # Force every early-expiry roll to say "expired".
    monkeypatch.setattr(sc, "xfetch_expired", lambda created_at, ttl: True)
    await cache.get(tenant, req("prompt05"))
    # Only the matched entry may be dropped (its caller refreshes it).
    assert await r.scard(cache._index_key(tenant)) == 19
    await r.aclose()


@pytest.mark.asyncio
async def test_semantic_hit_uses_one_mget_round_trip():
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await r.ping()
    except Exception:
        pytest.skip("redis not reachable")
    cache, tenant = SemanticCache(r, Emb()), f"t-{uuid.uuid4().hex[:6]}"
    for i in range(20):
        await cache.set(tenant, req(f"prompt{i:02d}"), RESP)
    calls = {"get": 0, "mget": 0}
    real_get, real_mget = r.get, r.mget

    async def get(*a, **k):
        calls["get"] += 1
        return await real_get(*a, **k)

    async def mget(*a, **k):
        calls["mget"] += 1
        return await real_mget(*a, **k)

    r.get, r.mget = get, mget
    assert (await cache.get(tenant, req("prompt07"))) is not None
    assert calls == {"get": 0, "mget": 1}
    await r.aclose()
