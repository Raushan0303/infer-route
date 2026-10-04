"""Tenant isolation of the rate limiter, through the REAL middleware stack.

Builds the real `app.main.app` (so middleware registration order is the
production order), attaches a TokenBucketRateLimiter backed by a real
Redis, and checks that one tenant exhausting its bucket does not throttle
another tenant.

Requires Redis at INFERROUTE_TEST_REDIS_URL (default redis://localhost:6390/0);
skipped otherwise.
"""
import os
import uuid

import httpx
import pytest
import pytest_asyncio
import redis.asyncio as aioredis

from app.main import app
from app.ratelimit.token_bucket import TokenBucketRateLimiter

TEST_REDIS_URL = os.environ.get("INFERROUTE_TEST_REDIS_URL", "redis://localhost:6390/0")


@pytest_asyncio.fixture
async def limited_app():
    client = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        pytest.skip(f"Redis not reachable at {TEST_REDIS_URL}")
    limiter = TokenBucketRateLimiter(client)
    limiter._capacity = 5
    limiter._refill_rate = 0.001  # effectively no refill during the test
    app.state.rate_limiter = limiter
    yield app
    app.state.rate_limiter = None
    await client.aclose()


async def _send(asgi_app, api_key: str, n: int) -> list[int]:
    transport = httpx.ASGITransport(app=asgi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        codes = []
        for _ in range(n):
            # Unknown path: no route logic runs, only the middleware stack.
            r = await c.post("/v1/isolation-probe", headers={"Authorization": f"Bearer {api_key}"})
            codes.append(r.status_code)
        return codes


@pytest.mark.asyncio
async def test_rate_limit_bucket_is_per_tenant(limited_app):
    run = uuid.uuid4().hex[:8]
    tenant_a, tenant_b = f"tenant-a-{run}", f"tenant-b-{run}"

    a_codes = await _send(limited_app, tenant_a, 20)
    b_codes = await _send(limited_app, tenant_b, 5)

    # Tenant A is limited after its 5-token bucket is empty.
    assert a_codes.count(429) == 15
    # Tenant B has its own bucket: none of its 5 requests are throttled.
    assert b_codes.count(429) == 0, f"tenant B throttled by tenant A's traffic: {b_codes}"


@pytest.mark.asyncio
async def test_rate_limiter_sees_authenticated_tenant(limited_app, monkeypatch):
    seen = []
    original = limited_app.state.rate_limiter.check

    async def spy(tenant_id, cost=None):
        seen.append(tenant_id)
        return await original(tenant_id, cost)

    monkeypatch.setattr(limited_app.state.rate_limiter, "check", spy)
    await _send(limited_app, f"key-{uuid.uuid4().hex[:6]}", 1)
    assert seen and seen[0] != "anonymous", f"rate limiter keyed on {seen}"
