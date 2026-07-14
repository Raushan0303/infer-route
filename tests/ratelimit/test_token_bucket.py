import pytest
import asyncio
from unittest.mock import AsyncMock
from app.ratelimit.token_bucket import TokenBucketRateLimiter, RateLimitResult


@pytest.mark.asyncio
async def test_rate_limit_allows_under_capacity():
    mock_redis = AsyncMock()
    mock_redis.script_load.return_value = "sha123"
    mock_redis.evalsha.return_value = [1, 99, 0]

    limiter = TokenBucketRateLimiter(mock_redis)
    result = await limiter.check("tenant_a")
    assert result.allowed is True
    assert result.remaining == 99


@pytest.mark.asyncio
async def test_rate_limit_rejects_over_capacity():
    mock_redis = AsyncMock()
    mock_redis.script_load.return_value = "sha123"
    mock_redis.evalsha.return_value = [0, 0, 5]

    limiter = TokenBucketRateLimiter(mock_redis)
    result = await limiter.check("tenant_a")
    assert result.allowed is False
    assert result.retry_after == 5


@pytest.mark.asyncio
async def test_rate_limit_fails_open_on_redis_error():
    mock_redis = AsyncMock()
    mock_redis.script_load.side_effect = Exception("Redis down")

    limiter = TokenBucketRateLimiter(mock_redis)
    result = await limiter.check("tenant_a")
    assert result.allowed is True  # fail-open


@pytest.mark.asyncio
async def test_rate_limit_atomicity_1000_concurrent():
    """Simulate 1000 concurrent requests against a limit of 100.
    With a real Redis + Lua script, exactly 100 would pass.
    Here we simulate the Lua atomic behavior."""
    mock_redis = AsyncMock()
    mock_redis.script_load.return_value = "sha123"

    # Simulate atomic counter: first 100 allowed, rest rejected
    call_count = 0

    async def evalsha_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 100:
            return [1, 100 - call_count, 0]
        return [0, 0, 10]

    mock_redis.evalsha.side_effect = evalsha_side_effect

    limiter = TokenBucketRateLimiter(mock_redis)

    results = await asyncio.gather(
        *[limiter.check("tenant_a") for _ in range(1000)]
    )

    allowed = sum(1 for r in results if r.allowed)
    rejected = sum(1 for r in results if not r.allowed)

    assert allowed == 100
    assert rejected == 900
