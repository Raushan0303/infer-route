import pytest
from unittest.mock import AsyncMock, MagicMock
from app.ratelimit.quota import QuotaEnforcer


def make_mock_redis(pipe_results):
    """Create a mock redis where pipeline() is sync and returns a mock pipe."""
    
    async def _noop(*args, **kwargs):
        return None

    mock_redis = MagicMock()
    pipe = MagicMock()
    pipe.incrby = _noop
    pipe.execute = AsyncMock(return_value=pipe_results)
    mock_redis.pipeline.return_value = pipe
    return mock_redis, pipe


@pytest.mark.asyncio
async def test_quota_allows_under_limit():
    mock_redis, pipe = make_mock_redis([100, 100])

    enforcer = QuotaEnforcer(mock_redis)
    result = await enforcer.check_and_consume("tenant_a", 100)

    assert result.allowed is True
    assert result.daily_remaining > 0


@pytest.mark.asyncio
async def test_quota_rejects_over_daily_limit():
    mock_redis, pipe = make_mock_redis([2_000_000, 100])

    enforcer = QuotaEnforcer(mock_redis)
    result = await enforcer.check_and_consume("tenant_a", 100)

    assert result.allowed is False
    assert result.daily_remaining == 0
    assert result.reset_at > 0


@pytest.mark.asyncio
async def test_quota_rejects_over_monthly_limit():
    mock_redis, pipe = make_mock_redis([100, 20_000_000])

    enforcer = QuotaEnforcer(mock_redis)
    result = await enforcer.check_and_consume("tenant_a", 100)

    assert result.allowed is False


@pytest.mark.asyncio
async def test_quota_fails_open_on_redis_error():
    mock_redis = AsyncMock()
    mock_redis.pipeline.side_effect = Exception("Redis down")

    enforcer = QuotaEnforcer(mock_redis)
    result = await enforcer.check_and_consume("tenant_a", 100)

    assert result.allowed is True  # fail-open
