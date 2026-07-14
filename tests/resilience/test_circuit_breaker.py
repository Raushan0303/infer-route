import pytest
import time
from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState


@pytest.mark.asyncio
async def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker("test_provider")
    assert cb.state == CircuitBreakerState.CLOSED
    assert await cb.can_execute() is True


@pytest.mark.asyncio
async def test_opens_after_threshold_failures():
    cb = CircuitBreaker("test_provider", failure_threshold=5, cooldown_seconds=15)
    for _ in range(5):
        await cb.record_failure()

    assert cb.state == CircuitBreakerState.OPEN
    assert await cb.can_execute() is False


@pytest.mark.asyncio
async def test_half_open_after_cooldown():
    cb = CircuitBreaker("test_provider", failure_threshold=3, cooldown_seconds=1)

    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN

    # Wait for cooldown
    time.sleep(1.1)
    assert cb.state == CircuitBreakerState.HALF_OPEN
    assert await cb.can_execute() is True  # probe allowed


@pytest.mark.asyncio
async def test_half_open_closes_on_success():
    cb = CircuitBreaker("test_provider", failure_threshold=3, cooldown_seconds=1)

    for _ in range(3):
        await cb.record_failure()

    time.sleep(1.1)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    await cb.can_execute()  # consume the probe call
    await cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED


@pytest.mark.asyncio
async def test_half_open_back_to_open_on_failure():
    cb = CircuitBreaker("test_provider", failure_threshold=3, cooldown_seconds=1)

    for _ in range(3):
        await cb.record_failure()

    time.sleep(1.1)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    await cb.can_execute()
    await cb.record_failure()
    assert cb.state == CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_success_resets_failure_count():
    cb = CircuitBreaker("test_provider", failure_threshold=5, cooldown_seconds=15)

    for _ in range(3):
        await cb.record_failure()

    await cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED

    # 3 more failures should not trip (count reset)
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CircuitBreakerState.CLOSED
