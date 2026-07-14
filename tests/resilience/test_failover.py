import pytest
from unittest.mock import AsyncMock
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.retry_budget import RetryBudget
from app.resilience.failover import FailoverChain


class MockProvider:
    def __init__(self, provider_id):
        self.provider_id = provider_id


@pytest.mark.asyncio
async def test_failover_skips_open_breaker():
    cb1 = CircuitBreaker("p1", failure_threshold=1, cooldown_seconds=60)
    await cb1.record_failure()  # trip immediately
    cb2 = CircuitBreaker("p2", failure_threshold=5, cooldown_seconds=15)

    budget = RetryBudget(max_concurrent=10)
    chain = FailoverChain({"p1": cb1, "p2": cb2}, budget)

    p1 = MockProvider("p1")
    p2 = MockProvider("p2")

    async def action(provider):
        if provider.provider_id == "p1":
            raise RuntimeError("p1 down")
        return "result_from_p2"

    result = await chain.execute([p1, p2], action)
    assert result == "result_from_p2"


@pytest.mark.asyncio
async def test_failover_all_providers_fail():
    cb1 = CircuitBreaker("p1", failure_threshold=5, cooldown_seconds=15)
    cb2 = CircuitBreaker("p2", failure_threshold=5, cooldown_seconds=15)

    budget = RetryBudget(max_concurrent=10)
    chain = FailoverChain({"p1": cb1, "p2": cb2}, budget)

    p1 = MockProvider("p1")
    p2 = MockProvider("p2")

    async def action(provider):
        raise RuntimeError(f"{provider.provider_id} down")

    with pytest.raises(RuntimeError, match="All providers exhausted"):
        await chain.execute([p1, p2], action)


@pytest.mark.asyncio
async def test_failover_first_provider_succeeds():
    cb1 = CircuitBreaker("p1", failure_threshold=5, cooldown_seconds=15)
    budget = RetryBudget(max_concurrent=10)
    chain = FailoverChain({"p1": cb1}, budget)

    p1 = MockProvider("p1")

    async def action(provider):
        return "success"

    result = await chain.execute([p1], action)
    assert result == "success"
    assert cb1.state.name == "CLOSED"


@pytest.mark.asyncio
async def test_retry_budget_exhausted():
    budget = RetryBudget(max_concurrent=2)
    assert await budget.acquire() is True
    assert await budget.acquire() is True
    assert await budget.acquire() is False  # exhausted
    await budget.release()
    assert await budget.acquire() is True  # freed


@pytest.mark.asyncio
async def test_failover_records_circuit_breaker_failure():
    cb1 = CircuitBreaker("p1", failure_threshold=3, cooldown_seconds=15)
    budget = RetryBudget(max_concurrent=10)
    chain = FailoverChain({"p1": cb1}, budget)

    p1 = MockProvider("p1")

    async def action(provider):
        raise RuntimeError("p1 down")

    with pytest.raises(RuntimeError):
        await chain.execute([p1], action)

    assert cb1.state.name == "CLOSED"  # only 1 failure, threshold is 3
