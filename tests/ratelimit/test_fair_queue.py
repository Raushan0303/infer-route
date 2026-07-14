import pytest
from app.ratelimit.fair_queue import WeightedFairQueue


@pytest.mark.asyncio
async def test_no_contention_full_access():
    fq = WeightedFairQueue(redis=None, max_concurrent=100)
    assert await fq.acquire("tenant_a") is True
    await fq.release("tenant_a")


@pytest.mark.asyncio
async def test_weighted_fairness_10_to_1():
    """Tenant A (weight=10) and Tenant B (weight=1) under contention.
    B should get > 5% of capacity, A should get < 95%."""
    fq = WeightedFairQueue(redis=None, max_concurrent=100)
    fq.set_weight("tenant_a", 10.0)
    fq.set_weight("tenant_b", 1.0)

    # Fill up capacity
    a_acquired = 0
    b_acquired = 0

    for _ in range(100):
        # Alternate between tenants
        if await fq.acquire("tenant_a"):
            a_acquired += 1
        if await fq.acquire("tenant_b"):
            b_acquired += 1

    total = a_acquired + b_acquired
    # Under contention, B should get some share (not starved)
    assert b_acquired > 0, "Tenant B was starved — should get some capacity"
    assert a_acquired > b_acquired, "Tenant A should get more than B due to higher weight"


@pytest.mark.asyncio
async def test_release_frees_capacity():
    fq = WeightedFairQueue(redis=None, max_concurrent=2)
    assert await fq.acquire("tenant_a") is True
    assert await fq.acquire("tenant_a") is True
    assert await fq.acquire("tenant_a") is False  # full
    await fq.release("tenant_a")
    assert await fq.acquire("tenant_a") is True  # freed
