import pytest
import asyncio
from unittest.mock import AsyncMock
from app.cache.coalescer import RequestCoalescer


@pytest.mark.asyncio
async def test_coalesce_collapses_concurrent_requests():
    """50 concurrent identical requests should result in exactly 1 upstream call."""
    call_count = 0

    async def factory():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # simulate upstream latency
        return f"result_{call_count}"

    coalescer = RequestCoalescer()
    key = "tenant_a:some_hash"

    # Fire 50 concurrent requests with the same key
    results = await asyncio.gather(
        *[coalescer.coalesce(key, factory) for _ in range(50)]
    )

    assert call_count == 1  # only 1 upstream call
    assert all(r == "result_1" for r in results)  # all 50 got the same result
    assert len(set(results)) == 1


@pytest.mark.asyncio
async def test_coalesce_different_keys_independent():
    """Different keys should not coalesce — each gets its own upstream call."""
    call_count = 0

    async def factory():
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"

    coalescer = RequestCoalescer()

    results = await asyncio.gather(
        coalescer.coalesce("key_a", factory),
        coalescer.coalesce("key_b", factory),
        coalescer.coalesce("key_c", factory),
    )

    assert call_count == 3  # 3 different keys = 3 upstream calls
    assert len(set(results)) == 3


@pytest.mark.asyncio
async def test_coalesce_error_propagation():
    """If upstream fails, all coalesced requests should get the error."""
    async def factory():
        raise RuntimeError("upstream failed")

    coalescer = RequestCoalescer()
    key = "tenant_a:error_hash"

    results = await asyncio.gather(
        *[coalescer.coalesce(key, factory) for _ in range(10)],
        return_exceptions=True,
    )

    assert all(isinstance(r, RuntimeError) for r in results)
    assert all("upstream failed" in str(r) for r in results)


@pytest.mark.asyncio
async def test_coalesce_key_cleared_after_completion():
    """After a request completes, the in-flight key should be cleared."""
    async def factory():
        return "done"

    coalescer = RequestCoalescer()
    await coalescer.coalesce("key_a", factory)
    assert "key_a" not in coalescer._in_flight
    assert coalescer.in_flight_count == 0


@pytest.mark.asyncio
async def test_coalesce_key_cleared_after_error():
    """After a request fails, the in-flight key should be cleared."""
    async def factory():
        raise ValueError("oops")

    coalescer = RequestCoalescer()
    try:
        await coalescer.coalesce("key_a", factory)
    except ValueError:
        pass
    assert "key_a" not in coalescer._in_flight
