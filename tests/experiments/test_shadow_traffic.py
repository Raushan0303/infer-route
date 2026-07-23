import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from app.experiments.shadow_traffic import ShadowTrafficManager


@pytest.mark.asyncio
async def test_shadow_traffic_non_blocking():
    """Shadow call takes 2s, but user response should not be delayed."""
    shadow_mgr = ShadowTrafficManager()
    shadow_mgr._percentage = 1.0  # always shadow

    mock_adapter = AsyncMock()
    async def slow_complete(request):
        await asyncio.sleep(2.0)
        return MagicMock(content="shadow result", usage=MagicMock(input_tokens=10, output_tokens=20))

    mock_adapter.complete = slow_complete

    mock_request = MagicMock()

    start = time.monotonic()
    await shadow_mgr.maybe_shadow(mock_request, mock_adapter)
    elapsed = time.monotonic() - start

    # maybe_shadow should return almost immediately (fire-and-forget)
    assert elapsed < 0.1, f"Shadow traffic blocked user response for {elapsed:.2f}s"

    # Wait for shadow call to complete
    await asyncio.sleep(2.5)
    results = shadow_mgr.get_results()
    assert len(results) == 1
    assert results[0]["content"] == "shadow result"


@pytest.mark.asyncio
async def test_shadow_traffic_failure_logged_not_raised():
    shadow_mgr = ShadowTrafficManager()
    shadow_mgr._percentage = 1.0

    mock_adapter = AsyncMock()
    mock_adapter.complete.side_effect = RuntimeError("shadow provider down")

    mock_request = MagicMock()

    # Should not raise
    await shadow_mgr.maybe_shadow(mock_request, mock_adapter)
    await asyncio.sleep(0.2)

    results = shadow_mgr.get_results()
    assert len(results) == 1
    assert results[0]["error"] is not None
    assert "shadow provider down" in results[0]["error"]


@pytest.mark.asyncio
async def test_shadow_traffic_sampled():
    shadow_mgr = ShadowTrafficManager()
    shadow_mgr._percentage = 0.0  # never shadow

    mock_adapter = AsyncMock()
    mock_request = MagicMock()

    await shadow_mgr.maybe_shadow(mock_request, mock_adapter)
    await asyncio.sleep(0.1)

    assert len(shadow_mgr.get_results()) == 0
