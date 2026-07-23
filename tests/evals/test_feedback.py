import pytest
from app.evals.feedback import FeedbackStore


@pytest.mark.asyncio
async def test_record_feedback():
    store = FeedbackStore()
    await store.record(trace_id="trace_1", thumbs="up", comment="Great")
    entries = store.get_entries()
    assert len(entries) == 1
    assert entries[0].trace_id == "trace_1"
    assert entries[0].thumbs == "up"
    assert entries[0].comment == "Great"


@pytest.mark.asyncio
async def test_feedback_stats():
    store = FeedbackStore()
    await store.record(trace_id="t1", thumbs="up")
    await store.record(trace_id="t2", thumbs="up")
    await store.record(trace_id="t3", thumbs="down")

    stats = await store.get_stats()
    assert stats["total"] == 3
    assert stats["thumbs_up"] == 2
    assert stats["thumbs_down"] == 1
    assert abs(stats["up_rate"] - 2/3) < 0.01


@pytest.mark.asyncio
async def test_feedback_stats_empty():
    store = FeedbackStore()
    stats = await store.get_stats()
    assert stats["total"] == 0
    assert stats["up_rate"] == 0.0
