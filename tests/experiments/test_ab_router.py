import pytest
from app.experiments.ab_router import ABRouter


def test_sticky_assignment_same_user_same_variant():
    router = ABRouter()
    variant = router.assign_variant("tenant_a", "user_1")
    for _ in range(100):
        assert router.assign_variant("tenant_a", "user_1") == variant


def test_sticky_assignment_different_users_can_differ():
    router = ABRouter()
    variants = set()
    for i in range(100):
        variants.add(router.assign_variant("tenant_a", f"user_{i}"))
    # With 50/50 split, we should see both variants
    assert len(variants) == 2


def test_split_percentage_50_50():
    router = ABRouter()
    counts = {"A": 0, "B": 0}
    for i in range(1000):
        v = router.assign_variant("tenant_a", f"user_{i}")
        counts[v] += 1
    # Should be roughly 50/50 (within 10%)
    assert 400 <= counts["A"] <= 600
    assert 400 <= counts["B"] <= 600


def test_get_provider_for_variant():
    router = ABRouter()
    assert router.get_provider_for_variant("A") == "openai"
    assert router.get_provider_for_variant("B") == "anthropic"


def test_record_outcome_and_stats():
    router = ABRouter()
    router.record_outcome("A", cost=100, latency_ms=500, score=0.9)
    router.record_outcome("A", cost=200, latency_ms=700, score=0.8)
    router.record_outcome("B", cost=150, latency_ms=600, score=0.85)

    stats = router.get_stats()
    assert stats["A"]["count"] == 2
    assert stats["A"]["avg_cost"] == 150.0
    assert stats["A"]["avg_latency_ms"] == 600.0
    assert abs(stats["A"]["avg_score"] - 0.85) < 0.001
    assert stats["B"]["count"] == 1
