import pytest
from app.evals.side_by_side import SideBySideEval


@pytest.mark.asyncio
async def test_eval_comparison_a_wins_on_cost():
    eval = SideBySideEval()
    variant_a = [{"cost": 100, "latency_ms": 500, "score": 0.9}, {"cost": 120, "latency_ms": 600, "score": 0.85}]
    variant_b = [{"cost": 200, "latency_ms": 500, "score": 0.9}, {"cost": 220, "latency_ms": 600, "score": 0.85}]

    report = await eval.compare(variant_a, variant_b)
    assert report.cost_winner == "A"
    assert report.variant_a_cost < report.variant_b_cost


@pytest.mark.asyncio
async def test_eval_comparison_b_wins_on_latency():
    eval = SideBySideEval()
    variant_a = [{"cost": 100, "latency_ms": 800, "score": 0.9}, {"cost": 100, "latency_ms": 900, "score": 0.9}]
    variant_b = [{"cost": 100, "latency_ms": 200, "score": 0.9}, {"cost": 100, "latency_ms": 300, "score": 0.9}]

    report = await eval.compare(variant_a, variant_b)
    assert report.latency_winner == "B"


@pytest.mark.asyncio
async def test_eval_comparison_a_wins_on_quality():
    eval = SideBySideEval()
    variant_a = [{"cost": 100, "latency_ms": 500, "score": 0.95}, {"cost": 100, "latency_ms": 500, "score": 0.93}]
    variant_b = [{"cost": 100, "latency_ms": 500, "score": 0.75}, {"cost": 100, "latency_ms": 500, "score": 0.73}]

    report = await eval.compare(variant_a, variant_b)
    assert report.quality_winner == "A"
    assert report.overall_winner == "A"


@pytest.mark.asyncio
async def test_eval_comparison_tie():
    eval = SideBySideEval()
    variant_a = [{"cost": 100, "latency_ms": 500, "score": 0.9}]
    variant_b = [{"cost": 100, "latency_ms": 500, "score": 0.9}]

    report = await eval.compare(variant_a, variant_b)
    assert report.overall_winner == "tie"


@pytest.mark.asyncio
async def test_eval_empty_results():
    eval = SideBySideEval()
    report = await eval.compare([], [])
    assert report.sample_size_a == 0
    assert report.sample_size_b == 0
    assert report.overall_winner == ""
