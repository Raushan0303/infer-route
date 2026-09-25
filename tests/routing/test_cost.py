"""Tests for cost computation in InferRoute.

Tests:
1. Cost computation — correct math with split input/output pricing
2. Missing usage — returns $0 when tokens are 0
3. Cache hit cost — $0 (the original call already paid)
4. Split input/output pricing — different rates for input vs output
"""

import pytest

from app.routing.cost import compute_call_cost


def test_cost_computation():
    """Cost computation — correct math with split pricing."""
    # OpenAI gpt-4o-mini: $0.150/1K input, $0.600/1K output
    # 1000 input tokens, 500 output tokens
    cost = compute_call_cost(
        cost_per_1k_input=0.150,
        cost_per_1k_output=0.600,
        input_tokens=1000,
        output_tokens=500,
    )
    # Expected: (1000/1000)*0.150 + (500/1000)*0.600 = 0.150 + 0.300 = 0.450
    assert cost == pytest.approx(0.45, abs=0.001)


def test_missing_usage():
    """Missing usage — returns $0 when tokens are 0."""
    cost = compute_call_cost(
        cost_per_1k_input=0.150,
        cost_per_1k_output=0.600,
        input_tokens=0,
        output_tokens=0,
    )
    assert cost == 0.0


def test_cache_hit_cost():
    """Cache hit cost — $0 (the original call already paid).

    The gateway handles this by not calling compute_call_cost on cache hits.
    This test verifies the function returns $0 when called with 0 tokens
    (which is what happens for a cache hit).
    """
    cost = compute_call_cost(
        cost_per_1k_input=0.150,
        cost_per_1k_output=0.600,
        input_tokens=0,
        output_tokens=0,
    )
    assert cost == 0.0


def test_split_input_output_pricing():
    """Split input/output pricing — different rates for input vs output."""
    # Anthropic claude-3.5-sonnet: $3.00/1K input, $15.00/1K output
    # 500 input tokens, 200 output tokens
    cost = compute_call_cost(
        cost_per_1k_input=3.000,
        cost_per_1k_output=15.000,
        input_tokens=500,
        output_tokens=200,
    )
    # Expected: (500/1000)*3.0 + (200/1000)*15.0 = 1.5 + 3.0 = 4.5
    assert cost == pytest.approx(4.5, abs=0.001)


def test_zero_pricing_provider():
    """vLLM (local) — $0 pricing, cost is always $0."""
    cost = compute_call_cost(
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        input_tokens=10000,
        output_tokens=5000,
    )
    assert cost == 0.0


def test_large_token_count():
    """Large token count — cost scales linearly."""
    cost = compute_call_cost(
        cost_per_1k_input=0.150,
        cost_per_1k_output=0.600,
        input_tokens=1_000_000,  # 1M tokens
        output_tokens=100_000,
    )
    # Expected: (1M/1000)*0.150 + (100K/1000)*0.600 = 150 + 60 = 210
    assert cost == pytest.approx(210.0, abs=0.01)
