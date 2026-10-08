"""Tests for cost computation in InferRoute.

Prices are USD per 1M tokens — the unit providers publish list prices in.

Tests:
1. Cost computation — correct math with split input/output pricing
2. Missing usage — returns $0 when tokens are 0
3. Split input/output pricing — different rates for input vs output
4. Pricing follows the model that served the call, not the provider
"""

import pytest

from app.routing.cost import compute_call_cost
from app.routing.pricing import prices_for
from app.routing.strategies import ProviderHealth


def test_cost_computation():
    """gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output."""
    cost = compute_call_cost(
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        input_tokens=1000,
        output_tokens=500,
    )
    # 1000 * 0.15/1M + 500 * 0.60/1M = 0.00015 + 0.0003
    assert cost == pytest.approx(0.00045)


def test_missing_usage():
    cost = compute_call_cost(0.15, 0.60, input_tokens=0, output_tokens=0)
    assert cost == 0.0


def test_split_input_output_pricing():
    """claude-sonnet-4: $3 / 1M input, $15 / 1M output."""
    cost = compute_call_cost(3.00, 15.00, input_tokens=500, output_tokens=200)
    # 500 * 3/1M + 200 * 15/1M = 0.0015 + 0.003
    assert cost == pytest.approx(0.0045)


def test_zero_pricing_provider():
    """vLLM (local) — $0 pricing, cost is always $0."""
    assert compute_call_cost(0.0, 0.0, input_tokens=10000, output_tokens=5000) == 0.0


def test_one_million_tokens_costs_the_list_price():
    """1M input + 1M output tokens of gpt-4o-mini cost exactly 0.15 + 0.60."""
    assert compute_call_cost(0.15, 0.60, 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_pricing_follows_the_served_model():
    provider = ProviderHealth(provider_id="openai", adapter=None, base_url="",
                              cost_per_1m_input=0.15, cost_per_1m_output=0.60)
    # Same provider, but the request passed "gpt-4o" through unmapped.
    assert prices_for("gpt-4o", provider) == (2.50, 10.00)
    assert prices_for("gpt-4o-mini-2024-07-18", provider) == (0.15, 0.60)
    assert prices_for("llama-3.3-70b-versatile") == (0.59, 0.79)
    # Unknown model → provider's registered rates.
    assert prices_for("email-classifier:latest", provider) == (0.15, 0.60)


def test_hosted_llama_70b_is_more_expensive_than_gpt_4o_mini():
    """The fact that broke the old 'cheap tier' (Groq was labeled cheap)."""
    mini_in, mini_out = prices_for("gpt-4o-mini")
    llama_in, llama_out = prices_for("llama-3.3-70b-versatile")
    assert llama_in > mini_in and llama_out > mini_out
