"""Cost computation for LLM calls.

Computes the dollar cost of a single LLM call from per-1M-token list prices
(app/routing/pricing.py) and the actual token counts from the provider's
usage response.

This lives in InferRoute (not AgentMesh) because InferRoute is the layer
that selects the model and calls the provider — it already knows the
pricing and receives the token counts. AgentMesh just consumes the
cost_usd value from the response.
"""

import logging

logger = logging.getLogger("inferroute")


def compute_call_cost(
    cost_per_1m_input: float,
    cost_per_1m_output: float,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute the dollar cost of a single LLM call.

    Args:
        cost_per_1m_input: USD per 1M input tokens (the provider's list price unit).
        cost_per_1m_output: USD per 1M output tokens.
        input_tokens: Actual input (prompt) tokens from the provider response.
        output_tokens: Actual output (completion) tokens from the provider response.

    Returns:
        Dollar cost of the call, rounded to 8 decimal places.
    """
    cost = (
        input_tokens * cost_per_1m_input
        + output_tokens * cost_per_1m_output
    ) / 1_000_000
    return round(cost, 8)
