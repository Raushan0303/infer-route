"""Cost computation for LLM calls.

Computes the dollar cost of a single LLM call using the provider's
registered pricing (cost_per_1k_input, cost_per_1k_output) and the
actual token counts from the provider's usage response.

This lives in InferRoute (not AgentMesh) because InferRoute is the layer
that selects the model and calls the provider — it already knows the
pricing and receives the token counts. AgentMesh just consumes the
cost_usd value from the response.
"""

import logging

logger = logging.getLogger("inferroute")


def compute_call_cost(
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute the dollar cost of a single LLM call.

    Uses the provider's registered pricing and the actual token counts
    from the provider's usage response.

    Args:
        cost_per_1k_input: Provider's input token rate per 1K tokens.
        cost_per_1k_output: Provider's output token rate per 1K tokens.
        input_tokens: Actual input (prompt) tokens from the provider response.
        output_tokens: Actual output (completion) tokens from the provider response.

    Returns:
        Dollar cost of the call, rounded to 6 decimal places.
    """
    cost = (
        (input_tokens / 1000) * cost_per_1k_input
        + (output_tokens / 1000) * cost_per_1k_output
    )
    return round(cost, 6)
