"""List prices per 1M tokens, keyed by the model that actually served the call.

Pricing must follow the model, not the provider: the same provider can serve
several models (a request for "gpt-4o" passes through the OpenAI adapter
unmapped). When a model is not in the table, the provider's registered
rates are used.

Prices in USD per 1M tokens (input, output). Checked 8 Oct 2026:
  OpenAI   https://platform.openai.com/docs/pricing
  Groq     https://groq.com/pricing
  Anthropic https://www.anthropic.com/pricing
"""

MODEL_PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
}


def prices_for(model: str, provider=None) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for the model that served the call."""
    for name, prices in MODEL_PRICES_PER_1M.items():
        if model == name or model.startswith(name + "-"):
            return prices
    if provider is not None:
        return provider.cost_per_1m_input, provider.cost_per_1m_output
    return 0.0, 0.0
