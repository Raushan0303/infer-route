"""Dynamic provider router — scores providers in real-time and picks the best one.

Unlike static strategies (round_robin, cost_aware, etc.), the DynamicRouter
combines ALL signals into a single score:
  - Latency (EWMA, lower is better)
  - In-flight count (lower is better)
  - Cost per 1K tokens (lower is better)
  - Circuit breaker state (must be healthy)
  - Weight (developer preference)

The developer doesn't pick a strategy. They just list acceptable providers.
The gateway picks the best one dynamically.
"""
import logging
from app.routing.config_registry import RoutingConfig, RoutingTarget
from app.routing.provider_registry import ProviderRegistry
from app.routing.strategies import ProviderHealth
from app.core.constants import STATUS_HEALTHY

logger = logging.getLogger("inferroute")

# Scoring weights (sum to 1.0)
W_LATENCY = 0.35
W_LOAD = 0.25
W_COST = 0.20
W_WEIGHT = 0.20


class DynamicRouter:
    """Scores providers in real-time and picks the best one for a given config."""

    def __init__(self, registry: ProviderRegistry):
        self._registry = registry

    def _score_provider(
        self,
        provider: ProviderHealth,
        target: RoutingTarget,
        config: RoutingConfig,
    ) -> float:
        """Score a provider from 0.0 to 1.0. Higher is better."""
        # Unhealthy = disqualified
        if provider.status != STATUS_HEALTHY:
            return -1.0

        # Latency score: lower EWMA latency = higher score
        # Normalize: 0ms → 1.0, 2000ms → 0.0
        latency = provider.ewma_latency or 0
        if config.latency_budget_ms:
            # If there's a latency budget, score relative to it
            latency_score = max(0.0, 1.0 - (latency / config.latency_budget_ms))
        else:
            latency_score = max(0.0, 1.0 - (latency / 2000.0))

        # Load score: fewer in-flight = higher score
        # Normalize: 0 in-flight → 1.0, 50 in-flight → 0.0
        load_score = max(0.0, 1.0 - (provider.in_flight / 50.0))

        # Cost score: cheaper = higher score
        # Normalize: $0 → 1.0, $0.05/1K → 0.0
        cost = provider.cost_per_1k_tokens or 0
        if config.cost_ceiling:
            cost_score = max(0.0, 1.0 - (cost / config.cost_ceiling))
        else:
            cost_score = max(0.0, 1.0 - (cost / 0.05))

        # Weight score: developer preference
        weight_score = min(1.0, target.weight / 5.0) if target.weight > 0 else 0.0

        # Combined score
        score = (
            W_LATENCY * latency_score
            + W_LOAD * load_score
            + W_COST * cost_score
            + W_WEIGHT * weight_score
        )

        return score

    def select(
        self,
        config: RoutingConfig,
        circuit_breakers: dict = None,
    ) -> list[ProviderHealth]:
        """Select the best provider for a config. Returns ordered list (best first)."""
        all_providers = {p.provider_id: p for p in self._registry.get_all()}
        scored: list[tuple[float, ProviderHealth]] = []

        for target in config.targets:
            provider = all_providers.get(target.provider)
            if provider is None:
                logger.debug("Target provider not registered: %s", target.provider)
                continue

            # Check circuit breaker
            if circuit_breakers:
                breaker = circuit_breakers.get(target.provider)
                if breaker and breaker.state.name == "OPEN":
                    continue

            score = self._score_provider(provider, target, config)
            if score < 0:
                continue

            scored.append((score, provider))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build ordered list: best targets first, then fallback targets
        ordered = [p for _, p in scored]

        # Add fallback targets (lower priority)
        for target in config.fallback_targets:
            provider = all_providers.get(target.provider)
            if provider and provider not in ordered:
                if circuit_breakers:
                    breaker = circuit_breakers.get(target.provider)
                    if breaker and breaker.state.name == "OPEN":
                        continue
                ordered.append(provider)

        # If no matches, fall back to any healthy provider
        if not ordered:
            ordered = self._registry.get_healthy()

        return ordered
