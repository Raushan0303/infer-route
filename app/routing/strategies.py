from typing import Protocol
from dataclasses import dataclass, field
from itertools import cycle
import random
import time
import re
import asyncio
import logging

from app.core.config import settings

logger = logging.getLogger("inferroute")


@dataclass
class ProviderHealth:
    provider_id: str
    adapter: object
    base_url: str
    status: str = "HEALTHY"
    in_flight: int = 0
    ewma_latency: float = 0.0
    weight: float = 1.0
    last_check_time: float = field(default_factory=time.time)
    last_latency_ms: float = 0.0
    cost_per_1k_tokens: float = 0.0
    tier: str = "standard"  # "cheap", "standard", "premium"


class LoadBalancingStrategy(Protocol):

    def select(self, providers: list[ProviderHealth], request: object = None) -> ProviderHealth | None:
        ...


class RoundRobin:

    def __init__(self):
        self._cycle = None
        self._last_providers = None

    async def select(self, providers: list[ProviderHealth], request: object = None) -> ProviderHealth | None:
        healthy = [p for p in providers if p.status == "HEALTHY"]
        if not healthy:
            return None
        healthy_ids = [p.provider_id for p in healthy]
        if self._last_providers != healthy_ids:
            self._cycle = cycle(healthy)
            self._last_providers = healthy_ids
        return next(self._cycle)


class LeastConnections:

    async def select(self, providers: list[ProviderHealth], request: object = None) -> ProviderHealth | None:
        healthy = [p for p in providers if p.status == "HEALTHY"]
        if not healthy:
            return None
        return min(healthy, key=lambda p: p.in_flight)


class Weighted:

    async def select(self, providers: list[ProviderHealth], request: object = None) -> ProviderHealth | None:
        healthy = [p for p in providers if p.status == "HEALTHY"]
        if not healthy:
            return None
        weights = [p.weight for p in healthy]
        return random.choices(healthy, weights=weights, k=1)[0]


class LatencyAware:

    def __init__(self, alpha: float = None):
        self._alpha = alpha or settings.ewma_alpha

    async def select(self, providers: list[ProviderHealth], request: object = None) -> ProviderHealth | None:
        healthy = [p for p in providers if p.status == "HEALTHY"]
        if not healthy:
            return None
        if all(p.ewma_latency == 0.0 for p in healthy):
            return random.choice(healthy)
        return min(healthy, key=lambda p: p.ewma_latency)

    def update_latency(self, provider: ProviderHealth, latency_ms: float):
        provider.ewma_latency = (
            self._alpha * latency_ms + (1 - self._alpha) * provider.ewma_latency
        )
        provider.last_latency_ms = latency_ms


class CostAware:

    async def select(self, providers: list[ProviderHealth], request: object = None) -> ProviderHealth | None:
        healthy = [p for p in providers if p.status == "HEALTHY"]
        if not healthy:
            return None
        return min(healthy, key=lambda p: p.cost_per_1k_tokens)


COMPLEXITY_KEYWORDS = {
    "complex": [
        "analyze", "reasoning", "chain of thought", "step by step",
        "explain why", "design", "architect", "implement", "debug",
        "optimize", "refactor", "compare and contrast", "evaluate",
        "derive", "prove", "synthesize", "multi-step", "plan",
        "strategy", "algorithm", "complex", "nuanced", "critical thinking",
        "write code", "write a function", "write a class", "write a script",
        "code review", "system design", "trade-off", "tradeoff",
    ],
    "simple": [
        "hi", "hello", "hey", "thanks", "thank you", "ok", "yes", "no",
        "what is", "define", "list", "summarize", "translate", "count",
        "format", "extract", "classify", "label", "sentiment",
        "what time", "what date", "how many", "calculate", "convert",
    ],
}


class IntelligenceAware:

    COMPLEX_THRESHOLD = 3
    SIMPLE_THRESHOLD = 2

    def __init__(self, classifier=None):
        self._classifier = classifier

    async def classify_complexity(self, request: object) -> tuple[str, str]:
        """Returns (complexity, source) where source is 'heuristic', 'cache', or 'llm'."""
        if self._classifier is None:
            complexity, confidence = self._heuristic_classify(request)
            return complexity, "heuristic"
        complexity, confidence, source = await self._classifier.classify(request)
        return complexity, source

    def _heuristic_classify(self, request: object) -> tuple[str, float]:
        if request is None or not hasattr(request, "messages"):
            return "medium", 0.4

        prompt = " ".join(m.content.lower() for m in request.messages if hasattr(m, "content"))

        complex_score = sum(1 for kw in COMPLEXITY_KEYWORDS["complex"] if re.search(r'\b' + re.escape(kw) + r'\b', prompt))
        simple_score = sum(1 for kw in COMPLEXITY_KEYWORDS["simple"] if re.search(r'\b' + re.escape(kw) + r'\b', prompt))

        if request.max_tokens and request.max_tokens > 2000:
            complex_score += 2

        prompt_len = len(prompt)
        if prompt_len > 2000:
            complex_score += 2
        elif prompt_len > 500:
            complex_score += 1
        elif prompt_len < 100:
            simple_score += 1

        if complex_score >= 3 and simple_score == 0:
            return "complex", 0.9
        if simple_score >= 2 and complex_score == 0:
            return "simple", 0.9
        if complex_score >= 4:
            return "complex", 0.85
        if simple_score >= 3 and complex_score <= 1:
            return "simple", 0.80
        if complex_score >= 3:
            return "complex", 0.5
        if simple_score >= 2:
            return "simple", 0.5
        return "medium", 0.4

    async def select(self, providers: list[ProviderHealth], request: object = None) -> ProviderHealth | None:
        healthy = [p for p in providers if p.status == "HEALTHY"]
        if not healthy:
            return None

        complexity, source = await self.classify_complexity(request)

        prompt_preview = ""
        if request and hasattr(request, "messages"):
            prompt_preview = request.messages[0].content[:60] if request.messages else ""
        logger.info(
            "IntelligenceAware: complexity=%s source=%s prompt='%s...'",
            complexity, source, prompt_preview,
        )

        tier_map = {
            "simple": ["cheap", "standard", "premium"],
            "medium": ["standard", "premium", "cheap"],
            "complex": ["premium", "standard", "cheap"],
        }

        preferred_tiers = tier_map[complexity]

        for tier in preferred_tiers:
            tier_providers = [p for p in healthy if p.tier == tier]
            if tier_providers:
                if tier == "premium":
                    return max(tier_providers, key=lambda p: p.weight)
                elif tier == "cheap":
                    return min(tier_providers, key=lambda p: p.cost_per_1k_tokens)
                else:
                    return min(tier_providers, key=lambda p: p.in_flight)

        return min(healthy, key=lambda p: p.in_flight)


STRATEGY_MAP = {
    "round_robin": RoundRobin,
    "least_connections": LeastConnections,
    "weighted": Weighted,
    "latency_aware": LatencyAware,
    "cost_aware": CostAware,
    "intelligence_aware": IntelligenceAware,
}


def create_strategy(name: str) -> LoadBalancingStrategy:
    cls = STRATEGY_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}"
        )
    return cls()
