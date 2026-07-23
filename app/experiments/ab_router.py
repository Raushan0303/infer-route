import hashlib
import logging
from dataclasses import dataclass, field
from app.core.config import settings
from app.core.constants import VARIANT_A, VARIANT_B
from app.observability.metrics import ab_variant_total

logger = logging.getLogger("inferroute")


@dataclass
class VariantOutcome:
    cost: float = 0.0
    latency_ms: float = 0.0
    score: float = 0.0
    count: int = 0


class ABRouter:

    def __init__(self):
        self._split = settings.ab_split_percentage
        self._provider_a = settings.ab_variant_a_provider
        self._provider_b = settings.ab_variant_b_provider
        self._outcomes: dict[str, VariantOutcome] = {
            VARIANT_A: VariantOutcome(),
            VARIANT_B: VariantOutcome(),
        }

    def assign_variant(self, tenant_id: str, user_id: str = "") -> str:
        key = f"{tenant_id}:{user_id}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        bucket = h % 100

        if bucket < int(self._split * 100):
            variant = VARIANT_A
        else:
            variant = VARIANT_B

        ab_variant_total.labels(variant=variant).inc()
        return variant

    def get_provider_for_variant(self, variant: str) -> str:
        if variant == VARIANT_A:
            return self._provider_a
        return self._provider_b

    def record_outcome(self, variant: str, cost: float, latency_ms: float, score: float = None):
        outcome = self._outcomes.get(variant)
        if outcome is None:
            return
        outcome.cost += cost
        outcome.latency_ms += latency_ms
        if score is not None:
            outcome.score += score
        outcome.count += 1

    def get_stats(self) -> dict:
        stats = {}
        for variant, outcome in self._outcomes.items():
            if outcome.count > 0:
                stats[variant] = {
                    "avg_cost": outcome.cost / outcome.count,
                    "avg_latency_ms": outcome.latency_ms / outcome.count,
                    "avg_score": outcome.score / outcome.count if outcome.score else None,
                    "count": outcome.count,
                }
            else:
                stats[variant] = {"count": 0}
        return stats
