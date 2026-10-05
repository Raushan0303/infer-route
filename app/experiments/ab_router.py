import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from app.core.config import settings
from app.core.constants import VARIANT_A, VARIANT_B
from app.observability.metrics import ab_variant_total

logger = logging.getLogger("inferroute")

SWAP_LOG_KEY = "experiments:swaps"


@dataclass
class VariantOutcome:
    cost: float = 0.0  # USD, billed calls only
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
        self._served_by: dict[str, dict[str, int]] = {VARIANT_A: {}, VARIANT_B: {}}
        self._swaps: list[dict] = []

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

    def record_outcome(self, variant: str, cost: float, latency_ms: float, score: float = None,
                       provider: str = None):
        outcome = self._outcomes.get(variant)
        if outcome is None:
            return
        if provider:
            served = self._served_by[variant]
            served[provider] = served.get(provider, 0) + 1
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
            stats[variant]["provider"] = self.get_provider_for_variant(variant)
            stats[variant]["served_by"] = dict(self._served_by[variant])
        return stats

    async def promote(self, reason: str = "", new_challenger: str = None,
                      min_samples: int = 0, redis=None) -> dict:
        """Make the challenger (B) the control (A) and log the swap."""
        stats = self.get_stats()
        for v in (VARIANT_A, VARIANT_B):
            if stats[v]["count"] < min_samples:
                raise ValueError(
                    f"variant {v} has {stats[v]['count']} samples < min_samples={min_samples}"
                )
        swap = {
            "at": time.time(),
            "from_provider": self._provider_a,
            "to_provider": self._provider_b,
            "reason": reason,
            "stats_at_swap": stats,
        }
        self._provider_a = self._provider_b
        if new_challenger:
            self._provider_b = new_challenger
        swap["new_challenger"] = self._provider_b
        # Reset outcomes: the next experiment starts from zero.
        self._outcomes = {VARIANT_A: VariantOutcome(), VARIANT_B: VariantOutcome()}
        self._served_by = {VARIANT_A: {}, VARIANT_B: {}}
        self._swaps.append(swap)
        if redis is not None:
            try:
                await redis.rpush(SWAP_LOG_KEY, json.dumps(swap))
            except Exception as e:
                logger.warning("Swap log write failed (kept in memory): %s", e)
        logger.info("AB_PROMOTE from=%s to=%s reason=%s", swap["from_provider"], swap["to_provider"], reason)
        return swap

    async def list_swaps(self, redis=None) -> list[dict]:
        if redis is not None:
            try:
                return [json.loads(x) for x in await redis.lrange(SWAP_LOG_KEY, 0, -1)]
            except Exception as e:
                logger.warning("Swap log read failed (using memory): %s", e)
        return list(self._swaps)
