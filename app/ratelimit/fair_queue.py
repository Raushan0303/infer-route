import time
import logging
from app.core.config import settings
from app.core.constants import FAIR_QUEUE_KEY_PREFIX

logger = logging.getLogger("inferroute")


class WeightedFairQueue:

    def __init__(self, redis, max_concurrent: int = 100):
        self._redis = redis
        self._max_concurrent = max_concurrent
        self._tenant_weights: dict[str, float] = {}
        self._in_flight: dict[str, int] = {}

    def set_weight(self, tenant_id: str, weight: float):
        self._tenant_weights[tenant_id] = weight

    def _get_weight(self, tenant_id: str) -> float:
        return self._tenant_weights.get(tenant_id, 1.0)

    def _total_weight(self) -> float:
        return sum(self._get_weight(t) for t in self._in_flight if self._in_flight[t] > 0) or 1.0

    async def acquire(self, tenant_id: str) -> bool:
        total_in_flight = sum(self._in_flight.values())
        if total_in_flight < self._max_concurrent:
            self._in_flight[tenant_id] = self._in_flight.get(tenant_id, 0) + 1
            return True

        my_weight = self._get_weight(tenant_id)
        total_w = self._total_weight()
        my_share = my_weight / total_w
        my_current = self._in_flight.get(tenant_id, 0)
        my_allowed = max(1, int(self._max_concurrent * my_share))

        if my_current < my_allowed:
            self._in_flight[tenant_id] = my_current + 1
            return True

        return False

    async def release(self, tenant_id: str):
        if tenant_id in self._in_flight and self._in_flight[tenant_id] > 0:
            self._in_flight[tenant_id] -= 1
            if self._in_flight[tenant_id] == 0:
                del self._in_flight[tenant_id]

    @property
    def total_in_flight(self) -> int:
        return sum(self._in_flight.values())
