import asyncio
import logging
import time
from app.core.config import settings
from app.observability.metrics import shadow_traffic_total

logger = logging.getLogger("inferroute")


class ShadowTrafficManager:

    def __init__(self):
        self._percentage = settings.shadow_traffic_percentage
        self._target_provider = settings.shadow_target_provider
        self._results: list[dict] = []

    def _should_shadow(self) -> bool:
        import random
        return random.random() < self._percentage

    async def maybe_shadow(self, request, adapter) -> None:
        if not self._should_shadow():
            return

        shadow_traffic_total.labels(target_provider=self._target_provider).inc()

        async def _shadow_call():
            start = time.monotonic()
            try:
                response = await adapter.complete(request)
                latency = (time.monotonic() - start) * 1000
                self._results.append({
                    "provider": self._target_provider,
                    "content": response.content,
                    "latency_ms": latency,
                    "cost": response.usage.input_tokens + response.usage.output_tokens,
                    "error": None,
                })
                logger.debug("Shadow traffic completed: latency=%.1fms", latency)
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                self._results.append({
                    "provider": self._target_provider,
                    "content": None,
                    "latency_ms": latency,
                    "cost": 0,
                    "error": str(e),
                })
                logger.warning("Shadow traffic failed: %s", e)

        asyncio.create_task(_shadow_call())

    def get_results(self) -> list[dict]:
        return list(self._results)

    def clear_results(self):
        self._results.clear()
