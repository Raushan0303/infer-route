import asyncio
import logging
import time

from app.routing.provider_registry import ProviderRegistry
from app.core.constants import DEFAULT_HEALTH_CHECK_INTERVAL

logger = logging.getLogger("inferroute")


class HealthChecker:

    def __init__(
        self,
        registry: ProviderRegistry,
        interval: int = DEFAULT_HEALTH_CHECK_INTERVAL,
    ):
        self._registry = registry
        self._interval = interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Health checker started (interval=%ds)", self._interval)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _loop(self):
        while self._running:
            for provider in self._registry.get_all():
                try:
                    healthy = await provider.adapter.health_check()
                    if healthy:
                        self._registry.mark_healthy(provider.provider_id)
                    else:
                        self._registry.mark_unhealthy(provider.provider_id)
                    provider.last_check_time = time.time()
                except Exception as e:
                    self._registry.mark_unhealthy(provider.provider_id)
                    logger.debug(
                        "Health check failed for %s: %s",
                        provider.provider_id,
                        e,
                    )
            await asyncio.sleep(self._interval)
