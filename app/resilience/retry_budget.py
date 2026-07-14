import asyncio
import logging

logger = logging.getLogger("inferroute")


class RetryBudget:

    def __init__(self, max_concurrent: int = 50):
        self._max = max_concurrent
        self._current = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self._current < self._max:
                self._current += 1
                return True
            return False

    async def release(self):
        async with self._lock:
            if self._current > 0:
                self._current -= 1

    @property
    def available(self) -> int:
        return self._max - self._current

    @property
    def in_use(self) -> int:
        return self._current
