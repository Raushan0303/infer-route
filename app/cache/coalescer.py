import asyncio
import logging

logger = logging.getLogger("inferroute")


class RequestCoalescer:

    def __init__(self):
        self._in_flight: dict[str, asyncio.Future] = {}

    async def coalesce(self, key: str, factory) -> any:
        if key in self._in_flight:
            logger.debug("Coalescing request for key=%s", key)
            return await asyncio.shield(self._in_flight[key])

        future = asyncio.get_event_loop().create_future()
        self._in_flight[key] = future

        try:
            result = await factory()
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._in_flight.pop(key, None)

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)
