import time
import logging
from enum import Enum
from app.core.config import settings
from app.core.constants import CIRCUIT_CLOSED, CIRCUIT_OPEN, CIRCUIT_HALF_OPEN

logger = logging.getLogger("inferroute")


class CircuitBreakerState(Enum):
    CLOSED = CIRCUIT_CLOSED
    OPEN = CIRCUIT_OPEN
    HALF_OPEN = CIRCUIT_HALF_OPEN


class CircuitBreaker:

    def __init__(
        self,
        provider_id: str,
        failure_threshold: int = None,
        cooldown_seconds: int = None,
        half_open_max_calls: int = None,
    ):
        self.provider_id = provider_id
        self._failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self._cooldown = cooldown_seconds or settings.circuit_breaker_cooldown_seconds
        self._half_open_max = half_open_max_calls or settings.circuit_breaker_half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitBreakerState:
        if self._state == CircuitBreakerState.OPEN:
            if time.time() - self._last_failure_time >= self._cooldown:
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit breaker %s: OPEN → HALF_OPEN", self.provider_id)
        return self._state

    @property
    def state_value(self) -> int:
        s = self.state
        if s == CircuitBreakerState.CLOSED:
            return 0
        elif s == CircuitBreakerState.OPEN:
            return 1
        else:
            return 2

    async def can_execute(self) -> bool:
        s = self.state
        if s == CircuitBreakerState.CLOSED:
            return True
        if s == CircuitBreakerState.HALF_OPEN:
            if self._half_open_calls < self._half_open_max:
                self._half_open_calls += 1
                return True
            return False
        return False

    async def record_success(self):
        if self._state == CircuitBreakerState.HALF_OPEN:
            logger.info("Circuit breaker %s: HALF_OPEN → CLOSED (probe succeeded)", self.provider_id)
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0

    async def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker %s: HALF_OPEN → OPEN (probe failed)", self.provider_id)
        elif self._failure_count >= self._failure_threshold:
            self._state = CircuitBreakerState.OPEN
            logger.warning(
                "Circuit breaker %s: CLOSED → OPEN (%d consecutive failures)",
                self.provider_id, self._failure_count,
            )

    def reset(self):
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
