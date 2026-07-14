import logging
from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState
from app.resilience.retry_budget import RetryBudget
from app.observability.metrics import failover_total

logger = logging.getLogger("inferroute")


class FailoverChain:

    def __init__(self, circuit_breakers: dict[str, CircuitBreaker], retry_budget: RetryBudget):
        self._breakers = circuit_breakers
        self._retry_budget = retry_budget

    async def execute(self, providers: list, action):
        tried = []
        last_error = None

        for provider in providers:
            provider_id = provider.provider_id
            breaker = self._breakers.get(provider_id)

            if breaker and not await breaker.can_execute():
                logger.debug("Skipping %s — circuit breaker OPEN", provider_id)
                continue

            tried.append(provider)

            can_retry = await self._retry_budget.acquire()
            if not can_retry:
                logger.warning("Retry budget exhausted — cannot try %s", provider_id)
                break

            try:
                result = await action(provider)
                if breaker:
                    await breaker.record_success()
                await self._retry_budget.release()
                return result

            except Exception as e:
                last_error = e
                if breaker:
                    await breaker.record_failure()
                await self._retry_budget.release()

                if len(tried) < len(providers):
                    next_provider = providers[len(tried)] if len(tried) < len(providers) else None
                    if next_provider:
                        failover_total.labels(
                            from_provider=provider_id,
                            to_provider=next_provider.provider_id,
                        ).inc()
                    logger.warning(
                        "Provider %s failed: %s. Failing over to next.",
                        provider_id, e,
                    )
                continue

        if last_error:
            raise RuntimeError(
                f"All providers exhausted. Tried: {[p.provider_id for p in tried]}. Last error: {last_error}"
            )
        raise RuntimeError("No providers available (all circuit breakers open)")
