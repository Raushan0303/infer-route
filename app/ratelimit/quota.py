import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from app.core.config import settings
from app.core.constants import QUOTA_DAILY_KEY_PREFIX, QUOTA_MONTHLY_KEY_PREFIX

logger = logging.getLogger("inferroute")


@dataclass
class QuotaResult:
    allowed: bool
    daily_remaining: int
    monthly_remaining: int
    reset_at: int


class QuotaEnforcer:

    def __init__(self, redis):
        self._redis = redis
        self._daily_limit = settings.quota_daily_token_limit
        self._monthly_limit = settings.quota_monthly_token_limit

    def _daily_key(self, tenant_id: str) -> str:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{QUOTA_DAILY_KEY_PREFIX}:{tenant_id}:{today}"

    def _monthly_key(self, tenant_id: str) -> str:
        month = datetime.now(timezone.utc).strftime("%Y%m")
        return f"{QUOTA_MONTHLY_KEY_PREFIX}:{tenant_id}:{month}"

    def _seconds_until_midnight_utc(self) -> int:
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if tomorrow <= now:
            from datetime import timedelta
            tomorrow = tomorrow + timedelta(days=1)
        return int((tomorrow - now).total_seconds())

    async def check_and_consume(self, tenant_id: str, tokens: int) -> QuotaResult:
        try:
            daily_key = self._daily_key(tenant_id)
            monthly_key = self._monthly_key(tenant_id)

            pipe = self._redis.pipeline()
            await pipe.incrby(daily_key, tokens)
            await pipe.incrby(monthly_key, tokens)
            results = await pipe.execute()

            daily_used = int(results[0])
            monthly_used = int(results[1])

            daily_remaining = max(0, self._daily_limit - daily_used)
            monthly_remaining = max(0, self._monthly_limit - monthly_used)

            if daily_used > self._daily_limit or monthly_used > self._monthly_limit:
                pipe2 = self._redis.pipeline()
                await pipe2.incrby(daily_key, -tokens)
                await pipe2.incrby(monthly_key, -tokens)
                await pipe2.execute()

                return QuotaResult(
                    allowed=False,
                    daily_remaining=0,
                    monthly_remaining=max(0, self._monthly_limit - monthly_used + tokens),
                    reset_at=int(time.time()) + self._seconds_until_midnight_utc(),
                )

            return QuotaResult(
                allowed=True,
                daily_remaining=daily_remaining,
                monthly_remaining=monthly_remaining,
                reset_at=int(time.time()) + self._seconds_until_midnight_utc(),
            )
        except Exception as e:
            logger.warning("Quota enforcer Redis error (fail-open): %s", e)
            return QuotaResult(allowed=True, daily_remaining=0, monthly_remaining=0, reset_at=0)
