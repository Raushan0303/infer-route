import time
import logging
from dataclasses import dataclass
from app.core.config import settings
from app.core.constants import RATE_LIMIT_KEY_PREFIX

logger = logging.getLogger("inferroute")

LUA_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local tokens_key = key .. ":tokens"
local last_key = key .. ":last"

local tokens = tonumber(redis.call("GET", tokens_key)) or capacity
local last = tonumber(redis.call("GET", last_key)) or now

local elapsed = math.max(0, now - last)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

if tokens >= cost then
    tokens = tokens - cost
    redis.call("SET", tokens_key, tokens)
    redis.call("SET", last_key, now)
    return {1, math.floor(tokens), 0}
else
    local retry_after = math.ceil((cost - tokens) / refill_rate)
    redis.call("SET", tokens_key, tokens)
    redis.call("SET", last_key, now)
    return {0, math.floor(tokens), retry_after}
end
"""


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: int


class TokenBucketRateLimiter:

    def __init__(self, redis):
        self._redis = redis
        self._capacity = settings.rate_limit_capacity
        self._refill_rate = settings.rate_limit_refill_rate
        self._cost = settings.rate_limit_cost_per_request
        self._script_sha = None

    async def _ensure_script(self):
        if self._script_sha is None:
            self._script_sha = await self._redis.script_load(LUA_SCRIPT)

    async def check(self, tenant_id: str, cost: int = None) -> RateLimitResult:
        try:
            await self._ensure_script()
            key = f"{RATE_LIMIT_KEY_PREFIX}:{tenant_id}"
            now = time.time()
            result = await self._redis.evalsha(
                self._script_sha,
                1,
                key,
                self._capacity,
                self._refill_rate,
                cost or self._cost,
                now,
            )
            allowed = bool(result[0])
            remaining = int(result[1])
            retry_after = int(result[2])
            return RateLimitResult(allowed=allowed, remaining=remaining, retry_after=retry_after)
        except Exception as e:
            logger.warning("Rate limiter Redis error (fail-open): %s", e)
            return RateLimitResult(allowed=True, remaining=0, retry_after=0)
