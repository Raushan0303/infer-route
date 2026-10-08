"""Measure cross-tenant rate-limit spillover through the real middleware stack.

Tenant A (noisy) sends at 5x the per-tenant refill rate; tenant B (quiet)
sends well inside its limit at the same time. "Spillover" = share of
tenant B's requests rejected with 429. With per-tenant buckets it should
be 0%; with a shared bucket B is throttled by A's traffic.

Usage:
    python -m benchmarks.tenant_spillover --label after
Writes benchmarks/results/tenant_spillover_<label>.json
"""
import argparse
import asyncio
import json
import os
import pathlib
import time
import uuid

import httpx
import redis.asyncio as aioredis

from app.main import app
from app.ratelimit.token_bucket import TokenBucketRateLimiter

REDIS_URL = os.environ.get("INFERROUTE_TEST_REDIS_URL", "redis://localhost:6390/0")


async def tenant_loop(client, key, rps, duration, codes):
    interval = 1.0 / rps
    end = time.monotonic() + duration
    while time.monotonic() < end:
        t0 = time.monotonic()
        r = await client.post("/v1/spillover-probe", headers={"Authorization": f"Bearer {key}"})
        codes.append(r.status_code)
        await asyncio.sleep(max(0.0, interval - (time.monotonic() - t0)))


async def main(label, capacity, refill, duration, noisy_x, quiet_rps):
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    limiter = TokenBucketRateLimiter(redis)
    limiter._capacity, limiter._refill_rate = capacity, refill
    app.state.rate_limiter = limiter

    run = uuid.uuid4().hex[:8]
    a_codes, b_codes = [], []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bench") as c:
        # Noisy tenant uses 10 concurrent senders to reach noisy_x * refill rps.
        noisy_rps_each = noisy_x * refill / 10
        await asyncio.gather(
            *[tenant_loop(c, f"noisy-{run}", noisy_rps_each, duration, a_codes) for _ in range(10)],
            tenant_loop(c, f"quiet-{run}", quiet_rps, duration, b_codes),
        )
    await redis.aclose()

    result = {
        "label": label,
        "config": {"capacity": capacity, "refill_per_s": refill, "duration_s": duration,
                   "noisy_target_rps": noisy_x * refill, "quiet_rps": quiet_rps},
        "noisy": {"sent": len(a_codes), "rejected_429": a_codes.count(429)},
        "quiet": {"sent": len(b_codes), "rejected_429": b_codes.count(429)},
    }
    result["quiet_spillover_pct"] = round(100 * b_codes.count(429) / max(1, len(b_codes)), 1)
    out = pathlib.Path(__file__).parent / "results" / f"tenant_spillover_{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run")
    p.add_argument("--capacity", type=int, default=20)
    p.add_argument("--refill", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--noisy-x", type=float, default=5.0)
    p.add_argument("--quiet-rps", type=float, default=5.0)
    a = p.parse_args()
    asyncio.run(main(a.label, a.capacity, a.refill, a.duration, a.noisy_x, a.quiet_rps))
