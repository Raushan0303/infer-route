"""Measure cache-hit latency of CachePipeline.process against a real Redis.

  exact     : exact-match hit (one Redis GET)
  semantic@N: semantic hit with N entries in the tenant's index (the exact
              tier is forced cold so the timing is the semantic tier only)

The embedding service is a local stub (deterministic 1536-d vectors, no
network), so the numbers EXCLUDE the embedding API call that a real
semantic lookup makes first. Add that provider round trip on top.

Also reports hit rate and the index size before/after each run, which is
how the old per-entry XFetch eviction bug showed up.

Usage: python -m benchmarks.cache_latency --label after
Writes benchmarks/results/cache_latency_<label>.json
"""
import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import random
import statistics
import time
import uuid

import redis.asyncio as aioredis

from app.cache.coalescer import RequestCoalescer
from app.cache.exact_cache import ExactMatchCache
from app.cache.pipeline import CachePipeline
from app.cache.semantic_cache import SemanticCache
from app.gateway.models import InferRouteMessage, InferRouteRequest, InferRouteResponse, InferRouteUsage

REDIS_URL = os.environ.get("INFERROUTE_TEST_REDIS_URL", "redis://localhost:6390/0")
DIM = 1536


class StubEmbedding:
    """Deterministic unit-ish vectors from the text hash; no network."""

    async def embed(self, text):
        rnd = random.Random(hashlib.sha256(text.encode()).digest())
        return [rnd.uniform(-1, 1) for _ in range(DIM)]


class CountingUpstream:
    """Stands in for the provider: a cache miss lands here."""
    calls = 0

    async def route(self, request, extra_headers=None, preferred_provider=None):
        CountingUpstream.calls += 1
        return RESP


class ColdExact(ExactMatchCache):
    """Exact tier that always misses, so semantic timings are semantic only."""

    async def get(self, tenant_id, request):
        return None


def req(text):
    return InferRouteRequest(model="gpt-4o-mini", messages=[InferRouteMessage(role="user", content=text)])


RESP = InferRouteResponse(content="cached answer", model="gpt-4o-mini", provider="openai",
                          usage=InferRouteUsage(input_tokens=10, output_tokens=20))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


async def timed(fn, n, expect):
    out, hits = [], 0
    for _ in range(n):
        t0 = time.perf_counter()
        source = await fn()
        out.append((time.perf_counter() - t0) * 1000)
        hits += source == expect
    return {"n": n, "hit_rate": round(hits / n, 3),
            "p50_ms": round(statistics.median(out), 3), "p99_ms": round(pct(out, 0.99), 3)}


async def main(label):
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    exact, semantic = ExactMatchCache(redis), SemanticCache(redis, StubEmbedding())
    pipeline = CachePipeline(exact, semantic, RequestCoalescer(), CountingUpstream())
    results = {"label": label, "embedding": "local stub — excludes the embedding API round trip"}

    tenant = f"bench-{uuid.uuid4().hex[:8]}"
    await exact.set(tenant, req("exact prompt"), RESP)

    async def exact_hit():
        _, source = await pipeline.process(tenant, req("exact prompt"))
        return source
    results["exact"] = await timed(exact_hit, 500, "exact")

    semantic_pipeline = CachePipeline(ColdExact(redis), semantic, RequestCoalescer(), CountingUpstream())
    for n in (10, 100, 1000):
        tenant = f"bench-{uuid.uuid4().hex[:8]}"
        for i in range(n):
            await semantic.set(tenant, req(f"entry {i}"), RESP)
        before = await redis.scard(semantic._index_key(tenant))
        trials = 30 if n == 1000 else 100

        async def semantic_hit():
            # identical text → similarity 1.0 → should be a semantic hit
            _, source = await semantic_pipeline.process(tenant, req(f"entry {n // 2}"))
            return source
        results[f"semantic@{n}"] = await timed(semantic_hit, trials, "semantic")
        results[f"semantic@{n}"]["entries_before"] = before
        results[f"semantic@{n}"]["entries_after"] = await redis.scard(semantic._index_key(tenant))

    await redis.aclose()
    out = pathlib.Path(__file__).parent / "results" / f"cache_latency_{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run")
    asyncio.run(main(p.parse_args().label))
