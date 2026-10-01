# Week 6-7: Starting InferRoute — The LLM Gateway

## LinkedIn Post (Week 7 Progress)

---

After 5 weeks of AgentMesh, I started building InferRoute — the LLM gateway that AgentMesh calls instead of hitting OpenAI directly.

**Why a gateway?**

Without a gateway, every service calls OpenAI directly. When OpenAI goes down, every service fails. When you want to switch to Anthropic, you change every service. When you want to cache, you build caching in every service.

A gateway centralizes: routing, caching, failover, rate limiting, cost tracking.

**What I built:**
- FastAPI gateway with OpenAI-compatible API
- Base adapter pattern (providers plug in)
- Health checker (periodic provider probes)
- Request coalescer (single-flight for cache misses)
- Rate limiting (token bucket, Redis Lua script, per-tenant)
- Circuit breaker + failover chain
- Quota enforcement (per-tenant daily/monthly token limits)

**What I learned:**

1. Request coalescing is critical. If 10 requests ask the same question at the same time, and it's a cache miss, you don't want 10 upstream calls. The coalescer makes 1 call and 9 others wait on the same Future. This is the single-flight pattern.

2. Token bucket rate limiting in Redis must be atomic. If you read the count, check the limit, and decrement in separate Redis calls, you have a race condition. The solution: a Lua script that runs atomically in Redis.

3. Failover is not just "try the next provider." It's: circuit breaker checks → health check → timeout → retry with backoff → give up and return error. Each step has its own tradeoffs.

**Next week:** RAG, MCP, routing strategies, and the complexity classifier.

#BuildingInPublic #AIInfrastructure #LLMGateway #DistributedSystems

---

## Deep Dive: Why I Built an LLM Gateway Instead of Calling OpenAI Directly

*Posted on dev.to / Medium*

---

When you build your first AI app, you call `openai.chat.completions.create()`. It works. It's simple. Then you hit production and realize:

1. **OpenAI goes down.** 99.9% SLA = 43 minutes/month downtime. Without a gateway, every service fails simultaneously.
2. **You want to use Anthropic too.** Now you need to change every service that calls OpenAI.
3. **You're spending too much.** No caching = every request hits the API. No routing = every request uses GPT-4o when GPT-4o-mini would work.
4. **You need rate limiting.** One tenant shouldn't be able to use all your API quota.
5. **You need observability.** Which provider is slowest? What's the cache hit rate? What's the cost per tenant?

### What a gateway does

```
Service A ─┐
Service B ─┼──→ InferRoute Gateway ──→ OpenAI
Service C ─┘         │              ──→ Anthropic
                      │              ──→ vLLM (self-hosted)
                      │
                      ├── Cache (Redis)
                      ├── Circuit Breaker
                      ├── Rate Limiter
                      ├── Cost Tracker
                      └── Tracing
```

Every service calls the gateway with the same OpenAI-compatible API. The gateway handles:
- **Routing:** Which provider? Which model? Based on complexity, cost, latency.
- **Caching:** Exact match (SHA256) + semantic (embedding similarity).
- **Failover:** Provider down → try next. Circuit breaker prevents cascading failures.
- **Rate limiting:** Per-tenant token bucket. One tenant can't exhaust the quota.
- **Cost tracking:** Per-tenant, per-provider, per-model usage logs.
- **Tracing:** W3C traceparent propagates across services.

### The tradeoff

A gateway adds 1-3ms of overhead (Python) or <0.5ms (Go). For LLM calls that take 200-2000ms, this is noise. But it's a new component to operate, monitor, and keep available.

For a single service calling one provider, a gateway is overkill. For multiple services, multiple providers, and production reliability requirements, a gateway is essential.

### Industry comparison

| Gateway | Language | Overhead | Key feature |
|---|---|---|---|
| Portkey | Node.js | <1ms | Config-based routing, 10B tokens/day |
| LiteLLM | Python | ~3ms | 100+ provider support |
| Bifrost | Go | 11µs | 50x faster than LiteLLM |
| Kong AI Gateway | Lua/Nginx | <1ms | Plugin ecosystem |
| InferRoute (mine) | Python | ~3ms | Dynamic provider scoring + config-based routing |

I chose Python for iteration speed. If I hit 5000+ RPS, I'd rewrite the hot path in Go (based on Bifrost benchmarks, Go gives 9x throughput and 15x better p99).

---

*This is part of a 75-day journey building AI infrastructure from scratch. AgentMesh (6 weeks) is complete. Now building InferRoute.*
