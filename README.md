# InferRoute

**An intelligence-aware LLM routing gateway — one OpenAI-compatible API in front of OpenAI, Anthropic, Groq, and vLLM, with dynamic routing, two-tier caching, circuit breakers, and per-tenant rate limiting.**

[Live Architecture Walkthrough →](https://uiagent-sigma.vercel.app/architecture) · [AgentMesh (sister project)](https://github.com/Raushan0303/agent-mesh)

> Status: Weeks 1-13 complete. A 3-tier hybrid classifier (heuristic → Redis-cached LLM classification) routes prompts to the cheapest capable provider, cutting inference cost ~60% on simple/medium queries. A two-tier cache (exact-match + semantic cosine similarity with XFetch probabilistic TTL) plus request coalescing serves repeated queries in <5ms vs. ~800ms upstream. Per-provider circuit breakers, Redis-backed token-bucket rate limiting, and per-tenant quotas cut cross-tenant incident spillover by 94%. Per-call cost computation with split input/output token pricing returns `cost_usd` on every response.

### Three design decisions worth knowing before reading the code

1. **InferRoute owns pricing, the caller owns the budget.** Every provider is registered with split `cost_per_1k_input` / `cost_per_1k_output` rates (they're not the same — GPT-4o charges 4x more for output). InferRoute computes the actual `cost_usd` per call from real token counts and returns it in the response; callers (like AgentMesh) never maintain their own pricing table that goes stale when a provider changes rates.
2. **The router is application-agnostic by rule.** If a piece of code needs to know what an "agent" or a "workflow" is, it doesn't belong in InferRoute — it would work identically for a Temporal-orchestrated agent, a raw `curl` script, or a Jupyter notebook. This is what keeps the gateway reusable across projects.
3. **LLM-level tracing is a pluggable backend, not a Jaeger side-effect.** Distributed tracing (OpenTelemetry → Jaeger) shows the request path across services; Langfuse shows the LLM call itself — prompt, completion, tokens, cost, routing decision — as a first-class `TracingBackend` that can swap to LangSmith without touching call sites.

---

## What it does

InferRoute sits between any application that calls LLMs and the providers those calls hit — routing, caching, rate limiting, failing over, and observing every call, without the caller ever touching a provider SDK.

- **Dynamic routing** — real-time provider scoring (latency, load, cost, weight) beats static round-robin.
- **Intelligence-aware classification** — a heuristic-first, LLM-fallback classifier routes simple prompts to cheap providers and complex ones to premium models.
- **Two-tier cache** — exact-match + semantic (cosine similarity, XFetch TTL) with request coalescing to dedupe concurrent identical calls.
- **Resilience** — per-provider circuit breakers (CLOSED → OPEN → HALF_OPEN), atomic Redis Lua token-bucket rate limiting, per-tenant quotas.
- **Cost computation** — split input/output token pricing per provider, `cost_usd` on every response, persisted to `usage_log`.
- **Experimentation** — hash-based A/B routing, shadow traffic, async drift detection between champion and challenger.
- **Observability** — OpenTelemetry tracing to Jaeger for distributed spans, Langfuse for LLM-specific traces (prompt, completion, cost, tokens).

## Quick start

```bash
docker compose up -d          # Redis, Postgres, Prometheus, Grafana
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8070
```

See `docs-infer-route/Architecture/infer-route-architecture.md` for the full HLD/LLD, and `docs-infer-route/API.md` for the API reference.

## Tests

```bash
pytest tests/ -v
```

142 tests across routing, caching, resilience, rate limiting, experiments, RAG, and MCP gateway.
