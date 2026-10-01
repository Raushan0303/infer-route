# Week 9-10: Config-Based Routing, Dynamic Scoring, and the Portkey Lesson

## LinkedIn Post (Week 10 Progress)

---

Week 10 — the final week of InferRoute. This week I changed the routing architecture fundamentally.

**What I built:**
- OpenAI + vLLM adapters (3 providers now: OpenAI, Anthropic, vLLM)
- Provider registry with EWMA latency tracking (α=0.3)
- Postgres schema (routing_configs, usage_log, rag_documents, feedback)
- Config registry (Redis-cached per-tenant routing configs, 1ms lookup)
- Dynamic router (real-time provider scoring: 35% latency + 25% load + 20% cost + 20% weight)
- Routing service with 3 modes (config-based, heuristic, direct)
- REST API with config CRUD endpoints

**The big lesson:**

I started with an LLM-based complexity classifier. It was "smart" — it called a cheap LLM to classify prompt complexity, then routed to the right tier. It worked great in testing.

Then I researched how Portkey (10B tokens/day) does it. They don't classify at all. The developer creates a config per workflow node: "this node needs premium tier, this node needs cheap tier." The gateway looks up the config (1ms) and routes.

**Why this is better:**
1. No classification latency (0ms vs 200ms)
2. No classification cost (free vs $0.0001/request)
3. Deterministic (same config → same route)
4. Scales (config lookup is O(1), classification is O(prompt_length))

**What I'd do differently:** Start with config-based routing. The LLM classifier was over-engineering. But building it taught me why it doesn't scale — and that lesson is worth more than getting it right the first time.

**Next:** UIAgent — a Next.js dashboard to visualize everything.

#BuildingInPublic #AIInfrastructure #LLMGateway #Routing

---

## Deep Dive: Why I Abandoned LLM-Based Prompt Classification for Config-Based Routing

*Posted on dev.to / Medium*

---

This is the most important architectural decision I made in 75 days. And I got it wrong the first time.

### The "smart" approach: LLM-based classification

My first routing strategy was IntelligenceAware:

```
Prompt → LLM classifier (200ms) → complexity=high → tier=premium → OpenAI
```

The classifier was a cheap LLM call (gpt-4o-mini) that returned "low", "medium", or "high" complexity. Results were cached in Redis. It worked great in testing.

### Why it doesn't scale

At 100K unique prompts:
- 80% are handled by heuristic keywords (0ms) — fine
- 20% need the LLM classifier (200ms) — 20K calls
- Cache hit rate for unique prompts: ~30% (most prompts are unique)
- 14K classifier calls per 100K requests = 14K × 200ms = 2800 seconds of classifier latency
- Cost: 14K × $0.0001 = $1.40 per 100K requests just for classification

At 1M requests/day: $14/day for classification. At 10M: $140/day. Just to decide which provider to use.

### How Portkey does it (10B tokens/day)

Portkey doesn't classify. The developer creates a config:

```json
{
  "config_id": "sourcing-agent-decide",
  "targets": [
    {"provider": "openai", "model": "gpt-4o", "weight": 60},
    {"provider": "anthropic", "model": "claude-3.5-sonnet", "weight": 40}
  ],
  "fallback_targets": [
    {"provider": "vllm", "model": "llama-3.3-70b"}
  ],
  "latency_budget_ms": 2000,
  "cost_ceiling": 0.05
}
```

The gateway looks up the config (Redis, 1ms) and routes. No classification. No LLM call. No 200ms latency. No $140/day cost.

### What I built instead: Config + Dynamic Scoring

I took Portkey's config model and added dynamic scoring:

```
Developer creates config (per workflow node)
  → Gateway looks up config (1ms, Redis)
  → Dynamic router scores providers in real-time:
      35% latency (EWMA)
      25% load (in-flight requests)
      20% cost (per-token price)
      20% weight (developer preference)
  → Returns ordered list (best + fallbacks)
  → Circuit breaker + failover chain
```

This combines Portkey's simplicity (config-based, no classification) with real-time awareness (dynamic scoring based on actual provider conditions).

### The tradeoff

**Config-based:** Developer must know which tier each node needs. More work for the developer, less work for the gateway.

**LLM classification:** Gateway figures it out. Less work for the developer, more latency and cost for the gateway.

For production at scale, config-based wins. The developer knows their workflow better than a classifier does. And 1ms config lookup beats 200ms classification every time.

### What I learned

1. "Smart" is not always better. Deterministic + fast beats intelligent + slow.
2. Research how production systems work before building your own. I could have saved a week if I'd studied Portkey first.
3. The best routing is no routing — let the developer decide, and make the lookup as fast as possible.

---

*This is part of a 75-day journey building AI infrastructure from scratch. InferRoute is complete. Next: UIAgent — the dashboard.*
