# Week 7: 6 Routing Strategies and the Complexity Classifier

## LinkedIn Post (Week 7 Progress)

---

Week 7 of the 75-day journey. Building InferRoute's routing engine.

**What I built:**
- 6 routing strategies: IntelligenceAware, RoundRobin, LeastConnections, Weighted, LatencyAware, CostAware
- Complexity classifier (heuristic + LLM fallback)
- A/B router, shadow traffic, drift detector
- RAG service + MCP tool registry
- Embedding cache

**What I learned:**

1. Routing is not "pick a provider." It's "pick the right provider for THIS request, given latency, load, cost, and complexity." Each strategy optimizes for a different dimension.

2. The complexity classifier was my first attempt at "smart" routing. Heuristic keywords (0ms) handle ~80% of prompts. LLM fallback (200ms, cached) handles the rest. But I later realized this doesn't scale — more on that in Week 10.

3. A/B routing and shadow traffic are how you safely test new providers. Send 10% of traffic to a new provider (shadow) without affecting responses. Compare quality. If it's better, gradually shift traffic (A/B).

**Next week:** Distributed tracing, Anthropic adapter, cache implementations.

#BuildingInPublic #AIInfrastructure #LLMGateway #Routing

---

## Deep Dive: 6 Routing Strategies — When to Use Each

*Posted on dev.to / Medium*

---

Routing is the core of an LLM gateway. Here are 6 strategies, when to use each, and their tradeoffs.

### 1. RoundRobin
```
Request 1 → OpenAI
Request 2 → Anthropic
Request 3 → vLLM
Request 4 → OpenAI (cycle back)
```
- **When:** All providers are equal, you want simple load distribution.
- **Tradeoff:** Doesn't account for latency, cost, or load. One provider might be slow and get the same traffic.

### 2. Weighted
```
OpenAI (weight=50) → gets 50% of traffic
Anthropic (weight=30) → gets 30%
vLLM (weight=20) → gets 20%
```
- **When:** Providers have different capacities or costs.
- **Tradeoff:** Static weights. Doesn't adapt to real-time conditions.

### 3. LeastConnections
```
OpenAI (in_flight=15) → skip
Anthropic (in_flight=3) → pick this
vLLM (in_flight=8) → skip
```
- **When:** Requests have variable duration (streaming vs non-streaming).
- **Tradeoff:** Doesn't account for latency or cost. A provider with 3 connections might still be slow.

### 4. LatencyAware (EWMA)
```
OpenAI (EWMA=800ms) → skip
Anthropic (EWMA=400ms) → pick this
vLLM (EWMA=1200ms) → skip
```
- **When:** Latency is the priority (real-time chat, streaming).
- **Tradeoff:** EWMA (α=0.3) smooths out spikes but reacts slowly to sudden changes. A provider that was fast 5 minutes ago might be slow now.

### 5. CostAware
```
OpenAI ($0.03/1K tokens) → skip
Anthropic ($0.015/1K tokens) → skip
vLLM ($0.001/1K tokens) → pick this
```
- **When:** Cost is the priority (batch processing, internal tools).
- **Tradeoff:** Cheapest provider might be slowest or lowest quality.

### 6. IntelligenceAware (my custom strategy)
```
Prompt: "Write a haiku" → complexity=low → tier=cheap → vLLM
Prompt: "Analyze this legal contract" → complexity=high → tier=premium → OpenAI
```
- **When:** You want to match model capability to prompt complexity.
- **Tradeoff:** Classification adds latency (0ms for heuristic, 200ms for LLM). And it doesn't scale — I'll explain why in Week 10.

### The real question: which one should you use?

None of them alone. The right approach is **config-based routing** (which I built in Week 10): the developer specifies requirements per workflow node, and the gateway picks the best provider dynamically. But that's a story for another post.

---

*This is part of a 75-day journey building AI infrastructure from scratch. Follow along for weekly updates and technical deep dives.*
