# Week 8: Distributed Tracing, Cache Pipeline, and Provider Adapters

## LinkedIn Post (Week 8 Progress)

---

Week 8 of the 75-day journey. InferRoute gets tracing, real caching, and provider adapters.

**What I built:**
- W3C distributed tracing (traceparent propagation across AgentMesh → InferRoute → LLM provider)
- OTLP export to Jaeger
- Anthropic adapter (Claude support for failover)
- Exact cache (SHA256, Redis SETEX, TTL with jitter)
- Semantic cache (embedding cosine similarity ≥ 0.92)
- Cache pipeline (exact → semantic → coalesce → upstream)
- Prometheus metrics

**What I learned:**

1. Distributed tracing is not logging. Logging tells you what happened. Tracing tells you WHERE it happened and HOW LONG it took. A single trace spans AgentMesh (workflow start) → InferRoute (routing decision) → OpenAI (LLM call). Without tracing, you can't tell which layer is slow.

2. Cache TTL needs jitter. If 1000 cached responses all expire at the same time, you get a thundering herd — 1000 simultaneous upstream calls. Adding ±10% jitter to each TTL spreads the expiry over time.

3. The cache pipeline is the right abstraction. Each layer can short-circuit:
   - Exact match? → return (0ms, 100% savings)
   - Semantic match? → return (1ms, 95% savings)
   - Coalesce? → wait for in-flight request (0ms, 100% savings)
   - None? → call upstream (200-2000ms, 0% savings)

**Next week:** OpenAI + vLLM adapters, provider registry with EWMA latency tracking.

#BuildingInPublic #AIInfrastructure #DistributedTracing #Caching

---

## Deep Dive: W3C Traceparent — Tracing a Request Across Two Systems

*Posted on dev.to / Medium*

---

When AgentMesh starts a workflow that calls InferRoute, which calls OpenAI — how do you trace the full path? The answer is W3C Trace Context.

### The traceparent header

```
traceparent: 00-{trace-id}-{parent-span-id}-{trace-flags}
             00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
```

- **trace-id:** Unique per request chain. Same across all services.
- **parent-span-id:** Unique per service. Each service creates its own span.
- **trace-flags:** `01` = sampled (export to Jaeger), `00` = not sampled.

### How it flows

```
AgentMesh receives request
  → generates traceparent: 00-abc123-xyz789-01
  → starts span "workflow.execute"

AgentMesh calls InferRoute
  → sends header: traceparent: 00-abc123-xyz789-01
  → InferRoute receives, extracts trace-id and parent-span-id
  → starts span "gateway.route" (child of xyz789)

InferRoute calls OpenAI
  → sends header: traceparent: 00-abc123-def456-01
  → (OpenAI doesn't propagate, but we log the outbound call)

Result flows back, each span is exported to Jaeger.
```

### What you see in Jaeger

```
workflow.execute (AgentMesh)     1200ms
  ├── activity.research            300ms
  │   └── gateway.route (InferRoute)  280ms
  │       └── openai.chat            250ms
  ├── activity.score                400ms
  │   └── gateway.route (InferRoute)  380ms
  │       └── anthropic.complete     350ms
  └── activity.decide               500ms
      └── gateway.route (InferRoute)  480ms
          └── openai.chat            450ms
```

You can see exactly which activity is slow, which provider was used, and how long the LLM call took. Without this, you'd see "workflow took 1200ms" and have no idea where the time went.

### The implementation

AgentMesh:
```python
# Inject traceparent header when calling InferRoute
headers = {"traceparent": current_traceparent}
response = await llm_client.call(prompt, headers=headers)
```

InferRoute:
```python
# Extract traceparent from incoming request
traceparent = request.headers.get("traceparent")
# Start child span
with tracer.start_span("gateway.route", parent=traceparent):
    provider = route(prompt)
    response = await provider.complete(prompt)
```

### The tradeoff

Tracing adds ~0.5ms per span. For a workflow with 5 activities, that's 5 spans = 2.5ms. Negligible compared to LLM call latency (200-2000ms).

But tracing infrastructure (Jaeger, Tempo, Datadog) costs money and requires operational effort. For a learning project, Jaeger in Docker is fine. For production, you'd use a managed service.

---

*This is part of a 75-day journey building AI infrastructure from scratch. Follow along for weekly updates and technical deep dives.*
