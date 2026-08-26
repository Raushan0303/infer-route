from prometheus_client import Counter, Histogram, Gauge

from app.core.constants import (
    METRIC_REQUEST_TOTAL,
    METRIC_REQUEST_DURATION,
    METRIC_PROVIDER_IN_FLIGHT,
    METRIC_CACHE_HIT_TOTAL,
    METRIC_CACHE_MISS_TOTAL,
    METRIC_COALESCED_REQUESTS_TOTAL,
    METRIC_RATE_LIMIT_REJECTED,
    METRIC_CIRCUIT_BREAKER_STATE,
    METRIC_FAILOVER_TOTAL,
    METRIC_QUOTA_EXCEEDED,
    METRIC_AB_VARIANT_TOTAL,
    METRIC_SHADOW_TRAFFIC_TOTAL,
    METRIC_DRIFT_ALERT_TOTAL,
    METRIC_FEEDBACK_TOTAL,
    METRIC_RAG_QUERY_TOTAL,
    METRIC_RAG_QUERY_DURATION,
    METRIC_RAG_AUGMENTED_TOTAL,
    METRIC_RAG_CONTEXT_TOKENS,
    METRIC_MCP_TOOL_INVOKE_TOTAL,
    METRIC_MCP_REPLICA_HEALTH,
)

request_total = Counter(
    METRIC_REQUEST_TOTAL,
    "Total requests processed",
    ["provider", "status", "tenant_id"],
)

request_duration = Histogram(
    METRIC_REQUEST_DURATION,
    "Request duration in seconds",
    ["provider"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

provider_in_flight = Gauge(
    METRIC_PROVIDER_IN_FLIGHT,
    "In-flight requests per provider",
    ["provider"],
)

cache_hit_total = Counter(
    METRIC_CACHE_HIT_TOTAL,
    "Cache hits by type",
    ["cache_type"],
)

cache_miss_total = Counter(
    METRIC_CACHE_MISS_TOTAL,
    "Cache misses",
)

coalesced_requests_total = Counter(
    METRIC_COALESCED_REQUESTS_TOTAL,
    "Requests coalesced into existing in-flight calls",
)

rate_limit_rejected_total = Counter(
    METRIC_RATE_LIMIT_REJECTED,
    "Rate-limited requests rejected",
    ["tenant_id"],
)

circuit_breaker_state = Gauge(
    METRIC_CIRCUIT_BREAKER_STATE,
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["provider"],
)

failover_total = Counter(
    METRIC_FAILOVER_TOTAL,
    "Failover events",
    ["from_provider", "to_provider"],
)

quota_exceeded_total = Counter(
    METRIC_QUOTA_EXCEEDED,
    "Quota-exceeded rejections",
    ["tenant_id"],
)

ab_variant_total = Counter(
    METRIC_AB_VARIANT_TOTAL,
    "A/B variant assignments",
    ["variant"],
)

shadow_traffic_total = Counter(
    METRIC_SHADOW_TRAFFIC_TOTAL,
    "Shadow traffic calls",
    ["target_provider"],
)

drift_alert_total = Counter(
    METRIC_DRIFT_ALERT_TOTAL,
    "Drift detection alerts fired",
)

feedback_total = Counter(
    METRIC_FEEDBACK_TOTAL,
    "Feedback submissions",
    ["thumbs"],
)

rag_query_total = Counter(
    METRIC_RAG_QUERY_TOTAL,
    "RAG queries",
    ["tenant_id", "namespace"],
)

rag_query_duration = Histogram(
    METRIC_RAG_QUERY_DURATION,
    "RAG query latency (seconds)",
)

rag_augmented_total = Counter(
    METRIC_RAG_AUGMENTED_TOTAL,
    "RAG-augmented chat completions",
    ["tenant_id", "namespace"],
)

rag_context_tokens = Histogram(
    METRIC_RAG_CONTEXT_TOKENS,
    "Context tokens injected by RAG",
    ["namespace"],
)

mcp_tool_invoke_total = Counter(
    METRIC_MCP_TOOL_INVOKE_TOTAL,
    "MCP tool invocations",
    ["tool_name", "status"],
)

mcp_replica_health = Gauge(
    METRIC_MCP_REPLICA_HEALTH,
    "MCP replica health (1=healthy, 0=unhealthy)",
    ["tool_name", "replica_id"],
)
