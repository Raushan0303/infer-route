STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_LEAST_CONNECTIONS = "least_connections"
STRATEGY_WEIGHTED = "weighted"
STRATEGY_LATENCY_AWARE = "latency_aware"

DEFAULT_HEALTH_CHECK_INTERVAL = 10
DEFAULT_HEALTH_CHECK_TIMEOUT = 5

STATUS_HEALTHY = "HEALTHY"
STATUS_UNHEALTHY = "UNHEALTHY"
STATUS_DEGRADED = "DEGRADED"

DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 120
DEFAULT_WRITE_TIMEOUT = 30
DEFAULT_MAX_CONNECTIONS = 100

METRIC_REQUEST_TOTAL = "inferroute_request_total"
METRIC_REQUEST_DURATION = "inferroute_request_duration_seconds"
METRIC_PROVIDER_IN_FLIGHT = "inferroute_provider_in_flight"
METRIC_CIRCUIT_BREAKER_STATE = "inferroute_circuit_breaker_state"

CACHE_KEY_PREFIX_EXACT = "cache:exact"
CACHE_KEY_PREFIX_SEMANTIC = "cache:semantic"
CACHE_KEY_PREFIX_INFLIGHT = "cache:inflight"
CACHE_DEFAULT_TTL = 3600
CACHE_TTL_JITTER = 0.1
CACHE_SIMILARITY_THRESHOLD = 0.95
CACHE_EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_EMBEDDING_DIM = 1536

METRIC_CACHE_HIT_TOTAL = "inferroute_cache_hit_total"
METRIC_CACHE_MISS_TOTAL = "inferroute_cache_miss_total"
METRIC_COALESCED_REQUESTS_TOTAL = "inferroute_coalesced_requests_total"

RATE_LIMIT_KEY_PREFIX = "ratelimit:bucket"
QUOTA_DAILY_KEY_PREFIX = "quota:daily"
QUOTA_MONTHLY_KEY_PREFIX = "quota:monthly"
FAIR_QUEUE_KEY_PREFIX = "fairqueue"

CIRCUIT_CLOSED = "CLOSED"
CIRCUIT_OPEN = "OPEN"
CIRCUIT_HALF_OPEN = "HALF_OPEN"

CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_COOLDOWN_SECONDS = 15
CIRCUIT_HALF_OPEN_MAX_CALLS = 1

RETRY_BUDGET_MAX_CONCURRENT = 50

METRIC_RATE_LIMIT_REJECTED = "inferroute_rate_limit_rejected_total"
METRIC_CIRCUIT_BREAKER_STATE = "inferroute_circuit_breaker_state"
METRIC_FAILOVER_TOTAL = "inferroute_failover_total"
METRIC_QUOTA_EXCEEDED = "inferroute_quota_exceeded_total"

VARIANT_A = "A"
VARIANT_B = "B"

EXPERIMENT_KEY_PREFIX = "experiment"
FEEDBACK_KEY_PREFIX = "feedback"

DRIFT_DEFAULT_INTERVAL = 300
DRIFT_DEFAULT_THRESHOLD = 0.1

METRIC_AB_VARIANT_TOTAL = "inferroute_ab_variant_total"
METRIC_SHADOW_TRAFFIC_TOTAL = "inferroute_shadow_traffic_total"
METRIC_DRIFT_ALERT_TOTAL = "inferroute_drift_alert_total"
METRIC_FEEDBACK_TOTAL = "inferroute_feedback_total"

RAG_KEY_PREFIX = "rag:cache"
RAG_DEFAULT_TOP_K = 10
RAG_RERANK_TOP_K = 5
RAG_CACHE_TTL = 1800

MCP_KEY_PREFIX = "mcp"
MCP_HEALTH_CHECK_INTERVAL = 30

METRIC_RAG_QUERY_TOTAL = "inferroute_rag_query_total"
METRIC_RAG_QUERY_DURATION = "inferroute_rag_query_duration"
METRIC_RAG_AUGMENTED_TOTAL = "inferroute_rag_augmented_total"
METRIC_RAG_CONTEXT_TOKENS = "inferroute_rag_context_tokens"
METRIC_MCP_TOOL_INVOKE_TOTAL = "inferroute_mcp_tool_invoke_total"
METRIC_MCP_REPLICA_HEALTH = "inferroute_mcp_replica_health"
