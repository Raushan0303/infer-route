from pydantic_settings import BaseSettings
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    name: str
    adapter_class: str
    base_url: str
    api_key: str = ""
    weight: float = 1.0
    health_check_interval: int = 10
    health_check_endpoint: str = "/v1/models"
    max_connections: int = 100
    connect_timeout: int = 10
    read_timeout: int = 120


class Settings(BaseSettings):
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8070

    redis_url: str = "redis://localhost:6379/0"

    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/inferroute"

    default_strategy: str = "intelligence_aware"

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    vllm_base_url: str = "http://localhost:11434/v1"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    ewma_alpha: float = 0.3

    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_ttl_jitter: float = 0.1
    cache_similarity_threshold: float = 0.95
    cache_embedding_model: str = "text-embedding-3-small"
    cache_embedding_dim: int = 1536

    rate_limit_enabled: bool = True
    rate_limit_capacity: int = 100
    rate_limit_refill_rate: float = 10.0
    rate_limit_cost_per_request: int = 1

    quota_enabled: bool = True
    quota_daily_token_limit: int = 1_000_000
    quota_monthly_token_limit: int = 10_000_000

    fair_queue_enabled: bool = True

    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_cooldown_seconds: int = 15
    circuit_breaker_half_open_max_calls: int = 1

    retry_budget_max_concurrent: int = 50

    experiments_enabled: bool = True
    ab_split_percentage: float = 0.5
    ab_variant_a_provider: str = "openai"
    ab_variant_b_provider: str = "anthropic"

    shadow_traffic_enabled: bool = True
    shadow_traffic_percentage: float = 0.1
    shadow_target_provider: str = "vllm"

    drift_detection_enabled: bool = True
    drift_detection_interval: int = 300
    drift_threshold: float = 0.1

    eval_sample_rate: float = 0.1

    rag_enabled: bool = True
    rag_embedding_model: str = "text-embedding-3-small"
    rag_embedding_dim: int = 1536
    rag_top_k: int = 10
    rag_rerank_top_k: int = 5
    rag_cache_ttl: int = 1800

    mcp_enabled: bool = True
    mcp_health_check_interval: int = 30
    mcp_circuit_breaker_threshold: int = 3
    mcp_circuit_breaker_cooldown: int = 10

    tracing_enabled: bool = True
    tracing_service: str = "langfuse"
    otlp_exporter_endpoint: str = "http://localhost:4318/v1/traces"

    # Langfuse settings
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # LangSmith settings (alternative backend)
    langsmith_api_key: str = ""
    langsmith_project: str = "infer-route"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    class Config:
        env_file = ".env"
        env_prefix = "INFERROUTE_"


settings = Settings()
