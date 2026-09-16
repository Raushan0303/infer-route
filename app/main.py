from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.gateway.routes import router as gateway_router
from app.gateway.middleware import auth_middleware, metrics_middleware, rate_limit_middleware, tracing_middleware
from app.core.config import settings
from app.adapters.openai import OpenAIAdapter
from app.adapters.anthropic import AnthropicAdapter
from app.adapters.vllm import VLLMAdapter
from app.routing.provider_registry import ProviderRegistry
from app.routing.health_checker import HealthChecker
from app.routing.routing_service import RoutingService
from app.routing.complexity_classifier import ComplexityClassifier
from app.cache.exact_cache import ExactMatchCache
from app.cache.semantic_cache import SemanticCache
from app.cache.coalescer import RequestCoalescer
from app.cache.embedding import EmbeddingService
from app.cache.pipeline import CachePipeline
from app.ratelimit.token_bucket import TokenBucketRateLimiter
from app.ratelimit.quota import QuotaEnforcer
from app.resilience.circuit_breaker import CircuitBreaker
from app.observability.metrics import circuit_breaker_state
from app.experiments.ab_router import ABRouter
from app.experiments.shadow_traffic import ShadowTrafficManager
from app.experiments.drift_detector import DriftDetector
from app.evals.feedback import FeedbackStore
from app.rag.embeddings import RAGEmbeddingService
from app.rag.store import VectorStore
from app.rag.retriever import HybridRetriever
from app.rag.rag_service import RAGService
from app.mcp.registry import MCPRegistry
from app.mcp.gateway import MCPGateway
from app.observability.tracing import TracingContext
from app.core.database import db

from prometheus_client import generate_latest
from fastapi import Response

logger = logging.getLogger("inferroute")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build classifier adapter (use Groq if available — fast + cheap)
    classifier_adapter = None
    if settings.groq_api_key:
        classifier_adapter = OpenAIAdapter(
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key,
            provider_id="groq",
            model_mapping={
                "classifier": "llama-3.3-70b-versatile",
            },
        )

    # Redis client for classifier cache
    import redis.asyncio as aioredis
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)

    classifier = ComplexityClassifier(
        classifier_adapter=classifier_adapter,
        redis_client=redis_client,
    )

    from app.routing.strategies import IntelligenceAware
    strategy = IntelligenceAware(classifier=classifier)

    registry = ProviderRegistry(strategy=strategy)

    if settings.openai_api_key:
        registry.register(
            provider_id="openai",
            adapter=OpenAIAdapter(
                base_url="https://api.openai.com/v1",
                api_key=settings.openai_api_key,
                provider_id="openai",
                model_mapping={
                    "auto": "gpt-4o-mini",
                    "default": "gpt-4o-mini",
                    "email-classifier:latest": "gpt-4o-mini",
                    "llama-3.3-70b-versatile": "gpt-4o-mini",
                },
            ),
            base_url="https://api.openai.com/v1",
            weight=1.0,
            cost_per_1k_tokens=0.150,
            tier="premium",
        )

    if settings.groq_api_key:
        registry.register(
            provider_id="groq",
            adapter=OpenAIAdapter(
                base_url=settings.groq_base_url,
                api_key=settings.groq_api_key,
                provider_id="groq",
                model_mapping={
                    "gpt-4o-mini": "llama-3.3-70b-versatile",
                    "gpt-4o": "llama-3.3-70b-versatile",
                    "gpt-4-turbo": "llama-3.3-70b-versatile",
                    "auto": "llama-3.3-70b-versatile",
                    "default": "llama-3.3-70b-versatile",
                },
            ),
            base_url=settings.groq_base_url,
            weight=1.0,
            cost_per_1k_tokens=0.059,
            tier="cheap",
        )

    if settings.anthropic_api_key:
        registry.register(
            provider_id="anthropic",
            adapter=AnthropicAdapter(
                base_url="https://api.anthropic.com/v1",
                api_key=settings.anthropic_api_key,
            ),
            base_url="https://api.anthropic.com/v1",
            weight=1.0,
            cost_per_1k_tokens=0.800,
            tier="premium",
        )

    registry.register(
        provider_id="vllm",
        adapter=VLLMAdapter(
            base_url=settings.vllm_base_url,
            provider_id="vllm",
            model_mapping={
                "gpt-4o-mini": "email-classifier:latest",
                "gpt-4o": "email-classifier:latest",
                "gpt-4-turbo": "email-classifier:latest",
                "auto": "email-classifier:latest",
                "default": "email-classifier:latest",
            },
        ),
        base_url=settings.vllm_base_url,
        weight=1.0,
        cost_per_1k_tokens=0.0,
        tier="cheap",
    )

    health_checker = HealthChecker(registry)
    await health_checker.start()

    routing_service = RoutingService(registry)

    # Circuit breakers per provider
    for provider in registry.get_all():
        breaker = CircuitBreaker(provider_id=provider.provider_id)
        routing_service.register_circuit_breaker(provider.provider_id, breaker)
        circuit_breaker_state.labels(provider=provider.provider_id).set(0)

    app.state.routing_service = routing_service
    app.state.health_checker = health_checker
    app.state.registry = registry

    # Rate limiting + quota
    import redis.asyncio as aioredis
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)

    if settings.rate_limit_enabled:
        app.state.rate_limiter = TokenBucketRateLimiter(redis_client)
    else:
        app.state.rate_limiter = None

    if settings.quota_enabled:
        app.state.quota_enforcer = QuotaEnforcer(redis_client)
    else:
        app.state.quota_enforcer = None

    # Cache pipeline
    if settings.cache_enabled:
        embedding_service = EmbeddingService(
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
        )

        exact_cache = ExactMatchCache(redis_client)
        semantic_cache = SemanticCache(redis_client, embedding_service)
        coalescer = RequestCoalescer()

        app.state.cache_pipeline = CachePipeline(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            coalescer=coalescer,
            routing_service=routing_service,
        )
        app.state.redis = redis_client
        app.state.embedding_service = embedding_service
    else:
        app.state.cache_pipeline = None

    # Experiments + evals
    if settings.experiments_enabled:
        app.state.ab_router = ABRouter()
    else:
        app.state.ab_router = None

    if settings.shadow_traffic_enabled:
        app.state.shadow_traffic = ShadowTrafficManager()
    else:
        app.state.shadow_traffic = None

    # Postgres initialization
    try:
        await db.init()
        app.state.db = db
        logger.info("Postgres initialized")
    except Exception as e:
        logger.warning("Postgres init failed (running with in-memory fallback): %s", e)
        app.state.db = None

    # Config-based routing (Portkey-style per-tenant configs)
    from app.routing.config_registry import ConfigRegistry
    config_registry = ConfigRegistry(redis_client=redis_client if settings.cache_enabled else None)
    routing_service._config_registry = config_registry
    app.state.config_registry = config_registry
    logger.info("Config registry initialized (Redis-cached routing configs)")

    app.state.feedback_store = FeedbackStore()

    # Quota enforcement
    if settings.quota_enabled and redis_client:
        quota_enforcer = QuotaEnforcer(redis_client)
        app.state.quota_enforcer = quota_enforcer
        logger.info("Quota enforcement enabled: daily=%d monthly=%d", settings.quota_daily_token_limit, settings.quota_monthly_token_limit)
    else:
        app.state.quota_enforcer = None

    drift_detector = None
    if settings.drift_detection_enabled:
        drift_detector = DriftDetector(
            threshold=settings.drift_threshold,
            interval=settings.drift_detection_interval,
        )
        await drift_detector.start()
    app.state.drift_detector = drift_detector

    # RAG service
    rag_embedding = None
    rag_service = None
    if settings.rag_enabled:
        rag_embedding = RAGEmbeddingService(
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
        )
        vector_store = VectorStore()
        retriever = HybridRetriever(vector_store, rag_embedding)
        rag_service = RAGService(vector_store, retriever, redis_client)
        app.state.rag_embedding = rag_embedding
        app.state.rag_service = rag_service
    else:
        app.state.rag_embedding = None
        app.state.rag_service = None

    # MCP gateway
    if settings.mcp_enabled:
        mcp_registry = MCPRegistry()
        mcp_gateway = MCPGateway(mcp_registry)
        app.state.mcp_registry = mcp_registry
        app.state.mcp_gateway = mcp_gateway
    else:
        app.state.mcp_registry = None
        app.state.mcp_gateway = None

    # Tracing — pluggable backend (Langfuse default, LangSmith optional)
    if settings.tracing_enabled:
        tracing_ctx = TracingContext()
        app.state.tracing_ctx = tracing_ctx
        logger.info("Tracing enabled: service=%s", settings.tracing_service)
    else:
        app.state.tracing_ctx = None

    yield

    if drift_detector:
        await drift_detector.stop()
    if mcp_gateway:
        await mcp_gateway.close()
    if rag_embedding:
        await rag_embedding.close()
    await health_checker.stop()
    await app.state.routing_service.close()
    if hasattr(app.state, "tracing_ctx") and app.state.tracing_ctx:
        app.state.tracing_ctx.flush()
    if hasattr(app.state, "redis") and app.state.redis:
        await app.state.redis.aclose()
    if hasattr(app.state, "embedding_service") and app.state.embedding_service:
        await app.state.embedding_service.close()
    await db.close()


app = FastAPI(
    title="InferRoute",
    description="Production-grade LLM inference control plane",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(auth_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(tracing_middleware)
app.middleware("http")(metrics_middleware)
app.include_router(gateway_router)


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")
