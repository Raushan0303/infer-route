import pytest
from app.routing.strategies import (
    RoundRobin,
    LeastConnections,
    Weighted,
    LatencyAware,
    CostAware,
    IntelligenceAware,
    ProviderHealth,
)
from app.gateway.models import InferRouteRequest, InferRouteMessage


class TestRoundRobin:
    @pytest.mark.asyncio
    async def test_cycles_through_providers(self, healthy_providers):
        strategy = RoundRobin()
        selections = [(await strategy.select(healthy_providers)).provider_id for _ in range(6)]
        assert len(set(selections[:3])) == 3
        assert selections[0] == selections[3]

    @pytest.mark.asyncio
    async def test_returns_none_when_all_unhealthy(self, healthy_providers):
        for p in healthy_providers:
            p.status = "UNHEALTHY"
        strategy = RoundRobin()
        assert await strategy.select(healthy_providers) is None

    @pytest.mark.asyncio
    async def test_single_provider(self, healthy_providers):
        strategy = RoundRobin()
        single = healthy_providers[:1]
        assert (await strategy.select(single)).provider_id == "anthropic"


class TestLeastConnections:
    @pytest.mark.asyncio
    async def test_picks_fewest_in_flight(self, healthy_providers):
        strategy = LeastConnections()
        selected = await strategy.select(healthy_providers)
        assert selected.provider_id == "openai"

    @pytest.mark.asyncio
    async def test_returns_none_when_all_unhealthy(self, healthy_providers):
        for p in healthy_providers:
            p.status = "UNHEALTHY"
        strategy = LeastConnections()
        assert await strategy.select(healthy_providers) is None


class TestWeighted:
    @pytest.mark.asyncio
    async def test_distribution_matches_weights(self, healthy_providers):
        strategy = Weighted()
        counts = {"anthropic": 0, "openai": 0, "vllm": 0}
        for _ in range(10000):
            selected = await strategy.select(healthy_providers)
            counts[selected.provider_id] += 1
        total = sum(counts.values())
        assert 0.45 < counts["anthropic"] / total < 0.55
        assert 0.25 < counts["openai"] / total < 0.35
        assert 0.15 < counts["vllm"] / total < 0.25


class TestLatencyAware:
    @pytest.mark.asyncio
    async def test_picks_lowest_ewma(self, healthy_providers):
        strategy = LatencyAware()
        selected = await strategy.select(healthy_providers)
        assert selected.provider_id == "anthropic"

    @pytest.mark.asyncio
    async def test_cold_start_falls_back_to_random(self, healthy_providers):
        for p in healthy_providers:
            p.ewma_latency = 0.0
        strategy = LatencyAware()
        selected = await strategy.select(healthy_providers)
        assert selected is not None

    @pytest.mark.asyncio
    async def test_update_latency_changes_selection(self, healthy_providers):
        strategy = LatencyAware(alpha=0.5)
        assert (await strategy.select(healthy_providers)).provider_id == "anthropic"
        strategy.update_latency(healthy_providers[0], 3000)
        strategy.update_latency(healthy_providers[2], 200)
        assert (await strategy.select(healthy_providers)).provider_id == "vllm"


class TestCostAware:
    @pytest.mark.asyncio
    async def test_picks_cheapest(self):
        providers = [
            ProviderHealth(provider_id="openai", adapter=None, base_url="", cost_per_1m_input=0.150, cost_per_1m_output=0.600),
            ProviderHealth(provider_id="groq", adapter=None, base_url="", cost_per_1m_input=0.059, cost_per_1m_output=0.079),
            ProviderHealth(provider_id="anthropic", adapter=None, base_url="", cost_per_1m_input=3.000, cost_per_1m_output=15.000),
        ]
        strategy = CostAware()
        selected = await strategy.select(providers)
        assert selected.provider_id == "groq"

    @pytest.mark.asyncio
    async def test_returns_none_when_all_unhealthy(self):
        providers = [
            ProviderHealth(provider_id="openai", adapter=None, base_url="", status="UNHEALTHY", cost_per_1m_input=0.150, cost_per_1m_output=0.600),
        ]
        strategy = CostAware()
        assert await strategy.select(providers) is None

    @pytest.mark.asyncio
    async def test_zero_cost_is_cheapest(self):
        providers = [
            ProviderHealth(provider_id="groq", adapter=None, base_url="", cost_per_1m_input=0.059, cost_per_1m_output=0.079),
            ProviderHealth(provider_id="vllm", adapter=None, base_url="", cost_per_1m_input=0.0, cost_per_1m_output=0.0),
        ]
        strategy = CostAware()
        selected = await strategy.select(providers)
        assert selected.provider_id == "vllm"


class TestIntelligenceAware:
    def _make_providers(self):
        return [
            ProviderHealth(provider_id="openai", adapter=None, base_url="", tier="premium", cost_per_1m_input=0.150, cost_per_1m_output=0.600, weight=1.0),
            ProviderHealth(provider_id="groq", adapter=None, base_url="", tier="cheap", cost_per_1m_input=0.059, cost_per_1m_output=0.079, weight=1.0),
            ProviderHealth(provider_id="vllm", adapter=None, base_url="", tier="cheap", cost_per_1m_input=0.0, cost_per_1m_output=0.0, weight=1.0),
        ]

    @pytest.mark.asyncio
    async def test_simple_task_routes_to_cheap(self):
        providers = self._make_providers()
        request = InferRouteRequest(
            model="gpt-4o-mini",
            messages=[InferRouteMessage(role="user", content="hi")],
        )
        strategy = IntelligenceAware()
        selected = await strategy.select(providers, request=request)
        assert selected.tier == "cheap"

    @pytest.mark.asyncio
    async def test_complex_task_routes_to_premium(self):
        providers = self._make_providers()
        request = InferRouteRequest(
            model="gpt-4o",
            messages=[InferRouteMessage(role="user", content="Analyze the system design trade-offs of microservices vs monolith. Explain why you would choose one over the other. Write code to implement a sample service.")],
            max_tokens=4000,
        )
        strategy = IntelligenceAware()
        selected = await strategy.select(providers, request=request)
        assert selected.tier == "premium"
        assert selected.provider_id == "openai"

    @pytest.mark.asyncio
    async def test_medium_task_routes_to_standard(self):
        providers = [
            ProviderHealth(provider_id="openai", adapter=None, base_url="", tier="premium", cost_per_1m_input=0.150, cost_per_1m_output=0.600, weight=1.0),
            ProviderHealth(provider_id="groq", adapter=None, base_url="", tier="standard", cost_per_1m_input=0.059, cost_per_1m_output=0.079, weight=1.0),
            ProviderHealth(provider_id="vllm", adapter=None, base_url="", tier="cheap", cost_per_1m_input=0.0, cost_per_1m_output=0.0, weight=1.0),
        ]
        request = InferRouteRequest(
            model="gpt-4o-mini",
            messages=[InferRouteMessage(role="user", content="Tell me about the history of the Roman Empire and its impact on modern governance structures across Europe and North America.")],
        )
        strategy = IntelligenceAware()
        selected = await strategy.select(providers, request=request)
        assert selected.tier == "standard"

    @pytest.mark.asyncio
    async def test_classify_complexity_simple(self):
        strategy = IntelligenceAware()
        request = InferRouteRequest(
            model="gpt-4o-mini",
            messages=[InferRouteMessage(role="user", content="hello")],
        )
        complexity, source = await strategy.classify_complexity(request)
        assert complexity == "simple"
        assert source == "heuristic"

    @pytest.mark.asyncio
    async def test_classify_complexity_complex(self):
        strategy = IntelligenceAware()
        request = InferRouteRequest(
            model="gpt-4o",
            messages=[InferRouteMessage(role="user", content="Design and implement a distributed system with chain of thought reasoning. Analyze trade-offs and optimize the algorithm. Write code for the system.")],
            max_tokens=4000,
        )
        complexity, source = await strategy.classify_complexity(request)
        assert complexity == "complex"
        assert source == "heuristic"

    @pytest.mark.asyncio
    async def test_classify_complexity_medium(self):
        strategy = IntelligenceAware()
        request = InferRouteRequest(
            model="gpt-4o-mini",
            messages=[InferRouteMessage(role="user", content="Tell me about the history of the Roman Empire and its impact on modern governance structures across Europe and North America.")],
        )
        complexity, source = await strategy.classify_complexity(request)
        assert complexity == "medium"
        assert source == "heuristic"

    @pytest.mark.asyncio
    async def test_no_request_defaults_to_medium(self):
        providers = self._make_providers()
        strategy = IntelligenceAware()
        selected = await strategy.select(providers, request=None)
        assert selected is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_unhealthy(self):
        providers = [
            ProviderHealth(provider_id="openai", adapter=None, base_url="", status="UNHEALTHY", tier="premium"),
        ]
        strategy = IntelligenceAware()
        assert await strategy.select(providers) is None

    @pytest.mark.asyncio
    async def test_fallback_when_preferred_tier_empty(self):
        providers = [
            ProviderHealth(provider_id="openai", adapter=None, base_url="", tier="premium", weight=1.0, in_flight=0),
        ]
        request = InferRouteRequest(
            model="gpt-4o-mini",
            messages=[InferRouteMessage(role="user", content="hi")],
        )
        strategy = IntelligenceAware()
        selected = await strategy.select(providers, request=request)
        assert selected.provider_id == "openai"
