import json
import logging
import hashlib

logger = logging.getLogger("inferroute")

CLASSIFIER_SYSTEM_PROMPT = """You are a prompt complexity classifier. Classify the user's prompt into exactly one of three categories:

- "simple": Greetings, basic facts, translations, formatting, simple calculations, yes/no questions. Requires minimal reasoning.
- "medium": Explanations, summaries, general knowledge questions, moderate reasoning. A standard model handles this well.
- "complex": Code generation, system design, multi-step reasoning, analysis with trade-offs, architectural decisions, debugging. Needs a powerful model.

Respond with ONLY a JSON object: {"complexity": "simple|medium|complex", "confidence": 0.0-1.0}
No other text."""


class ComplexityClassifier:
    """Hybrid complexity classifier — heuristic first, LLM fallback for ambiguous cases."""

    def __init__(
        self,
        classifier_adapter=None,
        redis_client=None,
        cache_ttl: int = 3600,
        confidence_threshold: float = 0.7,
    ):
        self._adapter = classifier_adapter
        self._redis = redis_client
        self._cache_ttl = cache_ttl
        self._confidence_threshold = confidence_threshold

    def _cache_key(self, prompt: str) -> str:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return f"complexity:classify:{prompt_hash}"

    async def classify(self, request) -> tuple[str, float, str]:
        """Returns (complexity, confidence, source) where source is 'heuristic', 'cache', or 'llm'."""
        prompt = " ".join(m.content for m in request.messages if hasattr(m, "content"))

        # Step 1: Heuristic classification (0ms)
        complexity, confidence = self._heuristic_classify(prompt, request)

        if confidence >= self._confidence_threshold:
            return complexity, confidence, "heuristic"

        # Step 2: Check Redis cache for previous LLM classification
        if self._redis:
            try:
                cached = await self._redis.get(self._cache_key(prompt))
                if cached:
                    data = json.loads(cached)
                    return data["complexity"], data["confidence"], "cache"
            except Exception:
                pass

        # Step 3: LLM classification fallback (~200ms)
        if self._adapter:
            complexity, confidence = await self._llm_classify(prompt)

            # Cache the result
            if self._redis and complexity:
                try:
                    await self._redis.setex(
                        self._cache_key(prompt),
                        self._cache_ttl,
                        json.dumps({"complexity": complexity, "confidence": confidence}),
                    )
                except Exception:
                    pass

            return complexity, confidence, "llm"

        # No LLM adapter available — fall back to heuristic result
        return complexity, confidence, "heuristic"

    def _heuristic_classify(self, prompt: str, request) -> tuple[str, float]:
        """Fast keyword + signal based classification. Returns (complexity, confidence)."""
        prompt_lower = prompt.lower()
        prompt_len = len(prompt_lower)

        import re

        complex_keywords = [
            "analyze", "reasoning", "chain of thought", "step by step",
            "explain why", "design", "architect", "implement", "debug",
            "optimize", "refactor", "compare and contrast", "evaluate",
            "derive", "prove", "synthesize", "multi-step", "plan",
            "strategy", "algorithm", "complex", "nuanced", "critical thinking",
            "write code", "write a function", "write a class", "write a script",
            "code review", "system design", "trade-off", "tradeoff",
        ]
        simple_keywords = [
            "hi", "hello", "hey", "thanks", "thank you", "ok", "yes", "no",
            "what is", "define", "list", "summarize", "translate", "count",
            "format", "extract", "classify", "label", "sentiment",
            "what time", "what date", "how many", "calculate", "convert",
        ]

        complex_score = sum(1 for kw in complex_keywords if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower))
        simple_score = sum(1 for kw in simple_keywords if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower))

        if request.max_tokens and request.max_tokens > 2000:
            complex_score += 2

        if prompt_len > 2000:
            complex_score += 2
        elif prompt_len > 500:
            complex_score += 1
        elif prompt_len < 100:
            simple_score += 1

        # High confidence cases
        if complex_score >= 3 and simple_score == 0:
            return "complex", 0.9
        if simple_score >= 2 and complex_score == 0:
            return "simple", 0.9
        if complex_score >= 4:
            return "complex", 0.85
        if simple_score >= 3 and complex_score <= 1:
            return "simple", 0.80

        # Low confidence — needs LLM
        if complex_score >= 3:
            return "complex", 0.5
        if simple_score >= 2:
            return "simple", 0.5

        return "medium", 0.4

    async def _llm_classify(self, prompt: str) -> tuple[str, float]:
        """Use a fast cheap LLM to classify complexity. Returns (complexity, confidence)."""
        from app.gateway.models import InferRouteRequest, InferRouteMessage

        try:
            request = InferRouteRequest(
                model="classifier",
                messages=[
                    InferRouteMessage(role="system", content=CLASSIFIER_SYSTEM_PROMPT),
                    InferRouteMessage(role="user", content=prompt[:2000]),
                ],
                max_tokens=50,
                temperature=0.0,
            )

            response = await self._adapter.complete(request)
            content = response.content.strip()

            # Parse JSON response
            if content.startswith("{"):
                data = json.loads(content)
                complexity = data.get("complexity", "medium")
                confidence = float(data.get("confidence", 0.8))
                if complexity in ("simple", "medium", "complex"):
                    return complexity, confidence

            # Fallback: check for keywords in response
            if "simple" in content.lower():
                return "simple", 0.7
            elif "complex" in content.lower():
                return "complex", 0.7
            return "medium", 0.6

        except Exception as e:
            logger.warning("LLM classifier failed: %s — falling back to medium", e)
            return "medium", 0.5
