import time
import httpx
from app.gateway.models import (
    InferRouteRequest,
    InferRouteResponse,
    InferRouteUsage,
)


class VLLMAdapter:

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        max_connections: int = 100,
        connect_timeout: int = 10,
        read_timeout: int = 120,
        provider_id: str = "vllm",
        model_mapping: dict = None,
    ):
        headers = {"content-type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._provider_id = provider_id
        self._model_mapping = model_mapping or {}
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30, pool=5),
            limits=httpx.Limits(max_connections=max_connections),
        )

    async def complete(self, request: InferRouteRequest, extra_headers: dict = None) -> InferRouteResponse:
        start = time.monotonic()

        model = self._model_mapping.get(request.model, self._model_mapping.get("default", request.model))
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop

        resp = await self._client.post("/chat/completions", json=payload, headers=extra_headers)
        resp.raise_for_status()
        data = resp.json()

        latency_ms = (time.monotonic() - start) * 1000

        return InferRouteResponse(
            id=data.get("id", ""),
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", request.model),
            usage=InferRouteUsage(
                input_tokens=data["usage"].get("prompt_tokens", 0),
                output_tokens=data["usage"].get("completion_tokens", 0),
            ),
            provider=self._provider_id,
            latency_ms=latency_ms,
        )

    async def stream_complete(self, request: InferRouteRequest, extra_headers: dict = None):
        """Yield SSE chunks from provider. OpenAI-compatible streaming."""
        model = self._model_mapping.get(request.model, self._model_mapping.get("default", request.model))
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop

        async with self._client.stream("POST", "/chat/completions", json=payload, headers=extra_headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield line
                    if line == "data: [DONE]":
                        break

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
