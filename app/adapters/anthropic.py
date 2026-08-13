import time
import httpx
from app.gateway.models import (
    InferRouteRequest,
    InferRouteResponse,
    InferRouteUsage,
)


class AnthropicAdapter:

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_connections: int = 100,
        connect_timeout: int = 10,
        read_timeout: int = 120,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30, pool=5),
            limits=httpx.Limits(max_connections=max_connections),
        )

    async def complete(self, request: InferRouteRequest, extra_headers: dict = None) -> InferRouteResponse:
        start = time.monotonic()

        system_prompt = ""
        messages = []
        for m in request.messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.stop:
            payload["stop_sequences"] = request.stop

        resp = await self._client.post("/messages", json=payload, headers=extra_headers)
        resp.raise_for_status()
        data = resp.json()

        latency_ms = (time.monotonic() - start) * 1000

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        return InferRouteResponse(
            id=data.get("id", ""),
            content=content,
            model=data.get("model", request.model),
            usage=InferRouteUsage(
                input_tokens=data["usage"]["input_tokens"],
                output_tokens=data["usage"]["output_tokens"],
            ),
            provider="anthropic",
            latency_ms=latency_ms,
        )

    async def stream_complete(self, request: InferRouteRequest, extra_headers: dict = None):
        """Yield SSE chunks from Anthropic. Converts to OpenAI-compatible SSE format."""
        system_prompt = ""
        messages = []
        for m in request.messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.stop:
            payload["stop_sequences"] = request.stop

        async with self._client.stream("POST", "/messages", json=payload, headers=extra_headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield line

    async def health_check(self) -> bool:
        try:
            resp = await self._client.post(
                "/messages",
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
