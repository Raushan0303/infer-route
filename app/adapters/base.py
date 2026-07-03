from typing import Protocol
from app.gateway.models import InferRouteRequest, InferRouteResponse


class LLMProviderAdapter(Protocol):

    async def complete(self, request: InferRouteRequest) -> InferRouteResponse:
        ...

    async def health_check(self) -> bool:
        ...

    async def close(self) -> None:
        ...
