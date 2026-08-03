import logging
import httpx
from app.core.config import settings

logger = logging.getLogger("inferroute")


class EmbeddingService:

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = None,
        dim: int = None,
    ):
        self._model = model or settings.cache_embedding_model
        self._dim = dim or settings.cache_embedding_dim
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(connect=10, read=30, write=10, pool=5),
        )

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            "/embeddings",
            json={"input": text, "model": self._model},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    @property
    def dim(self) -> int:
        return self._dim

    async def close(self):
        await self._client.aclose()
