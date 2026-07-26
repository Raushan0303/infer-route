import logging
from app.core.config import settings

logger = logging.getLogger("inferroute")


class RAGEmbeddingService:

    def __init__(self, base_url: str = "https://api.openai.com/v1", api_key: str = ""):
        self._base_url = base_url
        self._api_key = api_key
        self._model = settings.rag_embedding_model
        self._dim = settings.rag_embedding_dim
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = await client.post(
            "/embeddings",
            json={"input": text, "model": self._model},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        resp = await client.post(
            "/embeddings",
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
