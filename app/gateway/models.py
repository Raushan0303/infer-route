from pydantic import BaseModel
from time import time


class InferRouteMessage(BaseModel):
    role: str
    content: str


class InferRouteRequest(BaseModel):
    messages: list[InferRouteMessage]
    model: str
    max_tokens: int | None = None
    temperature: float = 1.0
    stream: bool = False
    stop: list[str] | None = None
    namespace: str | None = None


class InferRouteUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class InferRouteResponse(BaseModel):
    id: str = ""
    content: str
    model: str
    usage: InferRouteUsage
    provider: str
    latency_ms: float = 0.0


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = int(time())
    model: str
    choices: list[dict]
    usage: dict
