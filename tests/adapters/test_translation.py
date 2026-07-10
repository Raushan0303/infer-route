import pytest
from unittest.mock import MagicMock, patch
from app.adapters.openai import OpenAIAdapter
from app.adapters.anthropic import AnthropicAdapter
from app.gateway.models import InferRouteRequest, InferRouteMessage


@pytest.mark.asyncio
async def test_openai_adapter_sends_correct_format():
    adapter = OpenAIAdapter(base_url="https://api.openai.com/v1", api_key="sk-test")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(adapter._client, "post", return_value=mock_response):
        req = InferRouteRequest(
            messages=[
                InferRouteMessage(role="system", content="You are helpful."),
                InferRouteMessage(role="user", content="Say hello"),
            ],
            model="gpt-4o-mini",
        )
        resp = await adapter.complete(req)

        call_args = adapter._client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."

        assert resp.content == "Hello!"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.provider == "openai"

    await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_extracts_system_prompt():
    adapter = AnthropicAdapter(
        base_url="https://api.anthropic.com/v1", api_key="sk-ant-test"
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(adapter._client, "post", return_value=mock_response):
        req = InferRouteRequest(
            messages=[
                InferRouteMessage(role="system", content="You are helpful."),
                InferRouteMessage(role="user", content="Say hello"),
            ],
            model="claude-sonnet-4-20250514",
        )
        resp = await adapter.complete(req)

        call_args = adapter._client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in payload["messages"])

        assert resp.content == "Hello!"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.provider == "anthropic"

    await adapter.close()
