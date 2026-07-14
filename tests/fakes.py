"""
fakes.py — Test doubles for the LLM and webhook client.

FakeLLM: Canned response queue — .invoke() returns an AIMessage
with .content from the next queued response.

FakeWebhookClient: Records POSTed payloads in a list instead of
sending them. Configurable .status_code for success/failure testing.
"""

from dataclasses import dataclass, field
from langchain_core.messages import AIMessage


class FakeLLM:
    """
    A fake LLM that returns canned responses in order.

    Usage:
        llm = FakeLLM(["response1", "response2"])
        result = llm.invoke([...])  # returns AIMessage(content="response1")
        result = llm.invoke([...])  # returns AIMessage(content="response2")
    """

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self._call_index = 0
        self.call_history: list[list] = []

    def invoke(self, messages, **kwargs):
        """Return the next canned response as an AIMessage."""
        self.call_history.append(messages)
        if self._call_index < len(self.responses):
            content = self.responses[self._call_index]
            self._call_index += 1
        else:
            content = '{"intent": "general"}'  # Safe fallback
        return AIMessage(content=content)


@dataclass
class FakeResponse:
    """Mimics an httpx.Response with a configurable status_code."""
    status_code: int = 200


class FakeWebhookClient:
    """
    A fake httpx client that records POST calls instead of sending them.

    Usage:
        client = FakeWebhookClient(status_code=200)
        resp = client.post("https://...", json={...})
        assert len(client.posts) == 1
        assert client.posts[0]["json"] == {...}
    """

    def __init__(self, status_code: int = 200, raise_on_post: Exception | None = None):
        self.status_code = status_code
        self.raise_on_post = raise_on_post
        self.posts: list[dict] = []

    def post(self, url, json=None, **kwargs):
        """Record the POST and return a FakeResponse (or raise)."""
        if self.raise_on_post:
            raise self.raise_on_post
        self.posts.append({"url": url, "json": json, **kwargs})
        return FakeResponse(status_code=self.status_code)
