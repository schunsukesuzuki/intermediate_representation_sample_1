"""The Ollama client can use internal Compose DNS but cannot call arbitrary hosts."""
import json

import pytest

from gnn_rag.llm import OllamaClient


def test_ollama_internal_host_and_payload(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, *_):
            return b'{"message":{"content":"grounded [E1]"}}'

    def fetch(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("gnn_rag.llm.urlopen", fetch)
    result = OllamaClient(endpoint="http://ollama:11434", timeout=300).answer("[E1]")
    assert result == "grounded [E1]"
    assert captured["url"] == "http://ollama:11434/api/chat"
    assert captured["data"]["stream"] is False
    assert captured["timeout"] == 300


@pytest.mark.parametrize("host", [
    "https://ollama:11434", "http://outside.test:11434", "http://ollama:8888",
    "http://ollama:11434/untrusted", "http://user:pass@ollama:11434",
])
def test_no_external_or_ambiguous_ollama_address(host):
    with pytest.raises(ValueError):
        OllamaClient(endpoint=host).answer("question")
