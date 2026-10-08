"""Deterministic test doubles. These are NOT semantic embedding models."""
from types import SimpleNamespace


class FakeEmbeddings:
    identity = "test:keyword-vectors:v1"

    def encode(self, texts):
        groups = [("solar", "efficiency", "23.8"), ("filter", "water", "aquapure"),
                  ("founded", "founder", "company", "priya")]
        return [[float(sum(word in text.lower() for word in group)) for group in groups] + [0.1]
                for text in texts]


def fake_client(answer="The efficiency is 23.8% [1].", finish_reason="stop"):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=answer), finish_reason=finish_reason)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), captured
