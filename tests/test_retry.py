import pytest

from intake import providers


class FakeRateLimit(Exception):
    status_code = 429


def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(providers, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)  # no real waiting in tests
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimit("Status 429. Rate limit exceeded")
        return "ok"

    assert providers._with_rate_limit_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(providers, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(providers, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)

    def always_limited():
        raise FakeRateLimit("Status 429")

    with pytest.raises(FakeRateLimit):
        providers._with_rate_limit_retry(always_limited)


def test_non_rate_limit_errors_are_not_retried(monkeypatch):
    monkeypatch.setattr(providers, "MIN_INTERVAL_S", 0.0)
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        providers._with_rate_limit_retry(broken)
    assert calls["n"] == 1


def test_strict_schema_shape():
    from intake.schema import Extraction

    schema = providers.strict_json_schema(Extraction)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "default" not in node and "title" not in node
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)

    check(schema)
    assert "language" in schema["properties"] and "$defs" in schema
