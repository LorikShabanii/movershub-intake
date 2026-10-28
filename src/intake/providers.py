"""LLM providers behind one small interface, plus an offline mock.

Three real providers, all using native structured outputs so the response is validated
against the pydantic schema before the pipeline sees it:

* ``GroqProvider``      — OpenAI-compatible endpoint with ``response_format=json_schema`` (strict).
  Groq's free plan needs no card; ``openai/gpt-oss-20b`` supports strict schemas. Default provider.
* ``MistralProvider``   — ``client.chat.parse(..., response_format=Extraction)`` (SDK 3.x: ``from mistralai.client import Mistral``).
* ``AnthropicProvider`` — ``client.messages.parse(..., output_format=Extraction)``.
  https://platform.claude.com/docs/en/build-with-claude/structured-outputs

Only synthetic test data is ever sent to a free tier (their terms allow training on inputs).
Pick with ``get_provider("groq" | "mistral" | "anthropic" | "mock")`` or ``INTAKE_PROVIDER``.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Protocol

from .rules import guess_language
from .schema import Extraction, empty_extraction


@dataclass
class ProviderResponse:
    extraction: Extraction
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str = "end_turn"
    note: str | None = None  # e.g. why a response fell back to an empty extraction


class LLMProvider(Protocol):
    def extract(self, system: str, user: str) -> ProviderResponse: ...


# --------------------------------------------------------------------------- rate limiting
# Free tiers allow roughly one request per second. Space calls out, and back off on 429s
# instead of failing the whole eval run on the second request.
MIN_INTERVAL_S = float(os.getenv("INTAKE_MIN_INTERVAL_S", "1.1"))
MAX_ATTEMPTS = int(os.getenv("INTAKE_MAX_ATTEMPTS", "6"))
_last_call_at = 0.0


def _is_rate_limited(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or "429" in str(exc) or "rate_limited" in str(exc).lower()


def _with_rate_limit_retry(call):
    global _last_call_at
    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        try:
            result = call()
            _last_call_at = time.monotonic()
            return result
        except Exception as exc:
            _last_call_at = time.monotonic()
            if not _is_rate_limited(exc) or attempt == MAX_ATTEMPTS:
                raise
            backoff = min(2 ** attempt, 30)  # 2, 4, 8, 16, 30 seconds
            print(f"rate limited (attempt {attempt}/{MAX_ATTEMPTS}), retrying in {backoff}s", file=sys.stderr)
            time.sleep(backoff)


# --------------------------------------------------------------------------- OpenAI-compatible (Groq)
def strict_json_schema(model) -> dict:
    """Pydantic schema -> the shape OpenAI-style strict mode wants.

    Strict mode requires every object to list all its properties as required and to forbid
    extra keys; it also rejects ``default`` and ``title``. Optional fields stay optional
    through ``anyOf [..., null]``, which pydantic already emits.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            node.pop("default", None)
            node.pop("title", None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


class GroqProvider:
    """Any OpenAI-compatible chat endpoint; defaults to Groq's free plan.

    Env: GROQ_API_KEY (or OPENAI_COMPAT_API_KEY), OPENAI_COMPAT_BASE_URL to point elsewhere.
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "openai/gpt-oss-20b"  # strict json_schema supported on Groq

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai") from e
        key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_COMPAT_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set in this shell")
        self.base_url = os.getenv("OPENAI_COMPAT_BASE_URL", self.DEFAULT_BASE_URL)
        self.model = model or os.getenv("INTAKE_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=key, base_url=self.base_url)
        self.schema = strict_json_schema(Extraction)

    def _call(self, system: str, user: str, strict: bool):
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        if strict:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": self.schema},
            }
        else:  # JSON mode: syntactically valid JSON, schema enforced locally instead
            kwargs["messages"][0]["content"] += (
                "\n\nAnswer with a single JSON object matching this JSON Schema exactly:\n"
                + json.dumps(self.schema)
            )
            kwargs["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        return self.client.chat.completions.create(**kwargs)

    def extract(self, system: str, user: str) -> ProviderResponse:
        mode, note = "strict", None
        try:
            response = _with_rate_limit_retry(lambda: self._call(system, user, strict=True))
        except Exception as exc:
            if _is_rate_limited(exc) or "400" not in str(exc):
                raise
            # Endpoint/model without strict schema support: fall back to JSON mode.
            mode = "json_object"
            response = _with_rate_limit_retry(lambda: self._call(system, user, strict=False))
        choice = response.choices[0]
        content = choice.message.content or ""
        try:
            parsed = Extraction.model_validate_json(content)
        except Exception as exc:  # schema violation -> empty, never a crash, but say why
            parsed = empty_extraction()
            note = f"{mode}: invalid JSON ({type(exc).__name__}: {str(exc)[:160]}); head={content[:120]!r}"
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            extraction=parsed,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
            stop_reason=str(getattr(choice, "finish_reason", "stop")),
            note=note or f"mode={mode}",
        )


# --------------------------------------------------------------------------- Mistral
class MistralProvider:
    """Mistral La Plateforme. Needs MISTRAL_API_KEY in the environment (never hard-code it)."""

    DEFAULT_MODEL = "mistral-small-latest"

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        try:
            from mistralai.client import Mistral  # SDK 3.x layout
        except ImportError:
            try:
                from mistralai import Mistral  # SDK 1.x / 2.x layout
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("pip install mistralai") from e
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not set in this shell")
        self.model = model or os.getenv("INTAKE_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.client = Mistral(api_key=key)

    def extract(self, system: str, user: str) -> ProviderResponse:
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=Extraction,
                max_tokens=self.max_tokens,
                temperature=0,
            )
        )
        choice = response.choices[0]
        parsed = getattr(choice.message, "parsed", None)
        if parsed is None:  # older SDKs expose only the JSON text
            try:
                parsed = Extraction.model_validate(json.loads(choice.message.content))
            except Exception:
                parsed = empty_extraction()
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            extraction=parsed,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
            stop_reason=str(getattr(choice, "finish_reason", "stop")),
        )


# --------------------------------------------------------------------------- Anthropic
class AnthropicProvider:
    """Claude API provider. Needs ANTHROPIC_API_KEY in the environment (never hard-code it)."""

    DEFAULT_MODEL = "claude-haiku-5-5"  # cheap and fast; structured outputs supported

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from e
        self.model = model or os.getenv("INTAKE_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.client = Anthropic()

    def extract(self, system: str, user: str) -> ProviderResponse:
        response = _with_rate_limit_retry(
            lambda: self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=Extraction,
            )
        )
        parsed = response.parsed_output
        if parsed is None:  # refusal or truncated JSON: fall back to an empty extraction, never crash
            parsed = empty_extraction()
        return ProviderResponse(
            extraction=parsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
            stop_reason=str(response.stop_reason),
        )


# --------------------------------------------------------------------------- Mock
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)\d{3}[\s-]?\d{3,4}")
_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_DMY = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b")
_REQ = re.compile(r"<customer_request>\n(.*)\n</customer_request>", re.S)


class MockProvider:
    """Deterministic, offline stand-in.

    Finds only what cheap regexes can find (email, phone, explicit dates, language) so that unit
    tests and the CI smoke run never need an API key. ``scripted`` lets a test inject a full
    Extraction for a given substring of the request.
    """

    def __init__(self, scripted: dict[str, Extraction] | None = None):
        self.scripted = scripted or {}
        self.calls = 0

    def extract(self, system: str, user: str) -> ProviderResponse:
        self.calls += 1
        m = _REQ.search(user)
        text = m.group(1) if m else user
        for key, ex in self.scripted.items():
            if key in text:
                return ProviderResponse(ex.model_copy(deep=True), 0, 0, "mock")

        ex = empty_extraction(guess_language(text))
        if e := _EMAIL.search(text):
            ex.contact_email.value, ex.contact_email.confidence, ex.contact_email.evidence = e.group(0), 0.95, e.group(0)
        if p := _PHONE.search(text):
            ex.contact_phone.value, ex.contact_phone.confidence, ex.contact_phone.evidence = p.group(0).strip(), 0.8, p.group(0).strip()
        if d := _ISO.search(text):
            ex.move_date.value, ex.move_date.confidence, ex.move_date.evidence = d.group(0), 0.9, d.group(0)
        elif d := _DMY.search(text):
            day, mon, year = d.groups()
            ex.move_date.value = f"{year}-{int(mon):02d}-{int(day):02d}"
            ex.move_date.confidence, ex.move_date.evidence = 0.85, d.group(0)
        return ProviderResponse(ex, len(user) // 4, 120, "mock")


PROVIDERS = ("groq", "mistral", "anthropic", "mock")


def get_provider(name: str) -> LLMProvider:
    if name == "groq":
        return GroqProvider()
    if name == "mistral":
        return MistralProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider '{name}' (use one of {PROVIDERS})")
