"""Cost and latency per request.

Prices are USD per million tokens. Defaults are taken from the public pricing page at the time of
writing and can be overridden with INTAKE_PRICE_IN / INTAKE_PRICE_OUT. Check
https://platform.claude.com/docs/en/about-claude/pricing before trusting a number in a report.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

PRICES_PER_MTOK = {
    # Free tiers bill nothing; set INTAKE_PRICE_IN/OUT from the provider's pricing page on a paid plan.
    "openai/gpt-oss-20b": (0.0, 0.0),
    "openai/gpt-oss-120b": (0.0, 0.0),
    "mistral-small-latest": (0.0, 0.0),
    "ministral-8b-latest": (0.0, 0.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "mock": (0.0, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    env_in, env_out = os.getenv("INTAKE_PRICE_IN"), os.getenv("INTAKE_PRICE_OUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    return PRICES_PER_MTOK.get(model, (0.0, 0.0))


@dataclass
class Usage:
    model: str = "none"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    model_called: bool = False
    extra: dict = field(default_factory=dict)

    def fill_cost(self) -> "Usage":
        p_in, p_out = price_for(self.model)
        self.cost_usd = round((self.input_tokens * p_in + self.output_tokens * p_out) / 1_000_000, 6)
        return self


class Stopwatch:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self.t0) * 1000)
