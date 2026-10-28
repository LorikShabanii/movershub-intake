import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate", ROOT / "evals" / "gate.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["gate"] = gate
spec.loader.exec_module(gate)

BASE = {"summary": {"field_accuracy": 0.90, "required_field_accuracy": 0.95, "decision_accuracy": 1.0, "flag_recall": 1.0,
                    "brier": 0.05, "cost_usd_per_request": 0.0010, "latency_p95_ms": 1000}}


def _run(latest_summary):
    return {r["metric"]: r["ok"] for r in gate.compare(BASE, {"summary": latest_summary})}


def test_equal_scores_pass():
    assert all(_run(BASE["summary"]).values())


def test_one_flaky_case_is_tolerated_but_a_real_drop_fails():
    one_case = dict(BASE["summary"], field_accuracy=0.83, decision_accuracy=0.917, latency_p95_ms=1200)
    r = _run(one_case)
    assert r["field_accuracy"] is True and r["decision_accuracy"] is True  # within one case
    assert r["latency_p95_ms"] is True  # within +25 %

    two_cases = dict(BASE["summary"], field_accuracy=0.75, decision_accuracy=0.833)
    r = _run(two_cases)
    assert r["field_accuracy"] is False and r["decision_accuracy"] is False


def test_flag_recall_has_zero_tolerance():
    assert _run(dict(BASE["summary"], flag_recall=0.99))["flag_recall"] is False


def test_cost_increase_beyond_tolerance_fails():
    r = _run(dict(BASE["summary"], cost_usd_per_request=0.0012))
    assert r["cost_usd_per_request"] is False


def test_brier_small_wobble_passes_large_fails():
    assert _run(dict(BASE["summary"], brier=0.06))["brier"] is True
    assert _run(dict(BASE["summary"], brier=0.10))["brier"] is False
