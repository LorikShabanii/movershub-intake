"""Run the pipeline over a versioned test set and score it.

    python evals/score.py --provider groq --testset evals/testset_v1.jsonl
    python evals/score.py --provider mock            # offline smoke run of the harness itself

Writes evals/results/<testset>-<provider>-<timestamp>.json and evals/results/latest.json.

Metrics (higher is better unless noted):
  field_accuracy           share of expected (field, value) pairs the pipeline got right
  required_field_accuracy  same, restricted to the fields a quote cannot do without
  decision_accuracy        share of cases routed to the expected queue
  flag_recall              share of expected safety flags that were raised
  brier                    mean (confidence - correct)^2 over extracted fields; lower is better (calibration)
  cost_usd_per_request     lower is better
  latency_p95_ms           lower is better
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intake.pipeline import run  # noqa: E402
from intake.providers import get_provider  # noqa: E402
from intake.schema import ALL_VALUE_FIELDS, REQUIRED_FIELDS  # noqa: E402

FUZZY_FIELDS = {"pickup_address", "dropoff_address", "home_size", "special_items", "contact_name"}
REQUIRED = set(REQUIRED_FIELDS) | {"contact_phone", "contact_email"}
_TOKEN = re.compile(r"[\wëçäöüß]+", re.UNICODE)


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(s)}


def field_correct(name: str, predicted, expected) -> bool:
    if expected is None or predicted is None:
        return predicted is None and expected is None
    if name == "contact_phone":
        return re.sub(r"\D", "", str(predicted)) == re.sub(r"\D", "", str(expected))
    if name in FUZZY_FIELDS:
        a, b = _tokens(str(predicted)), _tokens(str(expected))
        return bool(a and b) and len(a & b) / len(a | b) >= 0.6
    return str(predicted).strip().lower() == str(expected).strip().lower()


def score_case(case: dict, result) -> dict:
    out = {"id": case["id"], "decision": result.decision.value, "expected_decision": case["expected_decision"],
           "decision_ok": result.decision.value == case["expected_decision"],
           "flags": sorted({f.code for f in result.flags}),
           "cost_usd": result.usage.cost_usd, "latency_ms": result.usage.latency_ms,
           "stop_reason": result.usage.extra.get("stop_reason"), "note": result.usage.extra.get("note"), "fields": {}}
    expected_flags = set(case.get("expected_flags", []))
    raised = {f.code for f in result.flags}
    out["flag_hits"], out["flag_total"] = len(expected_flags & raised), len(expected_flags)

    expected = case.get("expected")
    if expected and result.extraction:
        ex = result.extraction
        for name in ALL_VALUE_FIELDS:
            f = getattr(ex, name)
            ok = field_correct(name, f.value, expected.get(name))
            out["fields"][name] = {"predicted": f.value, "expected": expected.get(name), "ok": ok,
                                   "confidence": f.confidence}
        out["fields"]["language"] = {"predicted": ex.language.value, "expected": expected.get("language"),
                                     "ok": ex.language.value == expected.get("language"), "confidence": None}
    return out


def aggregate(cases: list[dict]) -> dict:
    all_f = [v for c in cases for v in c["fields"].values()]
    req_f = [v for c in cases for k, v in c["fields"].items() if k in REQUIRED]
    brier_pool = [(v["confidence"], v["ok"]) for v in all_f
                  if v["confidence"] is not None and (v["predicted"] is not None or v["expected"] is not None)]
    lat = sorted(c["latency_ms"] for c in cases if c["latency_ms"])
    flag_total = sum(c["flag_total"] for c in cases)
    return {
        "n_cases": len(cases),
        "field_accuracy": round(sum(v["ok"] for v in all_f) / len(all_f), 4) if all_f else 0.0,
        "required_field_accuracy": round(sum(v["ok"] for v in req_f) / len(req_f), 4) if req_f else 0.0,
        "decision_accuracy": round(sum(c["decision_ok"] for c in cases) / len(cases), 4),
        "flag_recall": round(sum(c["flag_hits"] for c in cases) / flag_total, 4) if flag_total else 1.0,
        "brier": round(statistics.fmean((conf - float(ok)) ** 2 for conf, ok in brier_pool), 4) if brier_pool else 0.0,
        "cost_usd_per_request": round(sum(c["cost_usd"] for c in cases) / len(cases), 6),
        "latency_p50_ms": lat[len(lat) // 2] if lat else 0,
        "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.getenv("INTAKE_PROVIDER", "groq"), choices=["groq", "mistral", "anthropic", "mock"])
    ap.add_argument("--testset", default=str(ROOT / "evals" / "testset_v1.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated case ids; partial runs never touch latest.json")
    args = ap.parse_args(argv)

    provider = get_provider(args.provider)
    cases = [json.loads(l) for l in open(args.testset, encoding="utf-8") if l.strip()]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            print(f"no cases match --only {args.only}"); return 2
    scored = []
    t0 = time.time()
    for case in cases:
        today = date.fromisoformat(case["today"])
        res = run(case["text"], provider, today=today)
        s = score_case(case, res)
        scored.append(s)
        print(f"{'OK ' if s['decision_ok'] else 'BAD'} {case['id']:<22} {s['decision']:<20} "
              f"fields {sum(v['ok'] for v in s['fields'].values())}/{len(s['fields']) or '-'}  flags={s['flags']}")
        if s["note"] and not s["note"].startswith("mode="):
            print(f"    note: {s['note']}  (stop_reason={s['stop_reason']})")

    summary = aggregate(scored)
    report = {
        "testset": Path(args.testset).name,
        "provider": args.provider,
        "model": getattr(provider, "model", args.provider),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
        "summary": summary,
        "cases": scored,
    }
    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "partial" if args.only else Path(args.testset).stem
    out = Path(args.out) if args.out else out_dir / f"{prefix}-{args.provider}-{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.only:  # a partial run must not become the thing the gate compares against
        (out_dir / "latest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
