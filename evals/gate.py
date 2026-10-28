"""CI gate: compare the latest eval run against the committed baseline.

    python evals/gate.py                      # exit 1 if anything got worse
    python evals/gate.py --update-baseline    # accept the latest run as the new baseline (do this on purpose, in a PR)

"Worse" means: a higher-is-better metric dropped below baseline minus its tolerance, or a
lower-is-better metric rose above baseline plus its tolerance. Tolerances live in baseline.json
so a change to them is reviewable like any other change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evals" / "baseline.json"
LATEST = ROOT / "evals" / "results" / "latest.json"

HIGHER_IS_BETTER = ["field_accuracy", "required_field_accuracy", "decision_accuracy", "flag_recall"]
LOWER_IS_BETTER = ["brier", "cost_usd_per_request", "latency_p95_ms"]
# Accuracy tolerances equal ONE case on the 12-case testset_v1 (1/12 decisions; one all-null case
# costs 11 of 156 field checks). A set this small cannot separate model noise from a real
# regression, so the gate lets exactly one case flip and nothing more. Shrink these as the set grows.
DEFAULT_TOLERANCE = {
    "field_accuracy": 0.08,
    "required_field_accuracy": 0.10,
    "decision_accuracy": 0.09,
    "flag_recall": 0.0,
    "brier": 0.02,
    "cost_usd_per_request": 0.10,  # relative: +10 %
    "latency_p95_ms": 0.25,  # relative: +25 % (network noise)
}
RELATIVE = {"cost_usd_per_request", "latency_p95_ms"}


def compare(baseline: dict, latest: dict) -> list[dict]:
    tol = {**DEFAULT_TOLERANCE, **baseline.get("tolerance", {})}
    base, now = baseline["summary"], latest["summary"]
    rows = []
    for m in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        b, n = base.get(m), now.get(m)
        if b is None or n is None:
            continue
        t = tol[m]
        if m in HIGHER_IS_BETTER:
            limit = b - t
            ok = n >= limit
        else:
            limit = b * (1 + t) if m in RELATIVE else b + t
            ok = n <= limit
        rows.append({"metric": m, "baseline": b, "latest": n, "limit": round(limit, 6), "ok": ok})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", default=str(LATEST))
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)

    latest = json.loads(Path(args.latest).read_text(encoding="utf-8"))
    if args.update_baseline:
        old = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if Path(args.baseline).exists() else {}
        new = {"testset": latest["testset"], "provider": latest["provider"], "model": latest["model"],
               "run_at": latest["run_at"], "tolerance": old.get("tolerance", DEFAULT_TOLERANCE),
               "summary": latest["summary"]}
        Path(args.baseline).write_text(json.dumps(new, indent=2), encoding="utf-8")
        print(f"baseline updated from run at {latest['run_at']} ({latest['model']})")
        return 0

    if not Path(args.baseline).exists():
        print("no baseline.json yet: run score.py with the real provider, review, then gate.py --update-baseline")
        return 1
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    if baseline.get("testset") != latest.get("testset"):
        print(f"testset mismatch: baseline={baseline.get('testset')} latest={latest.get('testset')}")
        return 1

    rows = compare(baseline, latest)
    print(f"{'metric':<26}{'baseline':>12}{'latest':>12}{'limit':>12}  result")
    for r in rows:
        print(f"{r['metric']:<26}{r['baseline']:>12}{r['latest']:>12}{r['limit']:>12}  {'ok' if r['ok'] else 'REGRESSION'}")
    failed = [r["metric"] for r in rows if not r["ok"]]
    if failed:
        print(f"\nGATE FAILED: {', '.join(failed)} got worse than baseline. Not shipping.")
        return 1
    print("\nGATE PASSED: nothing scored worse than baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
