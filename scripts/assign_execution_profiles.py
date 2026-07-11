"""Assign an execution_profile to each paper-stage strategy by SELECTING the best
RISK ENGINE for it — the same selection the gauntlet now runs at promotion, applied
to the EXISTING population (a one-time backfill).

Why this exists: the gauntlet only selects + persists a profile for strategies that
pass through the promotion gate going forward (see
``forven/gauntlet/tasks.py:_select_and_persist_execution_profile``). Strategies
already at the paper stage carry no profile, so they size at the shared DEFAULT
engine (1% risk / 2x-ATR). This tool backfills them: for each strategy it sweeps the
candidate risk engines (fraction / atr / kelly / full) through the SHARED kernel and
picks the one that maximizes a RISK-ADJUSTED objective (default Sharpe), then writes
the winner into ``params['execution_profile']`` — the one key the kernel honors.

Selection logic is the SINGLE shared definition in
``forven.strategies.execution_selection`` — identical to the pipeline — so a
backfilled profile matches what a re-gauntlet would choose.

SAFE: dry-run report by default; --apply BACKS UP the DB first, then writes profiles.
Idempotent: a strategy that already carries a profile is skipped unless --force.

    python scripts/assign_execution_profiles.py                 # report only
    python scripts/assign_execution_profiles.py --apply         # back up + write profiles
    python scripts/assign_execution_profiles.py --only S02177   # single strategy
    python scripts/assign_execution_profiles.py --objective calmar
    python scripts/assign_execution_profiles.py --max-risk 0.03 # cap risk/trade (default 0.05)
    python scripts/assign_execution_profiles.py --max-dd 0.50   # reject profiles drawing >50%
    python scripts/assign_execution_profiles.py --force         # re-select even if a profile exists
    python scripts/assign_execution_profiles.py --json          # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys


def _load_paper_strategies(only: str | None) -> list[dict]:
    from forven.db import get_db

    with get_db() as conn:
        if only:
            rows = conn.execute(
                "SELECT id, name, type, runtime_type, symbol, timeframe, params, stage "
                "FROM strategies WHERE id = ?", (only,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, type, runtime_type, symbol, timeframe, params, stage "
                "FROM strategies WHERE LOWER(COALESCE(stage, status)) IN ('paper', 'paper_trading')"
            ).fetchall()
    return [dict(r) for r in rows]


def optimize_one(strat: dict, *, objective: str, max_risk: float, max_dd: float, min_trades: int) -> dict:
    """Select the best risk engine for one strategy via the SHARED selector."""
    from forven.db import _strategy_asset_token
    from forven.strategies.execution_selection import select_execution_profile

    asset = _strategy_asset_token(strat.get("symbol")) or str(strat.get("symbol") or "")
    sel = select_execution_profile(
        strategy_id=strat["id"],
        asset=asset,
        strategy_type=str(strat.get("runtime_type") or strat.get("type") or ""),
        params=strat["_params"],
        timeframe=strat["_tf"],
        objective=objective,
        max_risk=max_risk,
        max_dd=max_dd,
        min_trades=min_trades,
        regime_gate=False,  # match the paper scanner's kernel call (the parity reference)
        lean=True,  # use the SAME lean candidate grid the gauntlet promotion selection uses
                    # (~15 candidates vs ~30) — faster AND matches what a re-gauntlet picks.
    )
    return {
        "id": strat["id"], "name": strat.get("name"), "type": strat.get("type"),
        "timeframe": strat["_tf"], "stage": strat.get("stage"),
        "objective": objective,
        "baseline": sel.get("baseline"),
        "best": sel.get("best"),
        "chosen": sel.get("chosen"),
        "chosen_label": sel.get("chosen_label"),
        "n_candidates": sel.get("n_candidates"),
        "n_eligible": sel.get("n_eligible"),
    }


def _backup_db() -> str:
    from forven.backups import create_managed_db_backup

    return str(create_managed_db_backup("assign-execution-profiles"))


def _write_profile(strategy_id: str, profile: dict) -> None:
    from forven.db import get_db
    from forven.strategies.sizing import normalize_execution_controls

    normalized = normalize_execution_controls(profile) or profile
    with get_db() as conn:
        row = conn.execute("SELECT params FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        params = {}
        if row and row["params"]:
            try:
                params = json.loads(row["params"]) or {}
            except Exception:
                params = {}
        if not isinstance(params, dict):
            params = {}
        params["execution_profile"] = normalized
        conn.execute(
            "UPDATE strategies SET params = ?, updated_at = datetime('now') WHERE id = ?",
            (json.dumps(params, sort_keys=True), strategy_id),
        )


def _fmt_pct(v):
    return "   —" if v is None else f"{v:.2%}"


def _fmt_num(v):
    return "  —" if v is None else f"{v:.2f}"


def main(argv=None) -> int:
    from forven.strategies.execution_selection import candidate_profiles, profile_label

    ap = argparse.ArgumentParser(description="Select + assign the best risk engine per paper strategy (risk-adjusted).")
    ap.add_argument("--apply", action="store_true", help="Back up the DB, then write the winning profile into each strategy's params.")
    ap.add_argument("--only", help="Restrict to a single strategy id.")
    ap.add_argument("--objective", default="sharpe_ratio", help="Risk-adjusted objective: sharpe_ratio (default), sortino, calmar.")
    ap.add_argument("--max-risk", type=float, default=0.05, help="Cap on risk_per_trade searched (fraction; default 0.05).")
    ap.add_argument("--max-dd", type=float, default=0.50, help="Reject profiles whose backtest max drawdown exceeds this (fraction; default 0.50).")
    ap.add_argument("--min-trades", type=int, default=10, help="Reject profiles with fewer trades than this (default 10).")
    ap.add_argument("--force", action="store_true", help="Re-select even for strategies that already carry an execution_profile.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args(argv)

    strategies = _load_paper_strategies(args.only)
    if not strategies:
        print("No matching strategies found.")
        return 0

    for s in strategies:
        p = s.get("params")
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                p = {}
        s["_params"] = p if isinstance(p, dict) else {}
        s["_tf"] = (str(s.get("timeframe") or "1h").strip().lower() or "1h")
        s["_has_profile"] = isinstance(s["_params"].get("execution_profile"), dict) and bool(s["_params"]["execution_profile"])

    pending = [s for s in strategies if args.force or not s["_has_profile"]]
    skipped_existing = [s for s in strategies if not args.force and s["_has_profile"]]

    print(f"Selecting risk engines for {len(pending)} strateg(y/ies) "
          f"[objective={args.objective}, engines=fraction/atr/kelly/full, max_risk={args.max_risk:.0%}, max_dd={args.max_dd:.0%}]")
    if skipped_existing:
        print(f"  ({len(skipped_existing)} already carry a profile — skipped; use --force to re-select)")
    print(f"Grid: {len(candidate_profiles(max_risk=args.max_risk, lean=True))} candidates/strategy (lean, matches the gauntlet) via the shared kernel.\n")

    results = []
    for i, strat in enumerate(pending, 1):
        try:
            res = optimize_one(strat, objective=args.objective, max_risk=args.max_risk, max_dd=args.max_dd, min_trades=args.min_trades)
        except Exception as exc:
            res = {"id": strat.get("id"), "name": strat.get("name"), "error": str(exc), "baseline": None, "best": None, "chosen": None}
        results.append(res)
        print(f"  [{i}/{len(pending)}] {res.get('id')} -> {res.get('chosen_label', 'error')}", flush=True)

    applied = 0
    backup = None
    if args.apply:
        to_write = [r for r in results if isinstance(r.get("chosen"), dict) and r["chosen"] and not r.get("error")]
        if to_write:
            backup = _backup_db()
            for r in to_write:
                _write_profile(r["id"], r["chosen"])
                applied += 1

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0

    print()
    if backup:
        print(f"Backed up DB -> {backup}")
        print(f"Wrote execution_profile to {applied} strateg(y/ies) (default-engine winners keep the shared default).\n")
    header = f"{'id':<10} {'tf':<4} {'sharpe':<7} {'ret base→opt':<22} {'maxDD':<8} {'trades':<7} chosen engine"
    print(header)
    print("-" * (len(header) + 12))
    for r in sorted(results, key=lambda r: -((r.get("best") or {}).get("score") or float("-inf"))):
        if r.get("error"):
            print(f"{str(r.get('id')):<10} ERROR: {r['error']}")
            continue
        base = (r.get("baseline") or {}).get("total_return")
        best = r.get("best") or {}
        ret = f"{_fmt_pct(base)} → {_fmt_pct(best.get('total_return'))}"
        dd = best.get("max_drawdown")
        dds = "   —" if dd is None else f"{dd:.1%}"
        trades = best.get("trades")
        flag = "" if best.get("eligible", True) else "  (no eligible — best-effort)"
        print(f"{str(r['id']):<10} {str(r.get('timeframe'))[:4]:<4} {_fmt_num(best.get('sharpe')):<7} {ret:<22} {dds:<8} "
              f"{str(int(trades)) if trades is not None else '—':<7} {profile_label(r.get('chosen'))}{flag}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to back up + write these profiles.")
    else:
        print("\nProfiles written. Re-score + re-baseline so paper sizes off them:")
        print("  python scripts/rescore_paper_strategies.py --apply")
        print("  python scripts/reset_paper_trades.py   # clean re-baseline: close current positions, re-open correctly sized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
