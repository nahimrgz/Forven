"""Recover strategies wrongly archived after a cross-asset re-home.

Root cause (fixed in forven/policy.py): the cross-asset validation sweep's
"best context" selection scored an IS/OOS Sharpe gap with a SIGNED subtraction, so a
lucky-OOS context (weak/negative in-sample, lucky out-of-sample on a few trades)
sailed under the reject threshold, won "best", re-homed the strategy onto the wrong
asset, and was then killed by the promotion gate. Result: strategies that were strong
on their DECLARED `_asset` got re-homed onto SOL/ETH/etc. and archived.

This sweep finds those casualties — archived strategies whose current symbol no longer
matches their declared `_asset`, and which have a genuinely robust backtest ON their
declared asset — and recovers them:
  1. restore symbol / timeframe / metrics to that strong declared-asset backtest, and
  2. un-archive to quick_screen (the only valid transition out of archived) so the
     NOW-FIXED pipeline re-validates them and re-promotes the real winners.

IMPORTANT: the policy fix must be LIVE (backend restarted) before --apply, otherwise
the still-running old pipeline re-homes/re-archives them again. Read-only by default.

    python scripts/recover_cross_asset_rehomed_strategies.py                  # dry-run report
    python scripts/recover_cross_asset_rehomed_strategies.py --min-sharpe 1.0 --min-trades 30
    python scripts/recover_cross_asset_rehomed_strategies.py --ids S03523,S02893
    python scripts/recover_cross_asset_rehomed_strategies.py --apply          # back up + recover
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _base(sym) -> str:
    return str(sym or "").upper().split("/")[0].split("-")[0].strip()


def _declared_asset(params) -> str | None:
    try:
        p = json.loads(params) if isinstance(params, str) else params
    except Exception:
        return None
    if not isinstance(p, dict):
        return None
    for key in ("_asset", "asset"):
        v = p.get(key)
        if v and str(v).strip().upper() not in ("", "GENERIC"):
            return _base(v)
    return None


def _best_declared_backtest(conn, sid: str, declared: str, min_trades: int, min_sharpe: float):
    """Best non-deleted backtest on the strategy's declared asset meeting thresholds."""
    rows = conn.execute(
        "SELECT symbol, timeframe, metrics_json FROM backtest_results "
        "WHERE strategy_id = ? AND deleted_at IS NULL",
        (sid,),
    ).fetchall()
    best = None
    for r in rows:
        if _base(r["symbol"]) != declared:
            continue
        try:
            m = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
        except Exception:
            m = {}
        if not isinstance(m, dict):
            continue
        trades = m.get("total_trades") or m.get("num_trades") or 0
        sharpe = m.get("sharpe") or m.get("sharpe_ratio") or 0
        try:
            trades = float(trades)
            sharpe = float(sharpe)
        except (TypeError, ValueError):
            continue
        if trades < min_trades or sharpe < min_sharpe:
            continue
        if best is None or sharpe > best["sharpe"]:
            best = {
                "symbol": r["symbol"],
                "timeframe": r["timeframe"],
                "metrics": m,
                "trades": int(trades),
                "sharpe": round(sharpe, 2),
            }
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="back up the DB, then recover (default: report only)")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-sharpe", type=float, default=1.0)
    ap.add_argument("--ids", type=str, default="", help="comma-separated strategy ids to limit to")
    ap.add_argument("--limit", type=int, default=0, help="cap how many to recover (0 = no cap)")
    args = ap.parse_args()

    from forven.db import get_db

    only = {s.strip() for s in args.ids.split(",") if s.strip()}

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, symbol, timeframe, stage, status, params FROM strategies "
            "WHERE status IN ('archived', 'rejected') AND params LIKE '%_asset%'"
        ).fetchall()

        targets = []
        for r in rows:
            sid = r["id"]
            if only and sid not in only:
                continue
            declared = _declared_asset(r["params"])
            if not declared or _base(r["symbol"]) == declared:
                continue  # not re-homed off its declared asset
            best = _best_declared_backtest(conn, sid, declared, args.min_trades, args.min_sharpe)
            if best is None:
                continue  # no robust declared-asset backtest -> not a clear casualty
            targets.append((sid, declared, r["symbol"], best))

    targets.sort(key=lambda t: t[3]["sharpe"], reverse=True)
    if args.limit > 0:
        targets = targets[: args.limit]

    print(f"\nCross-asset re-home casualties (archived, robust on declared asset): {len(targets)}")
    print(f"  filters: min_trades={args.min_trades} min_sharpe={args.min_sharpe}\n")
    for sid, declared, cur, best in targets:
        print(
            f"  {sid:8} declared={declared:5} re-homed_to={str(cur):10} -> restore "
            f"{best['symbol']}/{best['timeframe']} (trades={best['trades']}, Sharpe={best['sharpe']})"
        )

    if not args.apply:
        print(f"\n[dry-run] would recover {len(targets)} strategies. Re-run with --apply once the "
              "policy fix is LIVE (backend restarted).")
        return 0

    if not targets:
        print("\nNothing to recover.")
        return 0

    from forven.backups import create_managed_db_backup

    backup = create_managed_db_backup("recover-cross-asset-rehomed-strategies")
    print(f"\nDB backed up to {backup}")

    from forven.brain import transition_stage

    recovered = 0
    failed = []
    for sid, declared, cur, best in targets:
        try:
            from forven.db import build_strategy_container_name  # canonical {ASSET}-{TYPE}-{ID} name
            new_name = None
            try:
                with get_db() as conn:
                    stype = conn.execute("SELECT type FROM strategies WHERE id = ?", (sid,)).fetchone()
                if stype:
                    new_name = build_strategy_container_name(symbol=best["symbol"], type_=stype["type"], strategy_id=sid)
            except Exception:
                new_name = None
            with get_db() as conn:
                if new_name:
                    conn.execute(
                        "UPDATE strategies SET symbol=?, timeframe=?, metrics=?, name=? WHERE id=?",
                        (best["symbol"], best["timeframe"], json.dumps(best["metrics"]), new_name, sid),
                    )
                else:
                    conn.execute(
                        "UPDATE strategies SET symbol=?, timeframe=?, metrics=? WHERE id=?",
                        (best["symbol"], best["timeframe"], json.dumps(best["metrics"]), sid),
                    )
            # Un-archive via the lifecycle path (archived -> quick_screen is the only valid move).
            transition_stage(
                strategy_id=sid,
                target_stage="quick_screen",
                reason=f"Recovered from cross-asset re-home (restored to declared {declared}); re-validate under abs()-gap fix",
                actor="ui",
                force=True,
            )
            recovered += 1
            print(f"  recovered {sid} -> {best['symbol']}/{best['timeframe']} (quick_screen)")
        except Exception as exc:
            failed.append((sid, str(exc)))
            print(f"  FAILED {sid}: {exc}")

    print(f"\nRecovered {recovered}/{len(targets)}. Failed: {len(failed)}.")
    if failed:
        for sid, err in failed[:10]:
            print(f"  {sid}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
