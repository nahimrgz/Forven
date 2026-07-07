from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable

from forven.db import get_db


DEFAULT_RECENT_LIMIT = 80
DEFAULT_SATURATION_THRESHOLD = 0.35
DEFAULT_HARD_SATURATION_THRESHOLD = 0.55
DEFAULT_OUTCOME_WINDOW_DAYS = 90
DEAD_FAMILY_MIN_ATTEMPTS = 8

# Mirrors the "Promising" bucket of api_core.calculate_backtest_verdict
# (forven/api_core.py ~line 5076: `total_trades >= 15 and sharpe >= 1.0 and
# profit_factor >= 1.3 and max_dd < 35`). Duplicated as constants rather than
# imported because api_core pulls in the whole API/router module graph — not
# worth that import cost for four threshold numbers. Keep these in sync by hand
# if that verdict bucket ever changes.
NEAR_MISS_MIN_SHARPE = 1.0
NEAR_MISS_MIN_PROFIT_FACTOR = 1.3
NEAR_MISS_MIN_TRADES = 15
NEAR_MISS_MAX_DRAWDOWN_PCT = 35.0  # percent points, e.g. 35 == 35%
DEFAULT_NEAR_MISS_WINDOW_DAYS = 90
DEFAULT_NEAR_MISS_LIMIT = 6

FAMILY_LABELS = {
    "rsi": "RSI / oscillator momentum",
    "stochastic": "stochastic oscillator",
    "williams_r": "Williams %R oscillator",
    "macd": "MACD momentum",
    "ema": "EMA trend",
    "bollinger": "Bollinger / band mean reversion",
    "donchian": "Donchian breakout",
    "keltner": "Keltner channel",
    "vwap": "VWAP execution/mean reversion",
    "supertrend": "Supertrend",
    "adx": "ADX trend strength",
    "orb": "opening range breakout",
    "funding": "funding/carry",
    "volume": "volume/order-flow",
    "cross_asset": "cross-asset/relative value",
    "other": "other",
}

FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rsi", ("rsi", "connors")),
    ("stochastic", ("stochastic", "stoch", "kdj")),
    ("williams_r", ("williams_r", "williams-r", "williams %r", "williams")),
    ("macd", ("macd", "ppo", "trix")),
    ("ema", ("ema", "dema", "tema", "moving_average", "moving average")),
    ("bollinger", ("bollinger", "bb_", "band_reversion", "mean_reversion", "zscore")),
    ("donchian", ("donchian",)),
    ("keltner", ("keltner",)),
    ("vwap", ("vwap",)),
    ("supertrend", ("supertrend",)),
    ("adx", ("adx", "aroon")),
    ("orb", ("orb", "opening_range", "opening range")),
    ("funding", ("funding", "basis", "carry", "perp")),
    ("volume", ("volume", "obv", "mfi", "chaikin", "adl", "taker", "liquidation")),
    ("cross_asset", ("cross_asset", "cross-asset", "dominance", "relative_value", "relative value", "rotation")),
)

def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _flatten_payload(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return str(value)


def infer_strategy_family(*values: Any) -> str:
    text = " ".join(_flatten_payload(value) for value in values)
    normalized = re.sub(r"[^a-z0-9_% -]+", "_", text.lower())
    for family, patterns in FAMILY_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return family
    return "other"


def _row_family(row: Any) -> str:
    if hasattr(row, "get"):
        getter = row.get
    else:
        keys = set(row.keys())

        def getter(key: str, default: Any = None) -> Any:
            return row[key] if key in keys else default

    return infer_strategy_family(
        getter("type"),
        getter("runtime_type"),
        getter("name"),
        getter("display_id"),
        getter("id"),
        getter("params"),
        getter("metrics"),
        getter("notes"),
    )


def recent_strategy_family_counts(limit: int = DEFAULT_RECENT_LIMIT) -> dict[str, Any]:
    normalized_limit = max(1, min(int(limit or DEFAULT_RECENT_LIMIT), 500))
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT id, display_id, name, type, runtime_type, params, metrics, notes, created_at, updated_at
                FROM strategies
                ORDER BY datetime(COALESCE(updated_at, created_at, '1970-01-01T00:00:00+00:00')) DESC, id DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()
    except Exception:
        return {"total": 0, "counts": {}, "shares": {}, "top_family": None}

    counts = Counter(_row_family(row) for row in rows)
    total = sum(counts.values())
    shares = {family: count / total for family, count in counts.items()} if total else {}
    top_family = counts.most_common(1)[0][0] if counts else None
    return {
        "total": total,
        "counts": dict(counts),
        "shares": shares,
        "top_family": top_family,
    }


def saturated_strategy_families(
    *,
    limit: int = DEFAULT_RECENT_LIMIT,
    threshold: float = DEFAULT_SATURATION_THRESHOLD,
) -> list[dict[str, Any]]:
    stats = recent_strategy_family_counts(limit=limit)
    total = int(stats.get("total") or 0)
    if total <= 0:
        return []
    counts = stats.get("counts") if isinstance(stats.get("counts"), dict) else {}
    shares = stats.get("shares") if isinstance(stats.get("shares"), dict) else {}
    saturated: list[dict[str, Any]] = []
    for family, count in counts.items():
        share = float(shares.get(family) or 0.0)
        if share >= threshold:
            saturated.append(
                {
                    "family": family,
                    "label": FAMILY_LABELS.get(family, family.replace("_", " ")),
                    "count": int(count),
                    "share": share,
                    "total": total,
                    "severity": "hard" if share >= DEFAULT_HARD_SATURATION_THRESHOLD else "soft",
                }
            )
    saturated.sort(key=lambda item: (item["share"], item["count"]), reverse=True)
    return saturated


def family_outcome_stats(days: int = DEFAULT_OUTCOME_WINDOW_DAYS) -> dict[str, dict[str, int]]:
    """Per-family generation outcomes over a recent window.

    attempts = strategies created in the window; survivors = those that reached
    the paper stage (or beyond) at least once. This is the survivor signal that
    lets the diversity guard steer by OUTCOME (dead vs live regions of the
    search space), not just by generation frequency.
    """
    window = -abs(int(days or DEFAULT_OUTCOME_WINDOW_DAYS))
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.display_id, s.name, s.type, s.runtime_type,
                       s.params, s.metrics, s.notes,
                       MAX(CASE WHEN e.to_state IN ('paper', 'live_graduated')
                                  OR s.stage IN ('paper', 'live_graduated')
                            THEN 1 ELSE 0 END) AS survived
                FROM strategies s
                LEFT JOIN strategy_events e ON e.strategy_id = s.id
                WHERE datetime(COALESCE(s.created_at, '1970-01-01T00:00:00+00:00'))
                      > datetime('now', ? || ' days')
                GROUP BY s.id
                """,
                (str(window),),
            ).fetchall()
    except Exception:
        return {}

    stats: dict[str, dict[str, int]] = {}
    for row in rows:
        family = _row_family(row)
        entry = stats.setdefault(family, {"attempts": 0, "survivors": 0})
        entry["attempts"] += 1
        if row["survived"]:
            entry["survivors"] += 1
    return stats


def render_strategy_diversity_guard(
    *,
    task_description: str = "",
    limit: int = DEFAULT_RECENT_LIMIT,
    threshold: float = DEFAULT_SATURATION_THRESHOLD,
    outcome_window_days: int = DEFAULT_OUTCOME_WINDOW_DAYS,
) -> str:
    saturated = saturated_strategy_families(limit=limit, threshold=threshold)
    outcomes = family_outcome_stats(days=outcome_window_days)
    dead = sorted(
        (
            (family, stats)
            for family, stats in outcomes.items()
            if family != "other"
            and stats["attempts"] >= DEAD_FAMILY_MIN_ATTEMPTS
            and stats["survivors"] == 0
        ),
        key=lambda item: item[1]["attempts"],
        reverse=True,
    )
    alive = sorted(
        ((family, stats) for family, stats in outcomes.items() if stats["survivors"] > 0),
        key=lambda item: item[1]["survivors"],
        reverse=True,
    )
    if not saturated and not dead and not alive:
        return ""

    lines = ["# STRATEGY DIVERSITY GUARD"]
    if saturated:
        lines.append(
            "Recent strategy memory is family-skewed. Treat saturated families as overrepresented prior art, not inspiration."
        )
        for item in saturated[:4]:
            pct = round(float(item["share"]) * 100)
            lines.append(f"- {item['label']}: {item['count']}/{item['total']} recent strategies ({pct}%).")

        # Family-agnostic guidance: steer away from whichever families are saturated on
        # THIS instance. (The old RSI-specific carve-out was a fossil from a past RSI
        # flood and unfairly singled out one family — removed so every family is treated
        # the same, driven purely by this instance's own saturation.)
        labels = [str(item["label"]) for item in saturated[:3]]
        lines.append("- Prefer families outside the saturated set: " + ", ".join(labels) + ".")

    # Outcome steering: frequency alone can't distinguish an over-mined dead
    # region from a productive one, so surface where recent attempts actually
    # went (reached paper) vs where the pipeline keeps rejecting everything.
    if dead:
        lines.append(
            f"Proven-dead regions (last {int(outcome_window_days)}d, ≥{DEAD_FAMILY_MIN_ATTEMPTS} attempts, zero reached paper):"
        )
        for family, stats in dead[:4]:
            label = FAMILY_LABELS.get(family, family.replace("_", " "))
            lines.append(
                f"- {label}: {stats['attempts']} candidates, 0 survivors. "
                "Do not propose more of these without a structurally different mechanism."
            )
    if alive:
        parts = [
            f"{FAMILY_LABELS.get(family, family.replace('_', ' '))} ({stats['survivors']}/{stats['attempts']} reached paper)"
            for family, stats in alive[:4]
        ]
        lines.append("Families with recent survivors (evidence of a live region): " + ", ".join(parts) + ".")

    if _normalize_text(task_description):
        lines.append(f"- Apply this guard while working on: {_normalize_text(task_description)[:240]}")

    return "\n".join(lines)


def render_failure_taxonomy(*, days: int = 30, limit: int = 8) -> str:
    """Render the structured gate-rejection taxonomy for generation steering.

    Surfaces the top (family × gate × reason × regime) rejection clusters from
    `gate_rejections` so ideation designs away from regions the pipeline has
    already rejected instead of re-mining them.
    """
    try:
        from forven.db import query_failure_taxonomy

        rows = query_failure_taxonomy(days=days)
    except Exception:
        return ""
    if not rows:
        return ""

    lines = [
        f"# FAILURE TAXONOMY (last {int(days)}d)",
        "Top structured rejection patterns from the promotion gates. Treat each as a disproven region: "
        "do not re-propose the same family/mechanism into the same failure mode without explicitly addressing it.",
    ]
    for row in rows[:limit]:
        gate = _normalize_text(row.get("gate")) or "?"
        reason = _normalize_text(row.get("reason_code")) or "unspecified"
        strategy_type = _normalize_text(row.get("strategy_type")) or "unknown family"
        regime = _normalize_text(row.get("regime_context")) or "any regime"
        try:
            count = int(row.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        example_ids = [
            sid.strip()
            for sid in _normalize_text(row.get("strategy_ids")).split(",")
            if sid.strip()
        ][:3]
        example_suffix = f" [e.g. {', '.join(example_ids)}]" if example_ids else ""
        lines.append(f"- {strategy_type} @ {gate}: {reason} ×{count} ({regime}){example_suffix}")
    return "\n".join(lines)


def _metrics_snapshot_section(snapshot: dict[str, Any], *keys: str) -> dict[str, Any]:
    """First present nested section of a metrics_snapshot blob, unwrapped.

    Mirrors the in_sample/out_of_sample convention used elsewhere for backtest
    metric blobs (see forven/policy.py `_metrics_section`), so a near-miss digest
    built from `gate_rejections.metrics_snapshot` reads OOS numbers the same way
    the promotion gates do instead of drifting to a second, incompatible shape.
    """
    if not isinstance(snapshot, dict):
        return {}
    for key in keys:
        section = snapshot.get(key)
        if isinstance(section, dict) and section:
            return section
    return {}


def _extract_promising_metrics(snapshot: Any) -> dict[str, float] | None:
    """Pull sharpe/profit_factor/total_trades/max_drawdown_pct out of a
    metrics_snapshot blob (JSON text or already-decoded dict).

    Every observed gate_rejections row today is flat (see forven/policy.py
    `_load_metrics_snapshot_for_rejection`), but other backtest metric blobs in
    this codebase nest an OOS leg under "out_of_sample" or "oos" (forven/policy.py
    `_metrics_section`). Preferring that nested leg when present keeps this in
    sync with those blobs' shape without assuming gate_rejections never grows one.
    Returns None for anything that isn't a parseable metrics dict.
    """
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except (TypeError, ValueError):
            return None
    if not isinstance(snapshot, dict) or not snapshot:
        return None

    oos = _metrics_snapshot_section(snapshot, "out_of_sample", "oos")
    merged = dict(snapshot)
    merged.update({k: v for k, v in oos.items() if v is not None})

    try:
        sharpe = float(merged.get("sharpe", merged.get("sharpe_ratio")) or 0.0)
        profit_factor = float(merged.get("profit_factor") or 0.0)
        total_trades = int(float(merged.get("total_trades") or 0))
        max_dd_raw = merged.get("max_drawdown_pct", merged.get("max_drawdown"))
        max_dd = float(max_dd_raw) if max_dd_raw is not None else 0.0
    except (TypeError, ValueError):
        return None

    # metrics_snapshot stores drawdown as a 0..1 ratio in every observed row;
    # calculate_backtest_verdict's threshold is in percent points (35 == 35%).
    # Only scale values that look like ratios so a future percent-points writer
    # isn't silently multiplied again.
    if abs(max_dd) <= 1.0:
        max_dd *= 100.0

    return {
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "total_trades": float(total_trades),
        "max_drawdown_pct": max_dd,
    }


def _is_near_miss(metrics: dict[str, float]) -> bool:
    return (
        metrics["sharpe"] >= NEAR_MISS_MIN_SHARPE
        and metrics["profit_factor"] >= NEAR_MISS_MIN_PROFIT_FACTOR
        and metrics["total_trades"] >= NEAR_MISS_MIN_TRADES
        and metrics["max_drawdown_pct"] < NEAR_MISS_MAX_DRAWDOWN_PCT
    )


def render_near_miss_digest(
    *,
    days: int = DEFAULT_NEAR_MISS_WINDOW_DAYS,
    limit: int = DEFAULT_NEAR_MISS_LIMIT,
) -> str:
    """Render gate-rejected strategies whose metrics cleared the 'Promising' bar.

    `render_failure_taxonomy` tells generation "this region is disproven" — but
    some rejections (overfit_reject on the IS leg, a sample-size floor) fire on
    strategies whose actual performance looked genuinely good. Without this,
    those get lumped in with true dead ends and the generator never learns to
    explore NEAR them instead of away from them. Best-effort: any failure
    (missing table, malformed JSON) yields "".
    """
    try:
        from forven.db import query_near_miss_rejections

        rows = query_near_miss_rejections(days=days, limit=limit)
    except Exception:
        return ""
    if not rows:
        return ""

    normalized_limit = max(1, int(limit or DEFAULT_NEAR_MISS_LIMIT))
    seen_strategy_ids: set[str] = set()
    entries: list[str] = []
    for row in rows:
        strategy_id = _normalize_text(row.get("strategy_id"))
        if not strategy_id or strategy_id in seen_strategy_ids:
            continue
        metrics = _extract_promising_metrics(row.get("metrics_snapshot"))
        if not metrics or not _is_near_miss(metrics):
            continue
        seen_strategy_ids.add(strategy_id)
        strategy_type = _normalize_text(row.get("strategy_type")) or "unknown family"
        gate = _normalize_text(row.get("gate")) or "?"
        reason = _normalize_text(row.get("reason_code")) or "unspecified"
        entries.append(
            f"- {strategy_id} ({strategy_type}): Sharpe {metrics['sharpe']:.2f}, "
            f"PF {metrics['profit_factor']:.2f}, {int(metrics['total_trades'])} trades, "
            f"DD {metrics['max_drawdown_pct']:.1f}% — died on {gate}/{reason}. "
            "Explore near this, don't abandon the mechanism."
        )
        if len(entries) >= normalized_limit:
            break

    if not entries:
        return ""

    lines = [
        f"# NEAR-MISS DIGEST (last {int(days)}d)",
        "These rejections cleared the same 'Promising' performance bar as "
        "calculate_backtest_verdict (Sharpe/profit-factor/sample-size/drawdown) "
        "but still died on a structural gate. Treat the neighborhood as a live "
        "region worth mutating, not a disproven one — address the specific "
        "failure reason below rather than walking away from the mechanism.",
    ]
    lines.extend(entries)
    return "\n".join(lines)


def filter_recall_records_for_diversity(records: Iterable[dict[str, Any]], *, max_family_share: float = 0.4) -> list[dict[str, Any]]:
    """Limit overrepresented families in retrieved examples.

    This is intentionally generic: callers can pass Chroma flattened records and
    get back a list where no family dominates the examples shown to an agent.
    """
    output: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        family = infer_strategy_family(record.get("document"), metadata)
        projected_total = len(output) + 1
        projected_share = (family_counts[family] + 1) / projected_total
        if projected_total > 3 and projected_share > max_family_share:
            continue
        output.append(record)
        family_counts[family] += 1
    return output
