from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trade_bot.DEFAULT import (
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_RISK_AI_BETA_TICKERS,
    DEFAULT_RISK_BROAD_EQUITY_TICKERS,
    DEFAULT_RISK_COMMODITY_TICKERS,
    DEFAULT_RISK_CREDIT_TICKERS,
    DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS,
    DEFAULT_RISK_DOLLAR_TICKERS,
    DEFAULT_RISK_DURATION_TICKERS,
    DEFAULT_RISK_ENERGY_TICKERS,
    DEFAULT_RISK_INTERNATIONAL_TICKERS,
    DEFAULT_RISK_SECTOR_TICKERS,
    DEFAULT_STRATEGY_FAMILY_HIGH_BETA_TICKERS,
    DEFAULT_STRATEGY_FAMILY_TBILL_TICKERS,
)
from trade_bot.research.strategy_naming import strategy_display_name

AI_GROWTH_TICKERS = set(DEFAULT_RISK_AI_BETA_TICKERS)
BROAD_US_TICKERS = set(DEFAULT_RISK_BROAD_EQUITY_TICKERS)
SECTOR_TICKERS = set(DEFAULT_RISK_SECTOR_TICKERS)
GLOBAL_TICKERS = set(DEFAULT_RISK_INTERNATIONAL_TICKERS)
DEFENSIVE_FACTOR_TICKERS = set(DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS)
HIGH_BETA_TICKERS = set(DEFAULT_STRATEGY_FAMILY_HIGH_BETA_TICKERS)
TBILL_TICKERS = set(DEFAULT_STRATEGY_FAMILY_TBILL_TICKERS)
DURATION_TICKERS = set(DEFAULT_RISK_DURATION_TICKERS)
CREDIT_TICKERS = set(DEFAULT_RISK_CREDIT_TICKERS)
COMMODITY_TICKERS = (
    set(DEFAULT_RISK_COMMODITY_TICKERS)
    | set(DEFAULT_RISK_ENERGY_TICKERS)
    | set(DEFAULT_RISK_DOLLAR_TICKERS)
)

SCORE_COLUMNS = [
    "promotion_score",
    "cagr",
    "sharpe",
    "max_drawdown",
    "calmar",
    "average_turnover",
    "walk_forward_positive_rate",
    "left_tail_regime_return",
]


def load_experiment_scorecards(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "scorecard.csv"):
        frame.insert(0, "iteration", iteration)
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        for column, default in {
            "phase": "legacy",
            "family": "unknown",
            "parent": "",
            "role": "unknown",
            "scenario_sizing": "",
            "decision_sanity": "",
        }.items():
            if column not in frame.columns:
                frame[column] = default
            else:
                frame[column] = frame[column].fillna(default)
        frame = _ensure_display_name(frame)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["iteration_rank"] = combined.groupby("iteration")["promotion_score"].rank(
        ascending=False,
        method="first",
    )
    return combined.sort_values(["iteration", "iteration_rank"])


def load_experiment_regime_metrics(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "regime_metrics.csv"):
        frame.insert(0, "iteration", iteration)
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_experiment_walk_forward(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "walk_forward_summary.csv"):
        frame.insert(0, "iteration", iteration)
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        elif "strategy" not in frame.columns:
            frame = frame.rename(columns={frame.columns[1]: "strategy"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _ensure_display_name(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "display_name" not in output:
        output["display_name"] = ""
    missing = output["display_name"].isna() | (output["display_name"].astype(str).str.len() == 0)
    if missing.any() and "strategy" in output:
        output.loc[missing, "display_name"] = output.loc[missing].apply(
            lambda row: strategy_display_name(
                str(row.get("strategy", "")),
                family=str(row.get("family", "")),
                phase=str(row.get("phase", "")),
            ),
            axis=1,
        )
    return output


def load_experiment_candidates(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "candidates.csv"):
        frame.insert(0, "iteration", iteration)
        for column, default in {
            "phase": "legacy",
            "family": "unknown",
            "parent": "",
            "role": "unknown",
            "scenario_sizing": "",
            "scenario_sizing_json": "",
            "decision_sanity": "",
            "decision_sanity_json": "",
        }.items():
            if column not in frame.columns:
                frame[column] = default
        frame = _ensure_display_name(frame)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_experiment_history(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty:
        return pd.DataFrame()

    return (
        scorecards.groupby(["iteration", "promotion_decision"], as_index=False)
        .agg(
            candidates=("strategy", "count"),
            best_score=("promotion_score", "max"),
            median_score=("promotion_score", "median"),
            best_calmar=("calmar", "max"),
            best_cagr=("cagr", "max"),
            best_max_drawdown=("max_drawdown", "max"),
        )
        .sort_values(["iteration", "best_score"], ascending=[True, False])
    )


def summarize_experiment_families(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty or "family" not in scorecards.columns:
        return pd.DataFrame()

    return (
        scorecards.groupby(["phase", "family"], as_index=False, dropna=False)
        .agg(
            candidates=("strategy", "count"),
            promoted=(
                "promotion_decision",
                lambda values: int((values == "promote_candidate").sum()),
            ),
            evolved=(
                "promotion_decision",
                lambda values: int((values == "evolve_next_iteration").sum()),
            ),
            best_score=("promotion_score", "max"),
            best_calmar=("calmar", "max"),
            best_cagr=("cagr", "max"),
            best_max_drawdown=("max_drawdown", "max"),
        )
        .sort_values(["best_score", "best_calmar"], ascending=False)
    )


def build_strategy_family_map(
    scorecards: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a semantic map of what each tested strategy is trying to do."""
    if scorecards.empty:
        return pd.DataFrame()

    candidate_index = _candidate_manifest_index(candidates)
    rows: list[dict[str, Any]] = []
    for _, row in scorecards.iterrows():
        candidate = _candidate_for_scorecard(row, candidate_index)
        manifest = _strategy_manifest(row, candidate)
        tickers = _manifest_tickers(manifest)
        satellite_tickers = _clean_tickers(manifest.get("satellite_tickers", []))
        all_tickers = list(dict.fromkeys([*tickers, *satellite_tickers]))
        defensive_ticker = _safe_text(manifest.get("defensive_ticker")).upper()
        strategy_type = _safe_text(manifest.get("type"))
        text = _classification_text(row, candidate, manifest, all_tickers)
        strategy_archetype = _classify_strategy_archetype(
            text,
            strategy_type=strategy_type,
            tickers=all_tickers,
        )
        equity_expression = _classify_equity_expression(
            text,
            strategy_type=strategy_type,
            tickers=all_tickers,
        )
        defensive_expression = _classify_defensive_expression(
            strategy_archetype=strategy_archetype,
            defensive_ticker=defensive_ticker,
            tickers=all_tickers,
        )
        risk_behavior = _classify_risk_behavior(
            text,
            strategy_type=strategy_type,
            archetype=strategy_archetype,
        )
        mapped = {
            "iteration": row.get("iteration"),
            "strategy": _safe_text(row.get("strategy")),
            "display_name": _safe_text(row.get("display_name")),
            "phase": _safe_text(row.get("phase")),
            "raw_family": _safe_text(row.get("family")),
            "role": _safe_text(row.get("role")),
            "parent": _safe_text(row.get("parent")),
            "strategy_archetype": strategy_archetype,
            "equity_expression": equity_expression,
            "defensive_expression": defensive_expression,
            "risk_behavior": risk_behavior,
            "strategy_type": strategy_type or "unknown",
            "defensive_ticker": defensive_ticker or "none",
            "ticker_count": len(all_tickers),
            "primary_tickers": ", ".join(all_tickers[:10]),
            "risk_read": _risk_read(
                archetype=strategy_archetype,
                equity_expression=equity_expression,
                defensive_expression=defensive_expression,
                risk_behavior=risk_behavior,
            ),
            "hypothesis": _first_available_text(row, candidate, "hypothesis"),
        }
        for column in [
            "promotion_decision",
            "scenario_sizing",
            "decision_sanity",
            "promotion_score",
            "cagr",
            "sharpe",
            "max_drawdown",
            "calmar",
            "average_turnover",
            "walk_forward_positive_rate",
            "left_tail_regime_return",
        ]:
            if column in row:
                mapped[column] = row.get(column)
        rows.append(mapped)

    family_map = pd.DataFrame(rows)
    return _with_numeric_scores(family_map).sort_values(
        ["promotion_score", "calmar", "strategy"],
        ascending=[False, False, True],
        na_position="last",
    )


def summarize_strategy_archetypes(family_map: pd.DataFrame) -> pd.DataFrame:
    if family_map.empty:
        return pd.DataFrame()
    return _summarize_cluster(
        family_map,
        ["strategy_archetype"],
        interpretation_column="strategy_archetype",
    )


def summarize_risk_behavior_matrix(family_map: pd.DataFrame) -> pd.DataFrame:
    if family_map.empty:
        return pd.DataFrame()
    return _summarize_cluster(
        family_map,
        ["risk_behavior", "equity_expression", "defensive_expression"],
        interpretation_column="risk_behavior",
    )


def summarize_family_clusters(family_map: pd.DataFrame) -> pd.DataFrame:
    if family_map.empty:
        return pd.DataFrame()
    return _summarize_cluster(
        family_map,
        [
            "strategy_archetype",
            "risk_behavior",
            "equity_expression",
            "defensive_expression",
        ],
        interpretation_column="strategy_archetype",
    )


def strategy_family_takeaways(family_map: pd.DataFrame) -> list[str]:
    if family_map.empty:
        return []

    archetypes = summarize_strategy_archetypes(family_map)
    if archetypes.empty:
        return []

    takeaways: list[str] = []
    top_score = archetypes.sort_values("best_score", ascending=False).iloc[0]
    takeaways.append(
        "Highest promotion-score family is "
        f"{top_score['strategy_archetype']}: best candidate "
        f"{top_score['best_strategy']} scored {_format_number(top_score['best_score'])} "
        f"with median CAGR {_format_percent_value(top_score['median_cagr'])}."
    )

    cagr_rows = archetypes[archetypes["median_cagr"].notna()]
    if not cagr_rows.empty:
        top_cagr = cagr_rows.sort_values("median_cagr", ascending=False).iloc[0]
        takeaways.append(
            "Best median return cluster is "
            f"{top_cagr['strategy_archetype']} at "
            f"{_format_percent_value(top_cagr['median_cagr'])}, with median max drawdown "
            f"{_format_percent_value(top_cagr['median_max_drawdown'])}."
        )

    drawdown_rows = archetypes[archetypes["median_max_drawdown"].notna()]
    if not drawdown_rows.empty:
        safest = drawdown_rows.sort_values("median_max_drawdown", ascending=False).iloc[0]
        takeaways.append(
            "Best median drawdown control is "
            f"{safest['strategy_archetype']} at "
            f"{_format_percent_value(safest['median_max_drawdown'])}; treat this as risk control, "
            "not automatically the best growth engine."
        )

    promoted = family_map[family_map["promotion_decision"] == "promote_candidate"]
    if not promoted.empty:
        concentration = promoted["strategy_archetype"].value_counts(normalize=True).iloc[0]
        leading_archetype = promoted["strategy_archetype"].value_counts().index[0]
        if concentration >= 0.4:
            takeaways.append(
                f"Promotion results are clustered: {leading_archetype} accounts for "
                f"{concentration:.0%} of promoted strategies. That is useful signal, but it raises "
                "correlated-model risk if every monitored strategy expresses the same bet."
            )

    behavior_count = family_map["risk_behavior"].nunique()
    equity_count = family_map["equity_expression"].nunique()
    takeaways.append(
        f"Coverage now spans {behavior_count} risk-behavior patterns and "
        f"{equity_count} equity expressions, so use this map to choose genuinely different "
        "paper candidates instead of several look-alike variants."
    )
    return takeaways


def summarize_experiment_operating_systems(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty:
        return pd.DataFrame()

    candidate_frame = scorecards.copy()
    if "robustness_score" not in candidate_frame:
        candidate_frame["robustness_score"] = float("nan")
    if "walk_forward_positive_rate" not in candidate_frame:
        candidate_frame["walk_forward_positive_rate"] = float("nan")
    if "left_tail_regime_cagr" not in candidate_frame:
        candidate_frame["left_tail_regime_cagr"] = float("nan")
    if "left_tail_regime_return" not in candidate_frame:
        candidate_frame["left_tail_regime_return"] = float("nan")
    robust_rows = candidate_frame[candidate_frame["robustness_score"].notna()]
    if not robust_rows.empty:
        candidate_frame = robust_rows

    sort_columns = ["promotion_score", "robustness_score", "calmar"]
    for column in sort_columns:
        if column not in candidate_frame:
            candidate_frame[column] = float("nan")
    leaders = (
        candidate_frame.sort_values(sort_columns, ascending=False)
        .drop_duplicates("family", keep="first")
        .sort_values(sort_columns, ascending=False)
    )
    columns = [
        "iteration",
        "strategy",
        "phase",
        "family",
        "role",
        "scenario_sizing",
        "promotion_decision",
        "promotion_score",
        "robustness_score",
        "cagr",
        "max_drawdown",
        "calmar",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "left_tail_regime_cagr",
        "hypothesis",
    ]
    return leaders[[column for column in columns if column in leaders.columns]]


def latest_experiment_iteration(scorecards: pd.DataFrame) -> int | None:
    if scorecards.empty or "iteration" not in scorecards:
        return None
    return int(scorecards["iteration"].max())


def _load_iteration_csvs(root: str | Path, filename: str) -> list[tuple[int, pd.DataFrame]]:
    experiment_root = Path(root)
    if not experiment_root.exists():
        return []

    frames = []
    for path in sorted(experiment_root.glob(f"iteration_*/{filename}")):
        frames.append((_iteration_from_path(path), pd.read_csv(path)))
    return frames


def _iteration_from_path(path: Path) -> int:
    name = path.parent.name
    try:
        return int(name.split("_")[-1])
    except (IndexError, ValueError):
        return -1


def _candidate_manifest_index(
    candidates: pd.DataFrame | None,
) -> tuple[dict[tuple[int | None, str], pd.Series], dict[str, pd.Series]]:
    by_key: dict[tuple[int | None, str], pd.Series] = {}
    by_strategy: dict[str, pd.Series] = {}
    if candidates is None or candidates.empty or "strategy" not in candidates:
        return by_key, by_strategy
    for _, row in candidates.iterrows():
        strategy = _safe_text(row.get("strategy"))
        if not strategy:
            continue
        iteration = _safe_int(row.get("iteration"))
        by_key[(iteration, strategy)] = row
        by_strategy[strategy] = row
    return by_key, by_strategy


def _candidate_for_scorecard(
    scorecard_row: pd.Series,
    candidate_index: tuple[dict[tuple[int | None, str], pd.Series], dict[str, pd.Series]],
) -> pd.Series | None:
    by_key, by_strategy = candidate_index
    strategy = _safe_text(scorecard_row.get("strategy"))
    iteration = _safe_int(scorecard_row.get("iteration"))
    candidate = by_key.get((iteration, strategy))
    if candidate is not None:
        return candidate
    return by_strategy.get(strategy)


def _strategy_manifest(row: pd.Series, candidate: pd.Series | None) -> dict[str, Any]:
    raw_values = []
    if candidate is not None:
        raw_values.append(candidate.get("strategy_json"))
    raw_values.append(row.get("strategy_json"))
    for raw in raw_values:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _manifest_tickers(manifest: dict[str, Any]) -> list[str]:
    tickers = _clean_tickers(manifest.get("tickers", []))
    allocation_weights = manifest.get("allocation_weights")
    if isinstance(allocation_weights, dict):
        tickers.extend(_clean_tickers(allocation_weights.keys()))
    return list(dict.fromkeys(tickers))


def _clean_tickers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            return []
    return [
        str(value).upper().strip()
        for value in values
        if str(value).strip() and str(value).strip().lower() != "nan"
    ]


def _classification_text(
    row: pd.Series,
    candidate: pd.Series | None,
    manifest: dict[str, Any],
    tickers: list[str],
) -> str:
    pieces = [
        _safe_text(row.get("strategy")),
        _safe_text(row.get("phase")),
        _safe_text(row.get("family")),
        _safe_text(row.get("role")),
        _safe_text(row.get("parent")),
        _safe_text(row.get("hypothesis")),
        _safe_text(row.get("scenario_sizing")),
        _safe_text(row.get("decision_sanity")),
        _safe_text(manifest.get("type")),
        " ".join(tickers),
    ]
    if candidate is not None:
        pieces.extend(
            [
                _safe_text(candidate.get("phase")),
                _safe_text(candidate.get("family")),
                _safe_text(candidate.get("role")),
                _safe_text(candidate.get("parent")),
                _safe_text(candidate.get("hypothesis")),
                _safe_text(candidate.get("scenario_sizing")),
                _safe_text(candidate.get("decision_sanity")),
            ]
        )
    return " ".join(piece for piece in pieces if piece).lower()


def _classify_strategy_archetype(
    text: str,
    *,
    strategy_type: str,
    tickers: list[str],
) -> str:
    ticker_set = set(tickers)
    if _has_any(text, ["buy_hold", "static", "reference", "60_40"]):
        return "Static baseline / reference"
    if strategy_type == "sector_regime_rotation" or "sector_regime" in text:
        if ticker_set & GLOBAL_TICKERS:
            return "Global / cross-asset rotation"
        return "Sector and factor rotation"
    if strategy_type == "ai_risk_cycle_overlay" or "risk_cycle" in text or "cycle_" in text:
        return "AI risk-cycle reentry"
    if _has_any(text, ["dip", "reentry", "cheap", "washout"]):
        return "Dip reentry / buy-the-dip"
    if _has_any(text, ["credit", "rates", "duration"]) or len(ticker_set & CREDIT_TICKERS) >= 3:
        return "Credit / rates defense"
    if _has_any(text, ["oil", "hormuz", "commodity", "reflation"]) or ticker_set & {
        "DBC",
        "USO",
        "BNO",
        "XLE",
        "XOP",
    }:
        return "Macro shock / commodities"
    if ticker_set & GLOBAL_TICKERS:
        return "Global / cross-asset rotation"
    if _has_any(text, ["low_vol", "defensive", "quality", "dividend"]) or ticker_set & (
        DEFENSIVE_FACTOR_TICKERS | {"XLU", "XLP", "XLV"}
    ):
        return "Defensive equity / low-vol"
    if _has_any(text, ["high_beta", "speculative", "liquidity"]) or ticker_set & HIGH_BETA_TICKERS:
        return "High-beta / liquidity"
    if (
        _has_any(text, ["ai", "bubble", "semis", "mega_cap"])
        or len(ticker_set & AI_GROWTH_TICKERS) >= 3
    ):
        return "AI growth / bubble escape"
    return "General momentum / off-ramp"


def _classify_equity_expression(
    text: str,
    *,
    strategy_type: str,
    tickers: list[str],
) -> str:
    ticker_set = set(tickers)
    if _has_any(text, ["reference", "60_40", "static"]):
        return "Static/reference mix"
    if strategy_type == "sector_regime_rotation" or len(ticker_set & SECTOR_TICKERS) >= 4:
        return "Sector/theme equities"
    if ticker_set & GLOBAL_TICKERS:
        return "Global equities"
    if ticker_set & HIGH_BETA_TICKERS:
        return "High-beta/speculative"
    if len(ticker_set & AI_GROWTH_TICKERS) >= 3 or _has_any(text, ["ai", "semis"]):
        return "AI / growth equities"
    if ticker_set & DEFENSIVE_FACTOR_TICKERS:
        return "Defensive equity/factors"
    if ticker_set & BROAD_US_TICKERS:
        return "Broad US equities"
    if ticker_set & CREDIT_TICKERS:
        return "Credit/rates-led"
    return "Mixed/unclear equity sleeve"


def _classify_defensive_expression(
    *,
    strategy_archetype: str,
    defensive_ticker: str,
    tickers: list[str],
) -> str:
    ticker_set = set(tickers)
    if strategy_archetype == "Static baseline / reference":
        return "Static policy mix"
    defensive_groups = {
        "tbill": bool(defensive_ticker in TBILL_TICKERS or ticker_set & TBILL_TICKERS),
        "duration": bool(defensive_ticker in DURATION_TICKERS or ticker_set & DURATION_TICKERS),
        "credit": bool(ticker_set & CREDIT_TICKERS),
        "commodity": bool(ticker_set & COMMODITY_TICKERS),
    }
    group_count = sum(defensive_groups.values())
    if group_count >= 3:
        return "Multi-asset defense"
    if defensive_groups["tbill"]:
        return "T-bills/cash"
    if defensive_groups["duration"]:
        return "Treasuries/duration"
    if defensive_groups["credit"]:
        return "Credit as defensive bridge"
    if defensive_groups["commodity"]:
        return "Gold/commodities/dollar"
    return "Little/no defensive sleeve"


def _classify_risk_behavior(
    text: str,
    *,
    strategy_type: str,
    archetype: str,
) -> str:
    if archetype == "Static baseline / reference":
        return "Static benchmark"
    if strategy_type == "sector_regime_rotation" or "sector_regime" in text:
        return "Sector-regime gating"
    if _has_any(text, ["cooldown", "hysteresis", "whipsaw", "low_churn"]):
        return "Cooldown/hysteresis"
    if _has_any(text, ["dip", "reentry", "cheap", "washout"]) or strategy_type in {
        "ai_risk_cycle_overlay",
        "dip_reentry",
    }:
        return "Dip-reentry"
    if _has_any(text, ["credit", "rates", "duration"]):
        return "Credit/rates gate"
    if _has_any(text, ["scenario", "fragile_ai", "balanced", "defensive"]):
        return "Scenario-sized momentum"
    if _has_any(text, ["high_beta", "liquidity", "speculative"]):
        return "Risk-on liquidity probe"
    if _has_any(text, ["low_vol", "defensive", "quality", "dividend"]):
        return "Defensive ballast"
    return "Trend/off-ramp"


def _summarize_cluster(
    family_map: pd.DataFrame,
    group_columns: list[str],
    *,
    interpretation_column: str,
) -> pd.DataFrame:
    frame = _with_numeric_scores(family_map)
    for column in group_columns:
        if column not in frame:
            return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        best_row = _best_cluster_row(group)
        output = dict(zip(group_columns, key, strict=True))
        output.update(
            {
                "candidates": int(len(group)),
                "promoted": int((group["promotion_decision"] == "promote_candidate").sum()),
                "evolved": int((group["promotion_decision"] == "evolve_next_iteration").sum()),
                "rejected": int(
                    group["promotion_decision"].astype(str).str.startswith("reject").sum()
                ),
                "best_strategy": _safe_text(best_row.get("strategy")),
                "best_score": _numeric_or_nan(best_row.get("promotion_score")),
                "median_score": _numeric_or_nan(group["promotion_score"].median()),
                "best_cagr": _numeric_or_nan(group["cagr"].max()),
                "median_cagr": _numeric_or_nan(group["cagr"].median()),
                "best_calmar": _numeric_or_nan(group["calmar"].max()),
                "median_calmar": _numeric_or_nan(group["calmar"].median()),
                "best_max_drawdown": _numeric_or_nan(group["max_drawdown"].max()),
                "median_max_drawdown": _numeric_or_nan(group["max_drawdown"].median()),
                "median_walk_forward_positive_rate": _numeric_or_nan(
                    group["walk_forward_positive_rate"].median()
                ),
                "median_left_tail_regime_return": _numeric_or_nan(
                    group["left_tail_regime_return"].median()
                ),
                "median_turnover": _numeric_or_nan(group["average_turnover"].median()),
                "interpretation": _cluster_interpretation(
                    _safe_text(output.get(interpretation_column))
                ),
            }
        )
        rows.append(output)

    return pd.DataFrame(rows).sort_values(
        ["best_score", "median_score", "best_calmar"],
        ascending=False,
        na_position="last",
    )


def _best_cluster_row(group: pd.DataFrame) -> pd.Series:
    if "promotion_score" not in group or group["promotion_score"].isna().all():
        return group.iloc[0]
    return group.loc[group["promotion_score"].idxmax()]


def _with_numeric_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in SCORE_COLUMNS:
        if column not in output:
            output[column] = pd.NA
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if "promotion_decision" not in output:
        output["promotion_decision"] = ""
    return output


def _risk_read(
    *,
    archetype: str,
    equity_expression: str,
    defensive_expression: str,
    risk_behavior: str,
) -> str:
    return (
        f"{archetype}: expresses risk through {equity_expression}; cuts or parks risk via "
        f"{defensive_expression}; operating behavior is {risk_behavior.lower()}."
    )


def _cluster_interpretation(value: str) -> str:
    interpretations = {
        "AI growth / bubble escape": (
            "Targets AI/growth leadership while trying to exit when trend, volatility, or scenario "
            "pressure says the leadership trade is breaking."
        ),
        "AI risk-cycle reentry": (
            "Combines AI/growth upside with explicit re-risking logic after drawdowns, so it is "
            "designed to avoid getting stuck in cash."
        ),
        "Dip reentry / buy-the-dip": (
            "Adds risk when drawdown, recovery, breadth, or credit conditions suggest a washout is "
            "becoming tradable rather than just falling."
        ),
        "Sector and factor rotation": (
            "Rotates among sectors, factors, and defensive sleeves; useful for seeing whether the "
            "model is doing more than a cash versus QQQ decision."
        ),
        "Credit / rates defense": (
            "Uses credit, rates, and duration as the market-condition read, often as an intermediate "
            "step between full risk-on and T-bills."
        ),
        "Macro shock / commodities": (
            "Tests whether commodity, oil, dollar, or geopolitical shock sleeves improve transition "
            "handling."
        ),
        "Global / cross-asset rotation": (
            "Expands the equity decision outside narrow US mega-cap leadership and lets global or "
            "cross-asset strength compete."
        ),
        "Defensive equity / low-vol": (
            "Keeps equity exposure but tilts toward quality, dividends, low volatility, or defensive "
            "sectors."
        ),
        "High-beta / liquidity": (
            "Probes whether speculative rebound assets add enough upside to justify capped exposure."
        ),
        "Static baseline / reference": (
            "Reference portfolio, not an active timing model; use it as the hurdle and sanity check."
        ),
        "General momentum / off-ramp": (
            "Momentum model with an exit route, but without a more specific thematic or regime sleeve."
        ),
        "Static benchmark": "Reference behavior with no active risk timing.",
        "Trend/off-ramp": "Moves with trend evidence and exits when the trend breaks.",
        "Scenario-sized momentum": "Scales active risk based on scenario and market-pressure layers.",
        "Cooldown/hysteresis": "Requires larger or more persistent changes before trading.",
        "Dip-reentry": "Tries to add risk after washouts only when confirmation improves.",
        "Sector-regime gating": "Allows sector rotation only when broader regime gates permit it.",
        "Credit/rates gate": "Uses credit and rates confirmation before changing risk.",
        "Risk-on liquidity probe": "Adds capped high-beta exposure when liquidity/rebound evidence is strong.",
        "Defensive ballast": "Keeps risk lower through defensive equity and bond-like sleeves.",
    }
    return interpretations.get(value, "Grouped by similar objective and portfolio behavior.")


def _first_available_text(row: pd.Series, candidate: pd.Series | None, column: str) -> str:
    for source in [row, candidate]:
        if source is None:
            continue
        value = _safe_text(source.get(column))
        if value:
            return value
    return ""


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if value is pd.NA:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _format_percent_value(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"
