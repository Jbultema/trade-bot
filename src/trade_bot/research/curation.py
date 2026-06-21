from __future__ import annotations

import pandas as pd

DEFAULT_CURATED_SHELF_LIMIT = 25
PRUNED_STATUS = "pruned_dead_end"
REFERENCE_STATUS = "reference"
OPERATIONAL_STATUS = "operational_candidate"
ITERATE_STATUS = "needs_iteration"
ARCHIVE_STATUS = "research_archive"

RISK_REJECT_DECISIONS = {
    "reject_left_tail",
    "reject_regime_fragility",
    "reject_walk_forward_fragility",
}


def rank_strategy_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable research ranking before explicit curation/diversification."""
    if frame.empty:
        return frame.copy()
    ranked = add_research_status(frame)
    if "name" in ranked and "strategy" not in ranked:
        ranked = ranked.rename(columns={"name": "strategy"})

    for column in [
        "selection_adjusted_promotion_score",
        "promotion_score",
        "robustness_score",
        "calmar",
        "cagr",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "average_turnover",
        "iteration",
    ]:
        if column not in ranked:
            ranked[column] = float("nan")
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")

    if "promotion_decision" not in ranked:
        ranked["promotion_decision"] = ""
    if "validation_tier" not in ranked:
        ranked["validation_tier"] = ""

    ranked["_promotion_rank"] = (
        ranked["promotion_decision"]
        .astype(str)
        .map(
            {
                "promote_candidate": 0,
                "evolve_next_iteration": 1,
                "reject_or_hold_for_reference": 2,
                "reject_walk_forward_fragility": 3,
                "reject_regime_fragility": 4,
                "reject_left_tail": 5,
            }
        )
        .fillna(2)
    )
    ranked["_validation_rank"] = (
        ranked["validation_tier"]
        .astype(str)
        .map(
            {
                "paper_champion_candidate": 0,
                "paper_challenger_candidate": 1,
                "needs_more_holdout_evidence": 2,
                "reject_or_redesign": 3,
            }
        )
        .fillna(2)
    )
    ranked["_curation_sort_score"] = (
        ranked["selection_adjusted_promotion_score"]
        .fillna(ranked["promotion_score"])
        .fillna(ranked["robustness_score"])
        .fillna(ranked["calmar"])
    )
    ranked["_research_status_rank"] = (
        ranked["research_status"]
        .astype(str)
        .map(
            {
                OPERATIONAL_STATUS: 0,
                ITERATE_STATUS: 1,
                REFERENCE_STATUS: 2,
                ARCHIVE_STATUS: 3,
                PRUNED_STATUS: 4,
            }
        )
        .fillna(3)
    )
    ranked = ranked.sort_values(
        [
            "_research_status_rank",
            "_promotion_rank",
            "_validation_rank",
            "_curation_sort_score",
            "robustness_score",
            "calmar",
            "walk_forward_positive_rate",
            "left_tail_regime_return",
            "iteration",
        ],
        ascending=[True, True, True, False, False, False, False, False, False],
        na_position="last",
    )
    return ranked.drop(
        columns=["_promotion_rank", "_validation_rank", "_research_status_rank"],
        errors="ignore",
    )


def add_research_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify experiment rows into live candidates, archive rows, and dead-end prunes.

    Pruning here is an interface and curation decision, not deletion. Historical rows remain
    available for audit, but default views should not keep low-return or failed-risk ideas in
    the active operating queue.
    """
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    if "name" in output and "strategy" not in output:
        output = output.rename(columns={"name": "strategy"})
    for column in [
        "cagr",
        "calmar",
        "max_drawdown",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "promotion_score",
    ]:
        if column not in output:
            output[column] = float("nan")
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column, default in {
        "promotion_decision": "",
        "phase": "",
        "family": "",
        "role": "",
        "strategy": "",
        "operability_label": "",
    }.items():
        if column not in output:
            output[column] = default
        output[column] = output[column].fillna(default)

    statuses = []
    reasons = []
    for _, row in output.iterrows():
        status, reason = _research_status_for_row(row)
        statuses.append(status)
        reasons.append(reason)
    output["research_status"] = statuses
    output["prune_reason"] = reasons
    return output


def _research_status_for_row(row: pd.Series) -> tuple[str, str]:
    if bool(_reference_mask(pd.DataFrame([row])).iloc[0]):
        return REFERENCE_STATUS, "reference_or_baseline"

    decision = str(row.get("promotion_decision", ""))
    phase = str(row.get("phase", "")).lower()
    family = str(row.get("family", "")).lower()
    role = str(row.get("role", "")).lower()
    strategy = str(row.get("strategy", "")).lower()
    cagr = _numeric(row.get("cagr"))
    calmar = _numeric(row.get("calmar"))
    max_drawdown = _numeric(row.get("max_drawdown"))
    walk_forward_positive = _numeric(row.get("walk_forward_positive_rate"))
    left_tail_return = _numeric(row.get("left_tail_regime_return"))
    operability = str(row.get("operability_label", ""))

    if decision in RISK_REJECT_DECISIONS:
        return PRUNED_STATUS, f"{decision}_failed_validation"
    if cagr == cagr and cagr < 0.05:
        return PRUNED_STATUS, "low_cagr_below_5pct"
    if calmar == calmar and calmar < 0.25 and cagr == cagr and cagr < 0.08:
        return PRUNED_STATUS, "weak_return_and_risk_adjusted_profile"
    if max_drawdown == max_drawdown and max_drawdown < -0.25:
        return PRUNED_STATUS, "drawdown_worse_than_25pct"
    if "classic_dd" in strategy and cagr == cagr and cagr < 0.135:
        return PRUNED_STATUS, "reactive_drawdown_control_lost_too_much_growth"
    if "sklearn_future_state" in phase and cagr == cagr and cagr < 0.08:
        return PRUNED_STATUS, "low_return_future_state_ml_probe"
    if "future_state" in phase and cagr == cagr and cagr < 0.08:
        return PRUNED_STATUS, "low_return_future_state_probe"
    if walk_forward_positive == walk_forward_positive and walk_forward_positive < 0.55:
        return PRUNED_STATUS, "weak_walk_forward_positive_rate"
    if left_tail_return == left_tail_return and left_tail_return < -0.18:
        return PRUNED_STATUS, "left_tail_regime_loss_too_large"

    promoted = decision in {"promote_candidate", "evolve_next_iteration"}
    high_growth = cagr == cagr and cagr >= 0.10
    good_risk_adjusted = calmar == calmar and calmar >= 0.55
    tolerable_tail = max_drawdown != max_drawdown or max_drawdown >= -0.24
    human_operable = operability != "too_twitchy"
    if promoted and high_growth and good_risk_adjusted and tolerable_tail and human_operable:
        return OPERATIONAL_STATUS, "growth_and_risk_profile_still_operational"

    if promoted or (cagr == cagr and cagr >= 0.08) or "candidate" in role or "guardrail" in family:
        return ITERATE_STATUS, "keep_for_targeted_iteration_not_default_monitoring"
    return ARCHIVE_STATUS, "use_as_context_or_reference_only"


def select_curated_strategy_shelf(
    ranked_rows: pd.DataFrame,
    *,
    limit: int = DEFAULT_CURATED_SHELF_LIMIT,
) -> pd.DataFrame:
    """Select a balanced top-N shelf without letting one recent-history winner dominate."""
    if ranked_rows.empty:
        return ranked_rows.copy()
    limit = max(int(limit), 0)
    if limit == 0:
        return ranked_rows.head(0).copy()

    ranked = ranked_rows.copy().reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    selected_keys: set[str] = set()

    def append_rows(
        frame: pd.DataFrame, bucket: str, reason: str, max_rows: int | None = None
    ) -> None:
        nonlocal selected_rows
        if len(selected_rows) >= limit or frame.empty:
            return
        added = 0
        for _, row in frame.iterrows():
            if len(selected_rows) >= limit:
                break
            if max_rows is not None and added >= max_rows:
                break
            key = _strategy_key(row)
            if key in selected_keys:
                continue
            record = row.copy()
            record["curation_bucket"] = bucket
            record["curation_reason"] = reason
            selected_rows.append(record)
            selected_keys.add(key)
            added += 1

    reference_mask = _reference_mask(ranked)
    pruned_mask = ranked.get("research_status", pd.Series("", index=ranked.index)).eq(PRUNED_STATUS)
    non_reference = ranked[~reference_mask & ~pruned_mask].copy()
    reference = ranked[reference_mask].copy()

    anchor_count = min(5, limit)
    append_rows(
        non_reference.head(anchor_count),
        "score_anchor",
        "Highest-ranked candidates after validation, promotion, robustness, and score sorting.",
    )

    family_champions = _family_champions(non_reference)
    append_rows(
        family_champions,
        "family_champion",
        "Best available candidate in a distinct research family so the shelf is not one-theme only.",
        max_rows=max(0, min(12, limit - len(selected_rows))),
    )

    operating_candidates = _contains_any(
        non_reference,
        columns=["phase", "role", "family"],
        needles=["operating_system", "final_deep_dive"],
    )
    append_rows(
        non_reference[operating_candidates],
        "operating_candidate",
        "Operating-system or final-pass candidate with enough evidence to deserve paper monitoring.",
        max_rows=max(0, min(6, limit - len(selected_rows))),
    )

    active_candidates = _contains_any(
        non_reference,
        columns=["phase", "role", "family", "strategy", "strategy_name"],
        needles=["active"],
    )
    append_rows(
        non_reference[active_candidates],
        "active_probe",
        "Active-but-human-executable probe included to test whether extra trading effort adds value.",
        max_rows=max(0, min(6, limit - len(selected_rows))),
    )

    append_rows(
        non_reference,
        "score_fill",
        "Filled remaining shelf slots by the same validation-aware score order.",
    )

    append_rows(
        reference,
        "reference_anchor",
        "Reference allocation kept visible for comparison against tactical complexity.",
    )

    if not selected_rows:
        return ranked.head(0).copy()
    shelf = pd.DataFrame(selected_rows).reset_index(drop=True)
    shelf.insert(0, "curation_rank", range(1, len(shelf) + 1))
    return shelf


def _family_champions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "family" not in frame:
        return frame.head(0).copy()
    family = frame["family"].astype(str).replace({"": "unknown", "nan": "unknown"})
    champions = frame.assign(_family=family).drop_duplicates("_family", keep="first")
    return champions.drop(columns=["_family"], errors="ignore")


def _strategy_key(row: pd.Series) -> str:
    for column in ["strategy_id", "strategy_name", "strategy", "name"]:
        value = row.get(column)
        if value is not None and str(value) and str(value) != "nan":
            return f"{column}:{value}"
    return "row:" + str(hash(tuple(row.astype(str).tolist())))


def _reference_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    family = _string_column(frame, "family")
    phase = _string_column(frame, "phase")
    role = _string_column(frame, "role")
    strategy = _first_string_column(frame, ["strategy_name", "strategy", "name"])
    return (
        family.eq("reference_portfolio")
        | phase.eq("reference")
        | role.eq("reference_portfolio")
        | strategy.str.startswith("i41_ref_")
    )


def _contains_any(frame: pd.DataFrame, *, columns: list[str], needles: list[str]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].astype(str).str.lower()
        for needle in needles:
            mask = mask | values.str.contains(needle.lower(), regex=False, na=False)
    return mask


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index)
    return frame[column].astype(str)


def _first_string_column(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame:
            return frame[column].astype(str)
    return pd.Series("", index=frame.index)


def _numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
