from __future__ import annotations

import pandas as pd

DEFAULT_CURATED_SHELF_LIMIT = 25


def rank_strategy_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable research ranking before explicit curation/diversification."""
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
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
    ranked = ranked.sort_values(
        [
            "_promotion_rank",
            "_validation_rank",
            "_curation_sort_score",
            "robustness_score",
            "calmar",
            "walk_forward_positive_rate",
            "left_tail_regime_return",
            "iteration",
        ],
        ascending=[True, True, False, False, False, False, False, False],
        na_position="last",
    )
    return ranked.drop(columns=["_promotion_rank", "_validation_rank"], errors="ignore")


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
    non_reference = ranked[~reference_mask].copy()
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
