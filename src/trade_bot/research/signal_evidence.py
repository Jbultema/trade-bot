from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_SIGNAL_EVIDENCE_DATA_STATUS,
    DEFAULT_SIGNAL_EVIDENCE_METRIC_WEIGHTS,
    DEFAULT_SIGNAL_EVIDENCE_MIN_PAIRED_TESTS,
    DEFAULT_SIGNAL_EVIDENCE_PROMISING_SCORE,
    DEFAULT_SIGNAL_EVIDENCE_PROVEN_SCORE,
    DEFAULT_SIGNAL_EVIDENCE_SIGNAL_FAMILY_KEYWORDS,
)

SIGNAL_EVIDENCE_SUMMARY_COLUMNS = [
    "signal_family",
    "signal_label",
    "data_status",
    "evidence_tier",
    "candidate_count",
    "paired_tests",
    "best_strategy",
    "best_promotion_score",
    "best_cagr",
    "best_max_drawdown",
    "median_cagr",
    "median_max_drawdown",
    "median_reentry_score",
    "median_average_turnover",
    "median_delta_promotion_score",
    "median_delta_cagr",
    "median_delta_max_drawdown",
    "median_delta_calmar",
    "median_delta_reentry_score",
    "median_delta_average_turnover",
    "cagr_win_rate",
    "drawdown_win_rate",
    "reentry_win_rate",
    "churn_win_rate",
    "promotion_win_rate",
    "calmar_win_rate",
    "net_evidence_score",
    "evidence_label",
    "recommendation",
    "caveat",
]

SIGNAL_MARGINAL_TEST_COLUMNS = [
    "signal_family",
    "signal_label",
    "child_strategy",
    "parent_strategy",
    "iteration",
    "child_phase",
    "child_family",
    "child_role",
    "delta_promotion_score",
    "delta_cagr",
    "delta_max_drawdown",
    "delta_calmar",
    "delta_reentry_score",
    "delta_average_turnover",
    "delta_left_tail_regime_return",
    "child_cagr",
    "parent_cagr",
    "child_max_drawdown",
    "parent_max_drawdown",
    "child_average_turnover",
    "parent_average_turnover",
    "hypothesis",
]


def build_signal_family_evidence(
    scorecards: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize which signal families have empirical support.

    This is not a new trading signal. It is evidence infrastructure for deciding
    which research families deserve dashboard space, further experiments, or
    paper-monitoring priority.
    """

    tagged = _ensure_signal_family_tags(scorecards, candidates)
    if tagged.empty:
        return pd.DataFrame(columns=SIGNAL_EVIDENCE_SUMMARY_COLUMNS)
    marginal_tests = build_signal_family_marginal_tests(tagged)
    rows = [
        _family_summary_row(signal_family, tagged, marginal_tests)
        for signal_family in sorted(DEFAULT_SIGNAL_EVIDENCE_SIGNAL_FAMILY_KEYWORDS)
    ]
    return pd.DataFrame(rows, columns=SIGNAL_EVIDENCE_SUMMARY_COLUMNS).sort_values(
        ["net_evidence_score", "paired_tests", "best_promotion_score"],
        ascending=False,
    ).reset_index(drop=True)


def build_signal_family_marginal_tests(scorecards: pd.DataFrame) -> pd.DataFrame:
    """Build paired parent/control deltas by signal family.

    Deltas are child minus parent. Since max drawdown is negative, a positive
    delta means drawdown improved. Since turnover is a cost/churn proxy, a
    negative turnover delta is a win.
    """

    tagged = _ensure_signal_family_tags(scorecards)
    if tagged.empty or "parent" not in tagged or "strategy" not in tagged:
        return pd.DataFrame(columns=SIGNAL_MARGINAL_TEST_COLUMNS)

    latest = _latest_strategy_rows(tagged)
    indexed = latest.set_index("strategy", drop=False)
    rows: list[dict[str, object]] = []
    for _, child in latest.iterrows():
        parent_name = str(child.get("parent", "") or "").strip()
        if not parent_name or parent_name not in indexed.index:
            continue
        parent = indexed.loc[parent_name]
        child_families = _split_signal_families(child.get("signal_families", ""))
        parent_families = _split_signal_families(parent.get("signal_families", ""))
        incremental_families = child_families - parent_families
        if not incremental_families:
            incremental_families = child_families
        for signal_family in sorted(incremental_families):
            if signal_family not in DEFAULT_SIGNAL_EVIDENCE_SIGNAL_FAMILY_KEYWORDS:
                continue
            rows.append(_marginal_test_row(signal_family, child, parent))
    return pd.DataFrame(rows, columns=SIGNAL_MARGINAL_TEST_COLUMNS)


def tag_scorecard_signal_families(
    scorecards: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if scorecards.empty:
        return scorecards.copy()

    frame = scorecards.copy()
    if "strategy" not in frame and frame.index.name == "strategy":
        frame = frame.reset_index()
    if candidates is not None and not candidates.empty and "strategy" in frame:
        candidate_columns = [
            column
            for column in ["strategy", "hypothesis", "type", "phase", "family", "role"]
            if column in candidates.columns
        ]
        if candidate_columns:
            candidate_context = candidates[candidate_columns].drop_duplicates("strategy")
            frame = frame.merge(
                candidate_context,
                on="strategy",
                how="left",
                suffixes=("", "_candidate"),
            )
            if "hypothesis_candidate" in frame and "hypothesis" in frame:
                frame["hypothesis"] = frame["hypothesis"].fillna(frame["hypothesis_candidate"])
            elif "hypothesis_candidate" in frame:
                frame["hypothesis"] = frame["hypothesis_candidate"]
    frame["signal_families"] = frame.apply(_signal_family_tags_for_row, axis=1)
    return frame


def _ensure_signal_family_tags(
    scorecards: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if "signal_families" in scorecards:
        return scorecards.copy()
    return tag_scorecard_signal_families(scorecards, candidates)


def signal_evidence_takeaways(evidence: pd.DataFrame) -> list[str]:
    if evidence.empty:
        return ["No signal-family evidence has been computed yet."]
    top = evidence.sort_values("net_evidence_score", ascending=False).head(3)
    lines = []
    for _, row in top.iterrows():
        lines.append(
            f"{row['signal_label']}: {row['evidence_label']} "
            f"({int(row['paired_tests'])} paired test(s), best CAGR {float(row['best_cagr']):.1%})."
        )
    weak = evidence[evidence["evidence_label"].isin(["context_only", "not_proven"])]
    if not weak.empty:
        lines.append(
            "Weakest default candidates for pruning/context-only treatment: "
            + ", ".join(str(value) for value in weak.head(3)["signal_label"])
            + "."
        )
    return lines


def _family_summary_row(
    signal_family: str,
    tagged: pd.DataFrame,
    marginal_tests: pd.DataFrame,
) -> dict[str, object]:
    family_rows = _rows_for_signal_family(tagged, signal_family)
    family_tests = (
        marginal_tests[marginal_tests["signal_family"] == signal_family].copy()
        if not marginal_tests.empty and "signal_family" in marginal_tests
        else pd.DataFrame()
    )
    candidate_count = int(len(family_rows))
    paired_tests = int(len(family_tests))
    best = _best_row(family_rows)
    metrics = _family_metric_summary(family_rows)
    deltas = _marginal_delta_summary(family_tests)
    evidence_score = _net_evidence_score(family_tests)
    evidence_label = _evidence_label(
        paired_tests=paired_tests,
        candidate_count=candidate_count,
        net_evidence_score=evidence_score,
        median_delta_promotion_score=deltas["median_delta_promotion_score"],
    )
    return {
        "signal_family": signal_family,
        "signal_label": _signal_label(signal_family),
        "data_status": DEFAULT_SIGNAL_EVIDENCE_DATA_STATUS.get(signal_family, "unknown"),
        "evidence_tier": "paired_ablation" if paired_tests else "family_association",
        "candidate_count": candidate_count,
        "paired_tests": paired_tests,
        "best_strategy": str(best.get("strategy", "")),
        "best_promotion_score": _as_float(best.get("promotion_score")),
        "best_cagr": _as_float(best.get("cagr")),
        "best_max_drawdown": _as_float(best.get("max_drawdown")),
        **metrics,
        **deltas,
        "net_evidence_score": evidence_score,
        "evidence_label": evidence_label,
        "recommendation": _recommendation(evidence_label, paired_tests),
        "caveat": _caveat(signal_family, paired_tests),
    }


def _family_metric_summary(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "median_cagr": _median(rows, "cagr"),
        "median_max_drawdown": _median(rows, "max_drawdown"),
        "median_reentry_score": _median(rows, "reentry_score"),
        "median_average_turnover": _median(rows, "average_turnover"),
    }


def _marginal_delta_summary(tests: pd.DataFrame) -> dict[str, float]:
    deltas = {
        "median_delta_promotion_score": _median(tests, "delta_promotion_score"),
        "median_delta_cagr": _median(tests, "delta_cagr"),
        "median_delta_max_drawdown": _median(tests, "delta_max_drawdown"),
        "median_delta_calmar": _median(tests, "delta_calmar"),
        "median_delta_reentry_score": _median(tests, "delta_reentry_score"),
        "median_delta_average_turnover": _median(tests, "delta_average_turnover"),
    }
    deltas.update(
        {
            "cagr_win_rate": _win_rate(tests, "delta_cagr", positive_is_good=True),
            "drawdown_win_rate": _win_rate(tests, "delta_max_drawdown", positive_is_good=True),
            "reentry_win_rate": _win_rate(tests, "delta_reentry_score", positive_is_good=True),
            "churn_win_rate": _win_rate(tests, "delta_average_turnover", positive_is_good=False),
            "promotion_win_rate": _win_rate(
                tests, "delta_promotion_score", positive_is_good=True
            ),
            "calmar_win_rate": _win_rate(tests, "delta_calmar", positive_is_good=True),
        }
    )
    return deltas


def _net_evidence_score(tests: pd.DataFrame) -> float:
    if tests.empty:
        return 0.0
    score = 0.0
    for metric, weight in DEFAULT_SIGNAL_EVIDENCE_METRIC_WEIGHTS.items():
        delta_column = {
            "cagr_win_rate": "delta_cagr",
            "drawdown_win_rate": "delta_max_drawdown",
            "reentry_win_rate": "delta_reentry_score",
            "churn_win_rate": "delta_average_turnover",
            "promotion_win_rate": "delta_promotion_score",
            "calmar_win_rate": "delta_calmar",
        }[metric]
        score += float(weight) * _win_rate(
            tests,
            delta_column,
            positive_is_good=metric != "churn_win_rate",
        )
    return float(score)


def _marginal_test_row(signal_family: str, child: pd.Series, parent: pd.Series) -> dict[str, object]:
    return {
        "signal_family": signal_family,
        "signal_label": _signal_label(signal_family),
        "child_strategy": str(child.get("strategy", "")),
        "parent_strategy": str(parent.get("strategy", "")),
        "iteration": _as_float(child.get("iteration")),
        "child_phase": str(child.get("phase", "")),
        "child_family": str(child.get("family", "")),
        "child_role": str(child.get("role", "")),
        "delta_promotion_score": _delta(child, parent, "promotion_score"),
        "delta_cagr": _delta(child, parent, "cagr"),
        "delta_max_drawdown": _delta(child, parent, "max_drawdown"),
        "delta_calmar": _delta(child, parent, "calmar"),
        "delta_reentry_score": _delta(child, parent, "reentry_score"),
        "delta_average_turnover": _delta(child, parent, "average_turnover"),
        "delta_left_tail_regime_return": _delta(child, parent, "left_tail_regime_return"),
        "child_cagr": _as_float(child.get("cagr")),
        "parent_cagr": _as_float(parent.get("cagr")),
        "child_max_drawdown": _as_float(child.get("max_drawdown")),
        "parent_max_drawdown": _as_float(parent.get("max_drawdown")),
        "child_average_turnover": _as_float(child.get("average_turnover")),
        "parent_average_turnover": _as_float(parent.get("average_turnover")),
        "hypothesis": str(child.get("hypothesis", "")),
    }


def _signal_family_tags_for_row(row: pd.Series) -> str:
    context = " ".join(
        str(row.get(column, ""))
        for column in [
            "strategy",
            "display_name",
            "phase",
            "family",
            "role",
            "scenario_sizing",
            "future_state_model",
            "strategy_drawdown_model",
            "decision_sanity",
            "hypothesis",
        ]
    ).lower()
    tags = {
        family
        for family, keywords in DEFAULT_SIGNAL_EVIDENCE_SIGNAL_FAMILY_KEYWORDS.items()
        if _contains_any(context, keywords)
    }
    if str(row.get("decision_sanity", "")).strip():
        tags.add("decision_sanity")
    if str(row.get("future_state_model", "")).strip() or str(
        row.get("strategy_drawdown_model", "")
    ).strip():
        tags.add("ml_models")
    return ";".join(sorted(tags))


def _contains_any(context: str, keywords: Iterable[str]) -> bool:
    return any(keyword.lower() in context for keyword in keywords)


def _rows_for_signal_family(frame: pd.DataFrame, signal_family: str) -> pd.DataFrame:
    if frame.empty or "signal_families" not in frame:
        return pd.DataFrame()
    return frame[
        frame["signal_families"].fillna("").astype(str).str.split(";").map(
            lambda values: signal_family in set(values)
        )
    ].copy()


def _latest_strategy_rows(scorecards: pd.DataFrame) -> pd.DataFrame:
    frame = scorecards.copy()
    if "strategy" not in frame:
        return frame.iloc[:0].copy()
    if "iteration" in frame:
        frame["_sort_iteration"] = pd.to_numeric(frame["iteration"], errors="coerce").fillna(-1)
        frame = frame.sort_values(["strategy", "_sort_iteration"])
    return frame.drop_duplicates("strategy", keep="last").drop(columns=["_sort_iteration"], errors="ignore")


def _split_signal_families(value: object) -> set[str]:
    return {item for item in str(value or "").split(";") if item}


def _best_row(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=object)
    sort_column = "promotion_score" if "promotion_score" in rows else "cagr"
    return rows.sort_values(sort_column, ascending=False).iloc[0]


def _delta(child: pd.Series, parent: pd.Series, column: str) -> float:
    child_value = _as_float(child.get(column))
    parent_value = _as_float(parent.get(column))
    if not np.isfinite(child_value) or not np.isfinite(parent_value):
        return float("nan")
    return child_value - parent_value


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


def _win_rate(frame: pd.DataFrame, column: str, *, positive_is_good: bool) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float((values > 0.0).mean() if positive_is_good else (values < 0.0).mean())


def _evidence_label(
    *,
    paired_tests: int,
    candidate_count: int,
    net_evidence_score: float,
    median_delta_promotion_score: float,
) -> str:
    if paired_tests >= DEFAULT_SIGNAL_EVIDENCE_MIN_PAIRED_TESTS:
        if (
            net_evidence_score >= DEFAULT_SIGNAL_EVIDENCE_PROVEN_SCORE
            and median_delta_promotion_score >= 0.0
        ):
            return "validated_contributor"
        if net_evidence_score >= DEFAULT_SIGNAL_EVIDENCE_PROMISING_SCORE:
            return "promising_mixed"
        return "not_proven"
    if paired_tests:
        return "needs_more_ablation"
    if candidate_count >= DEFAULT_SIGNAL_EVIDENCE_MIN_PAIRED_TESTS:
        return "context_only"
    return "research_gap"


def _recommendation(evidence_label: str, paired_tests: int) -> str:
    if evidence_label == "validated_contributor":
        return "Keep in model-search surface and paper-candidate reviews."
    if evidence_label == "promising_mixed":
        return "Keep testing; inspect failure cases before promotion."
    if evidence_label == "needs_more_ablation":
        return "Create more paired ablations before treating as a driver."
    if evidence_label == "context_only":
        return "Show as explanatory context unless a paired test promotes it."
    if paired_tests:
        return "Deprioritize or retune; current paired evidence is weak."
    return "Track as research backlog, not an operating signal."


def _caveat(signal_family: str, paired_tests: int) -> str:
    data_status = DEFAULT_SIGNAL_EVIDENCE_DATA_STATUS.get(signal_family, "unknown")
    if paired_tests == 0:
        return f"No clean parent/control ablation yet; data status: {data_status}."
    return f"Paired tests are historical and cost-aware through backtest execution; data status: {data_status}."


def _signal_label(signal_family: str) -> str:
    labels = {
        "ai_value_chain": "AI value-chain segmentation",
        "breadth": "Breadth and participation",
        "concentration_dispersion": "Concentration / dispersion",
        "credit": "Credit conditions",
        "decision_sanity": "Decision sanity overlay",
        "earnings_revision": "Earnings revision proxies",
        "macro_policy": "Macro / policy pressure",
        "ml_models": "ML model overlays",
        "reentry_timing": "Re-entry timing",
        "sector_rotation": "Sector rotation",
        "trend_momentum": "Trend / momentum",
        "volatility": "Volatility / instability",
    }
    return labels.get(signal_family, signal_family.replace("_", " ").title())


def _as_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")
