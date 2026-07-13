from __future__ import annotations

from typing import Any

import pandas as pd

from trade_bot.dashboard.book_alignment import _render_book_alignment
from trade_bot.dashboard.briefs import _render_decision_brief, _render_operating_brief
from trade_bot.dashboard.components import _render_action_headline
from trade_bot.DEFAULTS import DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT
from trade_bot.research.action_headline import ActionHeadline
from trade_bot.research.baselines import BaselineRun
from trade_bot.trading.book_alignment import BookAlignmentRun


def render_operating_overview(
    *,
    baseline_run: BaselineRun,
    headline: ActionHeadline,
    open_ticket_count: int,
    experiment_scorecards: pd.DataFrame,
    default_book_alignment: BookAlignmentRun,
    previous_run: BaselineRun | None = None,
    execution_book_alignment: BookAlignmentRun | None = None,
) -> None:
    """Render the top operating readout before the deep-dive workbenches."""

    _render_action_headline(headline)
    _render_operating_brief(
        baseline_run=baseline_run,
        headline=headline,
        book_alignment=execution_book_alignment,
    )
    _render_decision_brief(
        baseline_run=baseline_run,
        headline=headline,
        open_ticket_count=open_ticket_count,
        experiment_scorecards=experiment_scorecards,
    )
    _render_book_alignment(
        default_book_alignment,
        heading="Book Alignment",
        show_position_plan=False,
    )


def execution_book_alignment_or_none(alignment: BookAlignmentRun) -> BookAlignmentRun | None:
    return alignment if book_alignment_is_usable(alignment) else None


def headline_position_plan(
    *,
    baseline_run: BaselineRun,
    default_book_alignment: BookAlignmentRun,
) -> pd.DataFrame:
    if book_alignment_is_usable(default_book_alignment):
        return default_book_alignment.position_plan
    return baseline_run.trade_decision.position_plan


def book_alignment_is_usable(alignment: BookAlignmentRun) -> bool:
    if not book_alignment_has_executions(alignment):
        return False
    warning = str(alignment.summary.iloc[0].get("account_value_warning", "")).strip()
    return not warning


def book_alignment_has_executions(alignment: BookAlignmentRun) -> bool:
    if alignment.summary.empty:
        return False
    return bool(alignment.summary.iloc[0].get("has_executions", False))


def book_alignment_needs_attention(alignment: BookAlignmentRun) -> bool:
    if alignment.summary.empty or not book_alignment_has_executions(alignment):
        return False
    row: dict[str, Any] = alignment.summary.iloc[0].to_dict()
    status = str(row.get("alignment_status", "unknown"))
    if status == "aligned":
        return False
    material_trade_count = _safe_int(row.get("material_trade_count"))
    max_abs_delta = abs(_safe_float(row.get("max_abs_delta")))
    return (
        material_trade_count > 0
        or max_abs_delta >= DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT
    )


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
