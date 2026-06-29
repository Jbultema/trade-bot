from __future__ import annotations

import pandas as pd

from trade_bot.dashboard.overview import (
    book_alignment_has_executions,
    book_alignment_needs_attention,
    execution_book_alignment_or_none,
)
from trade_bot.trading.book_alignment import BookAlignmentRun


def test_book_alignment_attention_requires_logged_executions() -> None:
    alignment = _alignment(
        has_executions=False,
        alignment_status="unstarted",
        material_trade_count=3,
        max_abs_delta=0.75,
    )

    assert not book_alignment_has_executions(alignment)
    assert not book_alignment_needs_attention(alignment)
    assert execution_book_alignment_or_none(alignment) is None


def test_book_alignment_attention_triggers_for_material_logged_drift() -> None:
    alignment = _alignment(
        has_executions=True,
        alignment_status="small_drift",
        material_trade_count=1,
        max_abs_delta=0.034,
    )

    assert book_alignment_has_executions(alignment)
    assert book_alignment_needs_attention(alignment)
    assert execution_book_alignment_or_none(alignment) is alignment


def test_book_alignment_attention_stays_closed_when_logged_book_is_aligned() -> None:
    alignment = _alignment(
        has_executions=True,
        alignment_status="aligned",
        material_trade_count=0,
        max_abs_delta=0.0,
    )

    assert not book_alignment_needs_attention(alignment)


def _alignment(
    *,
    has_executions: bool,
    alignment_status: str,
    material_trade_count: int,
    max_abs_delta: float,
) -> BookAlignmentRun:
    return BookAlignmentRun(
        summary=pd.DataFrame(
            [
                {
                    "has_executions": has_executions,
                    "alignment_status": alignment_status,
                    "material_trade_count": material_trade_count,
                    "max_abs_delta": max_abs_delta,
                }
            ]
        ),
        position_plan=pd.DataFrame(),
        holdings=pd.DataFrame(),
    )
