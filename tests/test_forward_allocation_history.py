from __future__ import annotations

import json

import pandas as pd
import pytest

from trade_bot.dashboard.forward_test import (
    _apply_allocation_view_window,
    _forward_weight_history_from_valuations,
    _forward_window_options,
    _prepare_allocation_frames,
)


def test_forward_weight_history_parses_daily_valuation_weights() -> None:
    valuations = pd.DataFrame(
        [
            {
                "window_id": "window-1",
                "valuation_date": "2026-01-02",
                "created_at_utc": "2026-01-02T21:00:00+00:00",
                "latest_weights_json": json.dumps({"QQQ": 0.6, "BIL": 0.4}),
            },
            {
                "window_id": "window-1",
                "valuation_date": "2026-01-03",
                "created_at_utc": "2026-01-03T21:00:00+00:00",
                "latest_weights_json": json.dumps({"QQQ": 0.55, "BIL": 0.45}),
            },
            {
                "window_id": "other-window",
                "valuation_date": "2026-01-03",
                "created_at_utc": "2026-01-03T21:00:00+00:00",
                "latest_weights_json": json.dumps({"SPY": 1.0}),
            },
        ]
    )

    history = _forward_weight_history_from_valuations(valuations, window_id="window-1")

    assert set(history.columns) == {"BIL", "QQQ"}
    assert history.index.tolist() == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
    ]
    assert history.loc[pd.Timestamp("2026-01-03"), "QQQ"] == 0.55


def test_forward_weight_history_ignores_bad_or_empty_weight_rows() -> None:
    valuations = pd.DataFrame(
        [
            {
                "window_id": "window-1",
                "valuation_date": "2026-01-02",
                "created_at_utc": "2026-01-02T21:00:00+00:00",
                "latest_weights_json": "not-json",
            },
            {
                "window_id": "window-1",
                "valuation_date": "not-a-date",
                "created_at_utc": "2026-01-03T21:00:00+00:00",
                "latest_weights_json": json.dumps({"QQQ": 1.0}),
            },
        ]
    )

    assert _forward_weight_history_from_valuations(valuations, window_id="window-1").empty


def test_forward_window_options_defaults_to_all_parseable_valued_windows() -> None:
    windows = pd.DataFrame(
        [
            {
                "window_id": "paper-default-strategy-a",
                "window_role": "challenger",
                "strategy_name": "strategy_a",
                "mode": "paper",
                "account": "default_paper_account",
                "start_date": "2026-01-01",
            },
            {
                "window_id": "paper-roster-strategy-b",
                "window_role": "champion",
                "strategy_name": "strategy_b",
                "mode": "paper",
                "account": "core_paper_roster",
                "start_date": "2026-01-01",
            },
            {
                "window_id": "paper-roster-bad-weights",
                "window_role": "challenger",
                "strategy_name": "bad_weights",
                "mode": "paper",
                "account": "core_paper_roster",
                "start_date": "2026-01-01",
            },
        ]
    )
    valuations = pd.DataFrame(
        [
            {
                "window_id": "paper-default-strategy-a",
                "valuation_date": "2026-07-06",
                "latest_weights_json": json.dumps({"QQQ": 0.5, "BIL": 0.5}),
            },
            {
                "window_id": "paper-roster-strategy-b",
                "valuation_date": "2026-07-06",
                "latest_weights_json": json.dumps({"SPY": 1.0}),
            },
            {
                "window_id": "paper-roster-bad-weights",
                "valuation_date": "2026-07-06",
                "latest_weights_json": "not-json",
            },
        ]
    )

    all_options = _forward_window_options(windows, valuations)
    scoped_options = _forward_window_options(
        windows,
        valuations,
        mode="paper",
        account="default_paper_account",
    )

    assert set(all_options["strategy_name"]) == {"strategy_a", "strategy_b"}
    assert "bad_weights" not in set(all_options["strategy_name"])
    assert scoped_options["strategy_name"].tolist() == ["strategy_a"]


def test_prepare_allocation_frames_adds_cash_and_compacts_small_assets() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-03"])
    historical = pd.DataFrame(
        {
            "QQQ": [0.5, 0.55],
            "IWM": [0.2, 0.15],
            "SMH": [0.1, 0.05],
        },
        index=index,
    )
    forward = pd.DataFrame(
        {
            "QQQ": [0.45, 0.4],
            "IWM": [0.2, 0.25],
            "BIL": [0.25, 0.3],
        },
        index=index,
    )

    prepared_historical, prepared_forward = _prepare_allocation_frames(
        historical,
        forward,
        max_assets=2,
    )

    assert "cash_or_unallocated" in prepared_historical
    assert "other_or_cash" in prepared_historical
    assert "other_or_cash" in prepared_forward
    assert prepared_historical.loc[index[0], "cash_or_unallocated"] == pytest.approx(0.2)
    assert prepared_forward.loc[index[1], "other_or_cash"] == pytest.approx(0.55)


def test_allocation_view_window_can_focus_forward_overlap() -> None:
    index = pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"])
    historical = pd.DataFrame({"QQQ": [0.5, 0.6, 0.7]}, index=index)
    forward = pd.DataFrame({"QQQ": [0.6, 0.7]}, index=index[1:])

    filtered_historical, filtered_forward = _apply_allocation_view_window(
        historical,
        forward,
        view="Forward overlap",
        start_date=pd.Timestamp("2026-01-01"),
    )

    assert filtered_historical.index.min() == pd.Timestamp("2026-01-02")
    assert filtered_forward.index.min() == pd.Timestamp("2026-01-02")
