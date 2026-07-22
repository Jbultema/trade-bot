from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import load_config
from trade_bot.dashboard.strategy_candidates import runtime_outcome_scorecards
from trade_bot.DEFAULTS import DEFAULT_CONFIG_PATH
from trade_bot.research.defensive_judgement import (
    DefensiveJudgementHorizon,
    build_defensive_judgement_audit,
    current_defensive_setup_context,
    defensive_false_alarm_bayes_update,
    effective_defensive_weight,
    write_defensive_judgement_report,
)


def _result(weights: pd.DataFrame, returns: pd.Series | None = None) -> BacktestResult:
    if returns is None:
        returns = pd.Series(0.0, index=weights.index)
    equity = 100.0 * (1.0 + returns).cumprod()
    return BacktestResult(
        name="candidate",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=weights.diff().abs().sum(axis=1).fillna(0.0),
        transaction_costs=pd.Series(0.0, index=weights.index),
    )


def test_defensive_judgement_counts_residual_cash_as_defensive() -> None:
    dates = pd.bdate_range("2026-01-02", periods=4)
    result = _result(
        pd.DataFrame(
            {
                "QQQ": [0.80, 0.35, 0.35, 0.20],
            },
            index=dates,
        )
    )

    defensive = effective_defensive_weight(result)

    assert defensive.iloc[0] == pytest.approx(0.20)
    assert defensive.iloc[1] == pytest.approx(0.65)
    assert defensive.iloc[3] == pytest.approx(0.80)


def test_defensive_report_replaces_dashboard_facing_readout(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=90)
    weights = pd.DataFrame({"QQQ": [0.40] * len(dates)}, index=dates)
    result = _result(weights)
    prices = pd.DataFrame(
        {
            "SPY": [100.0 + index * 0.1 for index in range(len(dates))],
            "QQQ": [100.0 + index * 0.2 for index in range(len(dates))],
            "BIL": [100.0 + index * 0.01 for index in range(len(dates))],
        },
        index=dates,
    )

    outputs = write_defensive_judgement_report(
        results={"focus": result},
        prices=prices,
        output_dir=tmp_path,
        focus_strategy="focus",
    )

    exposure = pd.read_csv(outputs["current_exposure"])
    readout = outputs["readout"].read_text(encoding="utf-8")
    assert exposure.iloc[0]["current_defensive_weight"] == pytest.approx(0.60)
    assert str(dates.max().date()) in readout
    assert "Current effective defensive weight: 60.0%" in readout


def test_defensive_judgement_classifies_episode_outcomes() -> None:
    dates = pd.bdate_range("2026-01-02", periods=18)
    weights = pd.DataFrame({"QQQ": 1.0}, index=dates)
    weights.iloc[2:7, 0] = 0.30
    weights.iloc[10:, 0] = 0.30
    result = _result(weights)
    prices = pd.DataFrame(
        {
            "SPY": [
                100,
                101,
                100,
                98,
                94,
                93,
                94,
                96,
                97,
                98,
                100,
                101,
                102,
                104,
                106,
                108,
                109,
                110,
            ],
            "QQQ": [
                100,
                101,
                100,
                99,
                98,
                97,
                98,
                99,
                100,
                101,
                100,
                103,
                106,
                109,
                112,
                115,
                117,
                119,
            ],
            "BIL": [100 + i * 0.01 for i in range(18)],
        },
        index=dates,
    )
    horizon = DefensiveJudgementHorizon(
        label="1w",
        trading_days=5,
        drawdown_correct_threshold=-0.03,
        false_alarm_excess_threshold=0.01,
        false_alarm_drawdown_floor=-0.02,
    )

    audit = build_defensive_judgement_audit(
        result,
        prices,
        thresholds=(0.65,),
        horizons=(horizon,),
    )

    one_week = audit["summary"].iloc[0]
    assert one_week["episode_starts"] == 2
    assert one_week["correct_defense"] == 1
    assert one_week["false_alarm"] == 1
    assert one_week["correct_defense_rate"] == pytest.approx(0.5)
    assert one_week["false_alarm_rate"] == pytest.approx(0.5)
    assert one_week["benchmark_ticker"] == "SPY"

    qqq_audit = build_defensive_judgement_audit(
        result,
        prices,
        thresholds=(0.65,),
        horizons=(horizon,),
        benchmark_ticker="QQQ",
    )
    assert qqq_audit["summary"].iloc[0]["benchmark_ticker"] == "QQQ"
    assert "rerisk_within_horizon_rate" in qqq_audit["summary"]
    assert "median_missed_upside" in qqq_audit["summary"]


def test_defensive_judgement_tracks_episode_follow_through() -> None:
    dates = pd.bdate_range("2026-01-02", periods=90)
    weights = pd.DataFrame({"QQQ": [1.0] * 90}, index=dates)
    weights.iloc[30:38, 0] = 0.30
    result = _result(weights)
    prices = pd.DataFrame(
        {
            "SPY": [100.0] * 90,
            "QQQ": [100.0] * 90,
            "BIL": [100.0 + i * 0.01 for i in range(90)],
        },
        index=dates,
    )
    prices.loc[dates[31:35], "QQQ"] = [97.0, 95.0, 94.0, 98.0]
    prices.loc[dates[35:], "QQQ"] = [100.0 + i * 0.3 for i in range(55)]

    audit = build_defensive_judgement_audit(
        result,
        prices,
        thresholds=(0.65,),
        benchmark_ticker="QQQ",
    )

    event = audit["events"][audit["events"]["horizon"].eq("1m")].iloc[0]
    assert event["days_to_rerisk"] == 8
    assert bool(event["rerisked_within_horizon"]) is True
    assert event["benchmark_forward_max_drawdown"] <= -0.05
    summary = audit["summary"][audit["summary"]["horizon"].eq("1m")].iloc[0]
    assert summary["rerisk_within_horizon_rate"] == pytest.approx(1.0)
    assert summary["median_days_to_rerisk"] == pytest.approx(8.0)
    assert summary["median_avoided_drawdown"] is not None


def test_runtime_scorecards_include_defensive_judgement_metrics() -> None:
    dates = pd.bdate_range("2026-01-02", periods=30)
    weights = pd.DataFrame({"QQQ": [0.35] * 30}, index=dates)
    result = _result(weights)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i for i in range(30)],
            "QQQ": [100 + i * 2 for i in range(30)],
            "BIL": [100 + i * 0.01 for i in range(30)],
        },
        index=dates,
    )
    baseline_run = SimpleNamespace(
        prices=prices,
        results={"candidate": result},
        metrics=pd.DataFrame(
            [{"name": "candidate", "cagr": 0.2, "max_drawdown": -0.2}]
        ).set_index("name"),
        window_summary=pd.DataFrame(),
    )
    configured = load_config(DEFAULT_CONFIG_PATH)
    bot_config = SimpleNamespace(
        strategies={"candidate": object()},
        execution=configured.execution,
        data=configured.data,
    )

    scorecards = runtime_outcome_scorecards(
        baseline_run=baseline_run,
        bot_config=bot_config,
    )

    row = scorecards.iloc[0]
    assert row["current_defensive_weight"] == pytest.approx(0.65)
    assert row["current_risk_weight"] == pytest.approx(0.35)
    assert "qqq_defensive_false_alarm_rate" in row
    assert row["defensive_episode_starts"] >= 1
    assert row["defensive_judgement_label"] in {
        "thin_history",
        "frequent_false_alarm",
        "mixed_but_informative",
        "defensive_signal_useful",
        "weak_defensive_signal",
    }


def test_defensive_false_alarm_bayes_update_uses_recent_evidence() -> None:
    dates = pd.bdate_range("2020-01-02", periods=8, freq="260B")
    events = pd.DataFrame(
        {
            "date": dates,
            "threshold": [0.65] * 8,
            "horizon": ["1m"] * 8,
            "defensive_weight": [0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73],
            "judgement": [
                "correct_defense",
                "correct_defense",
                "mixed_or_early",
                "correct_defense",
                "false_alarm",
                "false_alarm",
                "false_alarm",
                "mixed_or_early",
            ],
        }
    )

    update = defensive_false_alarm_bayes_update(
        events,
        threshold=0.65,
        horizon="1m",
        current_defensive_weight=0.70,
        recent_years=3.0,
        prior_strength=8.0,
    )

    assert update["historical_episode_starts"] == 8
    assert update["recent_episode_starts"] >= 3
    assert update["posterior_false_alarm_rate"] is not None
    assert update["sniff_test_label"] in {
        "recent_false_alarms_elevated",
        "false_alarm_risk_high",
        "mixed_context",
        "similar_setups_false_alarm_prone",
    }


def test_defensive_false_alarm_bayes_update_uses_similar_setups() -> None:
    dates = pd.bdate_range("2020-01-02", periods=10, freq="120B")
    events = pd.DataFrame(
        {
            "date": dates,
            "threshold": [0.65] * 10,
            "horizon": ["1m"] * 10,
            "defensive_weight": [0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75],
            "risk_off": [0.10, 0.12, 0.14, 0.16, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
            "benchmark_trailing_21d_return": [
                0.05,
                0.04,
                0.06,
                0.03,
                -0.05,
                -0.04,
                -0.06,
                -0.05,
                -0.03,
                -0.04,
            ],
            "benchmark_forward_max_drawdown": [
                -0.01,
                -0.01,
                -0.01,
                -0.01,
                -0.08,
                -0.07,
                -0.06,
                -0.09,
                -0.07,
                -0.08,
            ],
            "benchmark_excess_vs_cash": [0.03, 0.03, 0.04, 0.03, -0.01, 0.00, -0.02, 0.00, -0.01, 0.01],
            "strategy_excess_vs_benchmark": [0.0] * 10,
            "judgement": [
                "false_alarm",
                "false_alarm",
                "false_alarm",
                "mixed_or_early",
                "correct_defense",
                "correct_defense",
                "correct_defense",
                "correct_defense",
                "correct_defense",
                "mixed_or_early",
            ],
        }
    )

    update = defensive_false_alarm_bayes_update(
        events,
        threshold=0.65,
        horizon="1m",
        current_defensive_weight=0.72,
        current_setup={"risk_off": 0.33, "benchmark_trailing_21d_return": -0.04},
        recent_years=10.0,
        prior_strength=8.0,
    )

    assert update["similar_episode_starts"] >= 5
    assert update["similar_correct_defense_rate"] is not None
    assert update["similar_correct_defense_rate"] > update["similar_false_alarm_rate"]
    assert update["similarity_basis"] in {
        "multi_feature_context",
        "nearest_multi_feature_context",
    }


def test_current_defensive_setup_context_uses_point_in_time_features() -> None:
    dates = pd.bdate_range("2026-01-02", periods=80)
    weights = pd.DataFrame({"QQQ": [0.35] * 80}, index=dates)
    result = _result(weights)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0 + i for i in range(80)],
            "SPY": [100.0 + i * 0.5 for i in range(80)],
            "BIL": [100.0 + i * 0.01 for i in range(80)],
        },
        index=dates,
    )
    scenario_context = pd.DataFrame(
        {
            "date": [dates[-3]],
            "risk_off": [0.31],
            "transition": [0.12],
            "defensive_or_transition_probability": [0.43],
            "constructive_probability": [0.21],
            "scenario_context": ["risk_off_elevated"],
        }
    )

    context = current_defensive_setup_context(
        result,
        prices,
        benchmark_ticker="QQQ",
        scenario_context=scenario_context,
    )

    assert context["defensive_weight"] == pytest.approx(0.65)
    assert context["risk_off"] == pytest.approx(0.31)
    assert context["benchmark_trailing_21d_return"] is not None
