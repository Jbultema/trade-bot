from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.strategy_candidates import runtime_outcome_scorecards
from trade_bot.research.defensive_judgement import (
    DefensiveJudgementHorizon,
    build_defensive_judgement_audit,
    defensive_false_alarm_bayes_update,
    effective_defensive_weight,
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
    bot_config = SimpleNamespace(strategies={"candidate": object()})

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
    }
