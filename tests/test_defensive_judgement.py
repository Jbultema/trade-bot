from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.strategy_candidates import runtime_outcome_scorecards
from trade_bot.research.defensive_judgement import (
    DefensiveJudgementHorizon,
    build_defensive_judgement_audit,
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


def test_runtime_scorecards_include_defensive_judgement_metrics() -> None:
    dates = pd.bdate_range("2026-01-02", periods=30)
    weights = pd.DataFrame({"QQQ": [0.35] * 30}, index=dates)
    result = _result(weights)
    prices = pd.DataFrame(
        {
            "SPY": [100 + i for i in range(30)],
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
    assert row["defensive_episode_starts"] >= 1
    assert row["defensive_judgement_label"] in {
        "thin_history",
        "frequent_false_alarm",
        "mixed_but_informative",
        "defensive_signal_useful",
        "weak_defensive_signal",
    }
