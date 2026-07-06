from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import trade_bot.dashboard.performance as performance_module
from trade_bot.backtest.engine import BacktestResult


def test_performance_options_include_curated_research_shelf(monkeypatch) -> None:
    baseline_run = SimpleNamespace(results={"buy_hold_spy": _result("buy_hold_spy")})
    catalog = pd.DataFrame(
        [
            {
                "source": "baseline",
                "strategy": "buy_hold_spy",
                "display_name": "Buy Hold Spy",
                "approach_id": "baseline::buy_hold_spy",
            },
            {
                "source": "experiment",
                "strategy": "curated_growth_candidate",
                "display_name": "Curated Growth Candidate",
                "approach_id": "experiment::01::curated_growth_candidate",
            },
        ]
    )
    scorecards = pd.DataFrame(
        [
            {
                "iteration": 1,
                "strategy": "curated_growth_candidate",
                "phase": "growth_frontier",
                "family": "high_cagr",
                "role": "candidate",
                "promotion_decision": "promote_candidate",
                "promotion_score": 0.92,
                "robustness_score": 0.85,
                "cagr": 0.148,
                "max_drawdown": -0.214,
                "calmar": 0.69,
                "walk_forward_positive_rate": 0.80,
                "left_tail_regime_return": -0.08,
                "average_turnover": 0.06,
            }
        ]
    )
    monkeypatch.setattr(
        performance_module,
        "build_approach_catalog",
        lambda _bot_config: catalog,
    )

    options = performance_module._performance_option_frame(
        baseline_run,
        bot_config=object(),
        experiment_scorecards=scorecards,
    )

    assert options["strategy"].tolist() == ["buy_hold_spy", "curated_growth_candidate"]
    label_lookup = dict(zip(options["strategy"], options["label"], strict=False))
    assert label_lookup["buy_hold_spy"] == "configured | Buy Hold Spy"
    assert label_lookup["curated_growth_candidate"].startswith(
        "curated #01 | Curated Growth Candidate"
    )


def _result(name: str) -> BacktestResult:
    index = pd.bdate_range("2026-01-01", periods=3)
    returns = pd.Series([0.0, 0.01, 0.01], index=index, name=name)
    equity = 100.0 * (1.0 + returns).cumprod()
    weights = pd.DataFrame({"SPY": [1.0, 1.0, 1.0]}, index=index)
    return BacktestResult(
        name=name,
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index, name=name),
        transaction_costs=pd.Series(0.0, index=index, name=name),
    )
