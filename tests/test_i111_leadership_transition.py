from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import (
    BotConfig,
    DataConfig,
    ExecutionConfig,
    StrategyConfig,
)
from trade_bot.research.i111_leadership_transition import (
    TransitionCandidate,
    _current_allocations,
    build_leadership_transition_candidates,
    select_training_winners,
)


def test_leadership_candidates_are_native_and_use_risk_sleeve_caps() -> None:
    config = BotConfig(
        data=DataConfig(start="2005-01-01"),
        execution=ExecutionConfig(),
        primary_strategy="primary",
        universe={"core": ["QQQ", "BIL"]},
        strategies={
            "primary": StrategyConfig(
                type="dual_momentum",
                tickers=["QQQ"],
                defensive_ticker="BIL",
            )
        },
    )

    candidates = build_leadership_transition_candidates(config)

    assert len(candidates) == 48
    assert all(candidate.strategy.type == "dual_momentum_risk_repair" for candidate in candidates)
    assert all(
        candidate.strategy.risk_repair_ai_cap_basis == "risk_sleeve" for candidate in candidates
    )
    assert all(
        candidate.strategy.risk_repair_ai_excess_destination == "diversifier_mix"
        for candidate in candidates
    )


def test_training_selection_does_not_read_holdout_or_post_selection() -> None:
    rows: list[dict[str, object]] = []
    for candidate, training_score, later_score in (
        ("stable", 0.20, -1.0),
        ("later_lucky", 0.01, 5.0),
    ):
        for period in ("development", "validation", "holdout", "post_selection"):
            value = training_score if period in {"development", "validation"} else later_score
            rows.append(
                {
                    "candidate": candidate,
                    "diversifier_family": "test",
                    "period": period,
                    "cagr": value,
                    "sharpe": value,
                    "calmar": value,
                    "average_turnover": 0.01,
                }
            )

    selection = select_training_winners(pd.DataFrame(rows))

    assert selection.iloc[0]["candidate"] == "stable"


def test_current_allocation_separates_executed_holding_from_latest_target() -> None:
    index = pd.to_datetime(["2026-11-20", "2026-11-23"])
    columns = ["MSFT", "XLV", "BIL"]
    result = BacktestResult(
        name="challenger",
        equity=pd.Series([100.0, 100.0], index=index),
        returns=pd.Series([0.0, 0.0], index=index),
        gross_returns=pd.Series([0.0, 0.0], index=index),
        weights=pd.DataFrame(
            [[0.70, 0.0, 0.30], [0.70, 0.0, 0.30]],
            index=index,
            columns=columns,
        ),
        target_weights=pd.DataFrame(
            [[0.70, 0.0, 0.30], [0.175, 0.525, 0.30]],
            index=index,
            columns=columns,
        ),
        turnover=pd.Series([0.0, 0.0], index=index),
        transaction_costs=pd.Series([0.0, 0.0], index=index),
    )
    strategy = StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["MSFT"],
        defensive_ticker="BIL",
        risk_repair_ai_diversifier_tickers=["XLV"],
    )
    candidate = TransitionCandidate(
        name="challenger",
        diversifier_family="defensive_sectors",
        policy="confirmed_80_25",
        lookback_days=63,
        strategy=strategy,
    )

    current = _current_allocations(
        {"challenger": result},
        {"challenger": candidate},
    ).iloc[0]

    assert current["executed_ai_share_of_risk"] == 1.0
    assert current["target_ai_share_of_risk"] == 0.25
    assert current["target_replacement_weight"] == 0.525
