from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trade_bot.config import StrategyConfig
from trade_bot.research import strategy_decision
from trade_bot.research.strategy_decision import (
    StrategyDecisionUnavailableError,
    resolve_trade_decision_for_strategy,
)


def test_missing_named_operating_strategy_fails_instead_of_using_baseline() -> None:
    baseline = _baseline(results={})
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["QQQ"],
        defensive_ticker="BIL",
    )

    with pytest.raises(StrategyDecisionUnavailableError, match="missing_strategy"):
        resolve_trade_decision_for_strategy(
            baseline,  # type: ignore[arg-type]
            "missing_strategy",
            strategy_config=strategy,
        )


def test_alternate_strategy_uses_its_defensive_policy_without_primary_macro_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alternate_result = SimpleNamespace(name="alternate")
    baseline = _baseline(results={"alternate": alternate_result})
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["QQQ"],
        defensive_ticker="SHY",
    )
    captured: dict[str, object] = {}
    expected = SimpleNamespace(summary=pd.DataFrame())

    def fake_build_trade_decision(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(strategy_decision, "build_trade_decision", fake_build_trade_decision)

    resolved = resolve_trade_decision_for_strategy(
        baseline,  # type: ignore[arg-type]
        "alternate",
        strategy_config=strategy,
    )

    assert resolved is expected
    assert captured["defensive_ticker"] == "SHY"
    signal_inclusion = captured["signal_inclusion"]
    assert signal_inclusion.summary.iloc[0]["decision"] == "not_evaluated_for_strategy"
    assert "no allocation authority" in signal_inclusion.summary.iloc[0]["rationale"]


def test_operating_strategy_without_explicit_defensive_policy_is_unavailable() -> None:
    alternate_result = SimpleNamespace(name="alternate")
    baseline = _baseline(results={"alternate": alternate_result})
    strategy = StrategyConfig(type="fixed_allocation", tickers=["SPY"])

    with pytest.raises(StrategyDecisionUnavailableError, match="defensive ticker"):
        resolve_trade_decision_for_strategy(
            baseline,  # type: ignore[arg-type]
            "alternate",
            strategy_config=strategy,
        )


def _baseline(*, results: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        trade_decision=SimpleNamespace(
            summary=pd.DataFrame([{"strategy": "primary"}]),
        ),
        results=results,
        current_state=object(),
        event_risk=object(),
        news_monitor=object(),
        prices=pd.DataFrame(),
    )
