from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_bot.research.factor_attribution import (
    build_factor_attribution,
    build_factor_decay_monitor,
    build_implementation_shortfall,
    build_ticket_shortfall_audit,
)


def _price_from_returns(returns: pd.Series, start: float = 100.0) -> pd.Series:
    return start * (1.0 + returns.fillna(0.0)).cumprod()


def test_factor_attribution_recovers_dominant_market_beta() -> None:
    dates = pd.bdate_range("2022-01-03", periods=160)
    market_returns = pd.Series(np.linspace(-0.01, 0.012, len(dates)), index=dates)
    qqq_returns = market_returns * 1.1
    noise = pd.Series(np.sin(np.arange(len(dates))) * 0.0001, index=dates)
    strategy_returns = 0.75 * market_returns + noise
    prices = pd.DataFrame(
        {
            "SPY": _price_from_returns(market_returns),
            "QQQ": _price_from_returns(qqq_returns),
        }
    )
    strategy_equity = _price_from_returns(strategy_returns, start=10_000)

    run = build_factor_attribution(
        strategy_equity,
        prices,
        factor_specs=(
            ("market_beta", "SPY", "Market beta", "Broad market."),
            ("qqq_beta", "QQQ", "QQQ beta", "Growth proxy."),
        ),
        min_observations=40,
    )

    market = run.factor_attribution.set_index("factor").loc["market_beta"]
    summary = run.summary.iloc[0]

    assert float(summary["factor_model_r_squared"]) > 0.95
    assert float(market["beta"]) > 0.30
    assert float(market["absolute_contribution_share"]) > 0.20
    assert run.factor_attribution["risk_contribution_pct"].sum() == pytest.approx(1.0)
    assert not np.allclose(
        run.factor_attribution["return_contribution"],
        run.factor_attribution["risk_contribution_pct"],
    )


def test_factor_decay_monitor_flags_large_beta_change() -> None:
    dates = pd.bdate_range("2022-01-03", periods=180)
    market_returns = pd.Series(0.001, index=dates)
    market_returns.iloc[::5] = -0.002
    low_beta = 0.25 * market_returns.iloc[:100]
    high_beta = 1.15 * market_returns.iloc[100:]
    strategy_returns = pd.concat([low_beta, high_beta])
    prices = pd.DataFrame({"SPY": _price_from_returns(market_returns)})
    strategy_equity = _price_from_returns(strategy_returns, start=10_000)

    decay = build_factor_decay_monitor(
        strategy_equity,
        prices,
        recent_lookback_days=70,
        min_observations=40,
    )
    market_decay = decay.set_index("factor").loc["market_beta"]

    assert bool(market_decay["drift_flag"])
    assert float(market_decay["recent_beta"]) > float(market_decay["full_beta"])


def test_implementation_shortfall_rebases_ideal_to_actual_window() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    ideal_equity = pd.Series([100, 101, 103, 104, 106], index=dates)
    actual = pd.DataFrame(
        {
            "valuation_date": dates[1:],
            "equity": [10_000, 10_080, 10_100, 10_200],
        }
    )

    shortfall = build_implementation_shortfall(ideal_equity, actual)
    row = shortfall.iloc[0]

    assert row["observations"] == 4
    assert float(row["ideal_final_equity"]) > float(row["actual_final_equity"])
    assert float(row["shortfall_return"]) < 0


def test_ticket_shortfall_audit_flags_unexecuted_and_band_slippage() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticket_id": "ticket-1",
                "decision_id": "decision-1",
                "mode": "paper",
                "account": "acct",
                "strategy_name": "strategy",
                "ticker": "QQQ",
                "side": "BUY",
                "status": "executed",
                "reference_price": 100.0,
                "limit_low": 99.0,
                "limit_high": 101.0,
                "target_notional": 1000.0,
                "min_notional": 900.0,
                "max_notional": 1100.0,
            },
            {
                "ticket_id": "ticket-2",
                "decision_id": "decision-1",
                "mode": "paper",
                "account": "acct",
                "strategy_name": "strategy",
                "ticker": "BIL",
                "side": "SELL",
                "status": "open",
                "reference_price": 90.0,
                "limit_low": 89.1,
                "limit_high": 90.9,
                "target_notional": -500.0,
                "min_notional": 450.0,
                "max_notional": 550.0,
            },
        ]
    )
    executions = pd.DataFrame(
        [
            {
                "recommendation_id": "ticket-1",
                "executed_at_utc": "2024-01-02T15:00:00Z",
                "price": 102.0,
                "notional": 1000.0,
            }
        ]
    )

    audit = build_ticket_shortfall_audit(tickets, executions).set_index("ticket_id")

    assert audit.loc["ticket-1", "execution_status"] == "executed"
    assert not bool(audit.loc["ticket-1", "inside_price_band"])
    assert audit.loc["ticket-2", "execution_status"] == "not_executed"
