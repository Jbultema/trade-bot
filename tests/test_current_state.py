from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.current_state import build_current_state, vams_table


def test_vams_table_classifies_positive_and_negative_trends() -> None:
    index = pd.bdate_range("2024-01-01", periods=180)
    prices = pd.DataFrame(
        {
            "UP": [100.0 + value for value in range(180)],
            "DOWN": [280.0 - value for value in range(180)],
        },
        index=index,
    )

    vams = vams_table(prices, lookback_days=63, vol_days=21)

    assert vams.loc["UP", "vams_state"] == "bullish"
    assert vams.loc["DOWN", "vams_state"] == "bearish"


def test_build_current_state_produces_alerts_and_scenarios() -> None:
    index = pd.bdate_range("2024-01-01", periods=180)
    prices = pd.DataFrame(
        {
            "SPY": [100.0 + value for value in range(180)],
            "QQQ": [100.0 + value * 1.2 for value in range(180)],
            "RSP": [100.0 + value * 0.8 for value in range(180)],
            "IWM": [100.0 + value * 0.5 for value in range(180)],
            "MGC": [100.0 + value * 0.7 for value in range(180)],
            "HYG": [100.0 + value * 0.4 for value in range(180)],
            "LQD": [100.0 + value * 0.2 for value in range(180)],
            "GLD": [100.0 + value * 0.1 for value in range(180)],
            "TLT": [100.0 - value * 0.1 for value in range(180)],
            "VIXY": [100.0 - value * 0.2 for value in range(180)],
            "UUP": [100.0 - value * 0.1 for value in range(180)],
            "SPHB": [100.0 + value for value in range(180)],
            "SPLV": [100.0 + value * 0.3 for value in range(180)],
            "XLY": [100.0 + value * 0.7 for value in range(180)],
            "XLP": [100.0 + value * 0.2 for value in range(180)],
            "VTV": [100.0 + value * 0.5 for value in range(180)],
            "VUG": [100.0 + value * 0.6 for value in range(180)],
            "CPER": [100.0 + value * 0.4 for value in range(180)],
            "SMH": [100.0 + value * 1.1 for value in range(180)],
        },
        index=index,
    )
    returns = prices["SPY"].pct_change(fill_method=None).fillna(0.0)
    equity = 100.0 * (1.0 + returns).cumprod()
    weights = pd.DataFrame({"SPY": 1.0}, index=index)
    result = BacktestResult(
        name="demo",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )

    state = build_current_state(prices, {"demo": result}, preferred_strategy="demo")

    assert state.market_date == "2024-09-06"
    assert not state.strategy_alerts.empty
    assert not state.scenario_outlook.empty
    assert not state.scenario_lattice.empty
    assert not state.scenario_drivers.empty
    assert not state.signal_coverage.empty
    assert not state.confirmation_matrix.empty
    assert round(state.scenario_outlook["probability"].sum(), 6) == 1.0
    assert sorted(state.scenario_lattice["horizon"].unique()) == ["1m", "1w", "3m", "6m"]
    horizon_probability = state.scenario_lattice.groupby("horizon")["probability"].sum()
    assert horizon_probability.round(6).eq(1.0).all()
