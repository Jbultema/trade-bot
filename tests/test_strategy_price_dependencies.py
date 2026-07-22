from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from trade_bot.config import StrategyConfig, required_strategy_tickers
from trade_bot.research.approach_explorer import (
    _prepare_strategy_prices as approach_strategy_prices,
)
from trade_bot.research.approach_explorer import build_latest_approach_weights
from trade_bot.research.backtest_qc import _strategy_prices as qc_strategy_prices
from trade_bot.research.baselines import _strategy_prices as baseline_strategy_prices
from trade_bot.storage.warehouse import _candidate_strategy_prices as warehouse_strategy_prices
from trade_bot.strategies.momentum import build_strategy_weights


def test_native_risk_repair_uses_identical_price_inputs_and_weights_across_paths() -> None:
    strategy = _native_risk_repair_strategy()
    prices = _prices(required_strategy_tickers(strategy))

    baseline_prices = baseline_strategy_prices(prices, strategy)
    qc_prices = qc_strategy_prices(prices, strategy)
    warehouse_prices = warehouse_strategy_prices(prices, strategy)
    approach_prices, approach_strategy, missing = approach_strategy_prices(prices, strategy)

    expected_columns = required_strategy_tickers(strategy)
    assert list(baseline_prices.columns) == expected_columns
    assert list(qc_prices.columns) == expected_columns
    assert list(warehouse_prices.columns) == expected_columns
    assert list(approach_prices.columns) == expected_columns
    assert approach_strategy == strategy
    assert missing == []

    expected_weights = build_strategy_weights(baseline_prices, strategy)
    assert_frame_equal(build_strategy_weights(qc_prices, strategy), expected_weights)
    assert_frame_equal(build_strategy_weights(warehouse_prices, strategy), expected_weights)
    assert_frame_equal(
        build_strategy_weights(approach_prices, approach_strategy),
        expected_weights,
    )


@pytest.mark.parametrize(
    ("strategy_type", "expected"),
    [
        ("dip_reentry", {"HYG", "LQD", "RSP", "SPY"}),
        ("dip_reentry_overlay", {"HYG", "LQD", "RSP", "SPY"}),
        ("ai_risk_cycle_overlay", {"HYG", "LQD", "RSP", "SPY"}),
        (
            "sector_regime_rotation",
            {
                "HYG",
                "LQD",
                "RSP",
                "SPY",
                "SMH",
                "QQQ",
                "XLK",
                "DBC",
                "XLE",
                "XLI",
                "XLF",
                "TLT",
                "IEF",
                "SHY",
            },
        ),
    ],
)
def test_strategy_dependency_resolver_includes_hidden_signal_inputs(
    strategy_type: str,
    expected: set[str],
) -> None:
    strategy = StrategyConfig(
        type=strategy_type,  # type: ignore[arg-type]
        tickers=["QQQ", "NVDA"],
        satellite_tickers=["NVDA"] if strategy_type == "ai_risk_cycle_overlay" else [],
        defensive_ticker="BIL",
    )

    assert expected.issubset(required_strategy_tickers(strategy))


def test_dip_dependency_resolver_respects_disabled_confirmation_inputs() -> None:
    strategy = StrategyConfig(
        type="dip_reentry",
        tickers=["QQQ"],
        defensive_ticker="BIL",
        dip_credit_confirmation=False,
        dip_breadth_confirmation=False,
    )

    dependencies = required_strategy_tickers(strategy)
    assert "HYG" not in dependencies
    assert "LQD" not in dependencies
    assert "RSP" not in dependencies


def test_dip_reentry_paths_fail_closed_when_hidden_signal_is_missing() -> None:
    strategy = StrategyConfig(
        type="dip_reentry",
        tickers=["QQQ"],
        defensive_ticker="BIL",
    )
    prices = _prices(required_strategy_tickers(strategy)).drop(columns="HYG")

    with pytest.raises(KeyError, match="HYG"):
        baseline_strategy_prices(prices, strategy)
    with pytest.raises(KeyError, match="HYG"):
        qc_strategy_prices(prices, strategy)
    assert warehouse_strategy_prices(prices, strategy).empty
    approach_prices, approach_strategy, missing = approach_strategy_prices(prices, strategy)
    assert approach_prices.empty
    assert approach_strategy == strategy
    assert missing == ["HYG"]


@pytest.mark.parametrize("unusable_input", ["absent", "all_nan"])
def test_native_risk_repair_does_not_run_when_a_signal_input_is_unusable(
    unusable_input: str,
) -> None:
    strategy = _native_risk_repair_strategy()
    prices = _prices(required_strategy_tickers(strategy))
    if unusable_input == "absent":
        prices = prices.drop(columns="HYG")
    else:
        prices.loc[:, "HYG"] = float("nan")

    with pytest.raises(KeyError, match="HYG"):
        baseline_strategy_prices(prices, strategy)
    with pytest.raises(KeyError, match="HYG"):
        qc_strategy_prices(prices, strategy)
    assert warehouse_strategy_prices(prices, strategy).empty

    approach_prices, approach_strategy, missing = approach_strategy_prices(prices, strategy)
    assert approach_prices.empty
    assert approach_strategy == strategy
    assert missing == ["HYG"]
    latest = build_latest_approach_weights(prices, strategy)
    assert latest.iloc[0]["ticker"] == "n/a"
    assert "HYG" in str(latest.iloc[0]["note"])


def test_native_risk_repair_does_not_run_with_a_terminally_stale_signal() -> None:
    strategy = _native_risk_repair_strategy()
    prices = _prices(required_strategy_tickers(strategy))
    prices.loc[prices.index[-6:], "HYG"] = float("nan")

    with pytest.raises(KeyError, match="HYG"):
        baseline_strategy_prices(prices, strategy)
    with pytest.raises(KeyError, match="HYG"):
        qc_strategy_prices(prices, strategy)
    assert warehouse_strategy_prices(prices, strategy).empty

    approach_prices, approach_strategy, unusable = approach_strategy_prices(prices, strategy)
    assert approach_prices.empty
    assert approach_strategy == strategy
    assert unusable == ["HYG"]


def _native_risk_repair_strategy() -> StrategyConfig:
    return StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["QQQ", "SMH", "SOXX", "NVDA"],
        lookback_days=63,
        skip_days=5,
        top_n=2,
        defensive_ticker="BIL",
        min_return=0.025,
        ranking_metric="risk_adjusted_return",
        weighting="risk_adjusted_score",
        volatility_lookback_days=63,
        trend_filter_days=None,
        max_asset_weight=0.35,
        risk_repair_defensive_cap=0.85,
        risk_repair_defensive_release=0.15,
        risk_repair_ai_soft_cap=0.85,
        risk_repair_ai_soft_threshold=0.90,
        risk_repair_ai_excess_destination="diversifier_mix",
        risk_repair_ai_diversifier_tickers=["SPY", "RSP", "GLD", "TLT"],
    )


def _prices(tickers: list[str]) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=320)
    day = pd.Series(range(len(index)), index=index, dtype=float)
    frame = pd.DataFrame(index=index)
    for offset, ticker in enumerate(tickers, start=1):
        drift = 0.00005 * offset
        cycle = ((day + offset) % (13 + offset % 5) - 6.0) / 10_000.0
        frame[ticker] = 100.0 * (1.0 + drift + cycle).cumprod()
    return frame
