from __future__ import annotations

import pandas as pd

from trade_bot.config import StrategyConfig
from trade_bot.research.native_i111_risk_repair import (
    default_native_risk_repair_specs,
)
from trade_bot.strategies.momentum import (
    _apply_risk_repair_ai_cap,
    _apply_risk_repair_defensive_relief,
    _drawdown_component,
    _relative_component,
    _return_component,
    _risk_repair_momentum_mix,
    _trend_component,
    build_strategy_weights,
)


def test_native_risk_repair_releases_defensive_weight_in_constructive_regime() -> None:
    dates = pd.bdate_range("2024-01-01", periods=160)
    trend = pd.Series(range(len(dates)), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "QQQ": 100.0 + trend,
            "SMH": 100.0 + trend * 0.8,
            "NVDA": 100.0 + trend * 0.7,
            "MSFT": 100.0 + trend * 0.6,
            "SPY": 100.0 + trend * 0.5,
            "RSP": 100.0 + trend * 0.45,
            "HYG": 80.0 + trend * 0.03,
            "LQD": 100.0 + trend * 0.01,
            "BIL": 100.0 + trend * 0.001,
        },
        index=dates,
    )
    strategy = StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["QQQ", "SMH", "NVDA", "MSFT"],
        lookback_days=21,
        skip_days=0,
        top_n=2,
        defensive_ticker="BIL",
        min_return=0.50,
        trend_filter_days=None,
        max_asset_weight=None,
        risk_repair_defensive_cap=0.75,
        risk_repair_defensive_release=1.0,
    )

    weights = build_strategy_weights(prices, strategy)

    assert weights["BIL"].iloc[-1] == 0.75
    assert weights[["QQQ", "SMH", "NVDA", "MSFT"]].iloc[-1].sum() == 0.25


def test_native_ai_cap_only_reduces_ai_exposure_during_stress() -> None:
    dates = pd.bdate_range("2024-01-01", periods=180)
    up = pd.Series(range(len(dates)), index=dates, dtype=float)
    down = pd.Series(range(len(dates), 0, -1), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "NVDA": 100.0 + up,
            "AVGO": 100.0 + up * 0.9,
            "QQQ": 220.0 + down * 0.2,
            "SMH": 200.0 + down * 0.4,
            "SPY": 180.0 + down * 0.1,
            "RSP": 100.0 + up * 0.05,
            "HYG": 90.0 + down * 0.02,
            "LQD": 90.0 + up * 0.02,
            "BIL": 100.0 + up * 0.001,
        },
        index=dates,
    )
    strategy = StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["NVDA", "AVGO"],
        lookback_days=21,
        skip_days=0,
        top_n=2,
        defensive_ticker="BIL",
        min_return=-1.0,
        trend_filter_days=None,
        max_asset_weight=None,
        risk_repair_ai_soft_cap=0.60,
        risk_repair_ai_soft_threshold=0.50,
    )

    weights = build_strategy_weights(prices, strategy)

    assert weights[["NVDA", "AVGO"]].iloc[-1].sum() <= 0.600001
    assert weights["BIL"].iloc[-1] >= 0.399999


def test_native_ai_cap_can_rotate_excess_to_diversifiers() -> None:
    dates = pd.bdate_range("2024-01-01", periods=180)
    up = pd.Series(range(len(dates)), index=dates, dtype=float)
    down = pd.Series(range(len(dates), 0, -1), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "NVDA": 100.0 + up,
            "AVGO": 100.0 + up * 0.9,
            "QQQ": 220.0 + down * 0.2,
            "SMH": 200.0 + down * 0.4,
            "SPY": 100.0 + up * 0.2,
            "RSP": 100.0 + up * 0.15,
            "GLD": 100.0 + up * 0.1,
            "TLT": 100.0 + up * 0.05,
            "HYG": 90.0 + down * 0.02,
            "LQD": 90.0 + up * 0.02,
            "BIL": 100.0 + up * 0.001,
        },
        index=dates,
    )
    strategy = StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["NVDA", "AVGO"],
        lookback_days=21,
        skip_days=0,
        top_n=2,
        defensive_ticker="BIL",
        min_return=-1.0,
        trend_filter_days=None,
        max_asset_weight=None,
        risk_repair_ai_soft_cap=0.60,
        risk_repair_ai_soft_threshold=0.50,
        risk_repair_ai_excess_destination="diversifier_mix",
    )

    weights = build_strategy_weights(prices, strategy)

    assert weights[["NVDA", "AVGO"]].iloc[-1].sum() <= 0.600001
    assert weights[["SPY", "RSP", "GLD", "TLT"]].iloc[-1].sum() >= 0.399999
    assert weights["BIL"].iloc[-1] < 0.01


def test_risk_repair_does_not_buy_negative_momentum_diversifiers() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    decline = pd.Series(range(len(dates)), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": 120.0 - decline * 0.20,
            "RSP": 110.0 - decline * 0.15,
        },
        index=dates,
    )

    mix = _risk_repair_momentum_mix(
        prices,
        ["SPY", "RSP"],
        lookback_days=21,
        top_n=2,
    )

    assert mix.iloc[-1].sum() == 0.0


def test_ai_cap_sends_unallocatable_diversifier_excess_to_defensive_cash() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    decline = pd.Series(range(len(dates)), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "NVDA": 100.0 + decline,
            "SPY": 120.0 - decline * 0.20,
            "RSP": 110.0 - decline * 0.15,
            "BIL": 100.0 + decline * 0.001,
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {"NVDA": 0.8, "SPY": 0.0, "RSP": 0.0, "BIL": 0.2},
        index=dates,
    )

    repaired = _apply_risk_repair_ai_cap(
        prices,
        weights,
        ai_stress_score=pd.Series(1.0, index=dates),
        defensive_ticker="BIL",
        soft_cap=0.5,
        hard_cap=None,
        soft_threshold=0.5,
        hard_threshold=0.9,
        cap_basis="portfolio",
        excess_destination="diversifier_mix",
        diversifier_tickers=["SPY", "RSP"],
        lookback_days=21,
    )

    assert repaired.loc[dates[-1], ["SPY", "RSP"]].sum() == 0.0
    assert repaired.loc[dates[-1], "NVDA"] == 0.5
    assert repaired.loc[dates[-1], "BIL"] == 0.5


def test_defensive_relief_keeps_cash_when_no_risk_asset_has_positive_momentum() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    decline = pd.Series(range(len(dates)), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": 120.0 - decline * 0.20,
            "RSP": 110.0 - decline * 0.15,
            "BIL": 100.0 + decline * 0.001,
        },
        index=dates,
    )
    weights = pd.DataFrame({"SPY": 0.0, "RSP": 0.0, "BIL": 1.0}, index=dates)

    repaired = _apply_risk_repair_defensive_relief(
        prices,
        weights,
        risk_tickers=["SPY", "RSP"],
        defensive_ticker="BIL",
        constructive_score=pd.Series(1.0, index=dates),
        ai_stress_score=pd.Series(0.0, index=dates),
        floor=0.25,
        defensive_cap=0.75,
        release=1.0,
        lookback_days=21,
        top_n=2,
        hard_stress_threshold=0.9,
    )

    assert repaired.loc[dates[-1], ["SPY", "RSP"]].sum() == 0.0
    assert repaired.loc[dates[-1], "BIL"] == 1.0


def test_native_ai_cap_can_limit_ai_share_of_active_risk_sleeve() -> None:
    dates = pd.bdate_range("2024-01-01", periods=180)
    up = pd.Series(range(len(dates)), index=dates, dtype=float)
    down = pd.Series(range(len(dates), 0, -1), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "NVDA": 100.0 + up,
            "AVGO": 100.0 + up * 0.9,
            "QQQ": 220.0 + down * 0.2,
            "SMH": 200.0 + down * 0.4,
            "SPY": 180.0 + down * 0.1,
            "RSP": 100.0 + up * 0.05,
            "HYG": 90.0 + down * 0.02,
            "LQD": 90.0 + up * 0.02,
            "BIL": 100.0 + up * 0.001,
        },
        index=dates,
    )
    strategy = StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["NVDA", "AVGO"],
        lookback_days=21,
        skip_days=0,
        top_n=2,
        defensive_ticker="BIL",
        min_return=-1.0,
        trend_filter_days=None,
        max_asset_weight=None,
        risk_repair_defensive_cap=0.75,
        risk_repair_defensive_release=1.0,
        risk_repair_ai_soft_cap=0.60,
        risk_repair_ai_soft_threshold=0.50,
        risk_repair_ai_cap_basis="risk_sleeve",
    )

    weights = build_strategy_weights(prices, strategy)

    ai_weight = weights[["NVDA", "AVGO"]].iloc[-1].sum()
    assert ai_weight <= 0.600001
    assert weights["BIL"].iloc[-1] >= 0.399999


def test_default_native_specs_cover_relief_ai_and_combined_variants() -> None:
    names = {spec.name for spec in default_native_risk_repair_specs()}

    assert "relief_cap75_rel25" in names
    assert "ai_soft75_s085" in names
    assert "balanced_relief75_ai75" in names


def test_risk_repair_components_are_neutral_until_inputs_are_live() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "SPY": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5],
        },
        index=dates,
    )

    trend = _trend_component(prices, "QQQ", 5)
    absolute_return = _return_component(prices, "QQQ", 3, 0.0)
    relative_return = _relative_component(prices, "QQQ", "SPY", 3, 0.0)
    drawdown = _drawdown_component(prices, "QQQ", 5, -0.05)

    assert trend.iloc[:4].eq(0.5).all()
    assert absolute_return.iloc[:3].eq(0.5).all()
    assert relative_return.iloc[:3].eq(0.5).all()
    assert drawdown.iloc[:4].eq(0.5).all()
    assert trend.iloc[-1] == 1.0
    assert absolute_return.iloc[-1] == 1.0
    assert relative_return.iloc[-1] == 1.0
    assert drawdown.iloc[-1] == 0.0
