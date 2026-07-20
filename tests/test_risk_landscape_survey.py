from __future__ import annotations

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.risk_landscape_survey import (
    _architecture_variant_results,
    build_signal_family_rankings,
)


def test_signal_family_rankings_group_predictive_signals(tmp_path) -> None:
    path = tmp_path / "signal_rank.csv"
    pd.DataFrame(
        {
            "signal": ["health_qqq_drawdown", "cycle_qqq_unwind", "portfolio_max_stress_loss"],
            "predictive_score": [0.4, 0.3, 0.2],
            "spearman_to_break_severity": [0.5, 0.4, 0.3],
            "event_auc": [0.8, 0.7, 0.6],
            "risk_direction": ["higher_is_riskier"] * 3,
        }
    ).to_csv(path, index=False)

    rankings = build_signal_family_rankings(path)

    assert rankings.iloc[0]["signal_family"] == "market_structure"
    assert set(rankings["signal_family"]) == {
        "market_structure",
        "cycle_tracker",
        "portfolio_stress",
    }


def test_architecture_variants_return_named_backtests() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "QQQ": [100.0 + i * 0.1 for i in range(260)],
            "SPY": [100.0 + i * 0.05 for i in range(260)],
            "SMH": [100.0 + i * 0.15 for i in range(260)],
            "HYG": [100.0 + i * 0.01 for i in range(260)],
            "LQD": [100.0 + i * 0.005 for i in range(260)],
            "BIL": [100.0 + i * 0.002 for i in range(260)],
            "GLD": [100.0 + i * 0.01 for i in range(260)],
            "TLT": [100.0 - i * 0.002 for i in range(260)],
        },
        index=dates,
    )
    weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    weights["QQQ"] = 1.0
    returns = pd.Series(0.0, index=dates)
    result = BacktestResult(
        name="base",
        equity=pd.Series(100.0, index=dates),
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=dates),
        transaction_costs=pd.Series(0.0, index=dates),
    )

    variants = _architecture_variant_results(result, prices, transaction_cost_bps=0.0)

    assert "permanent_15pct_defensive_basket" in variants
    assert "hybrid_confirmed_stress_basket" in variants
    assert all(not variant.returns.empty for variant in variants.values())
