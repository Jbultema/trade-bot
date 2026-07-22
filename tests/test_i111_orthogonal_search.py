from __future__ import annotations

from trade_bot.config import (
    BotConfig,
    DataConfig,
    DrawdownControlConfig,
    ExecutionConfig,
    StrategyConfig,
    VolatilityTargetConfig,
)
from trade_bot.research.i111_orthogonal_search import build_i111_orthogonal_candidates


def test_orthogonal_candidate_grid_caps_at_requested_combination_count() -> None:
    primary = StrategyConfig(
        type="dual_momentum",
        tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "AMZN", "PLTR"],
        defensive_ticker="BIL",
        lookback_days=63,
        skip_days=5,
        top_n=4,
        min_return=0.025,
        trend_filter_days=None,
        max_asset_weight=0.35,
        volatility_target=VolatilityTargetConfig(
            annualized_volatility=0.185,
            lookback_days=21,
            max_leverage=1.0,
        ),
        drawdown_control=DrawdownControlConfig(
            equity_lookback_days=84,
            max_drawdown=-0.145,
            risk_multiplier=0.55,
        ),
    )
    config = BotConfig(
        data=DataConfig(start="2020-01-01", end="2021-01-01"),
        execution=ExecutionConfig(),
        primary_strategy="primary",
        universe={"core": ["SPY", "QQQ", "BIL"]},
        strategies={"primary": primary},
    )

    baselines, candidates = build_i111_orthogonal_candidates(
        config,
        max_new_combinations=12,
    )

    assert sorted(baselines) == ["baseline_primary"]
    assert len(candidates) == 12
    assert len({candidate.name for candidate in candidates}) == 12
    assert {candidate.strategy.type for candidate in candidates} == {"dual_momentum_risk_repair"}
    assert {candidate.mechanism_family for candidate in candidates} >= {
        "signal_speed",
        "ranking_quality",
        "concentration_shape",
    }
