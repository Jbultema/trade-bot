from __future__ import annotations

from collections import Counter

from trade_bot.config import (
    BotConfig,
    DataConfig,
    DrawdownControlConfig,
    ExecutionConfig,
    StrategyConfig,
    VolatilityTargetConfig,
)
from trade_bot.research.i111_frontier_search import (
    DEFAULT_FRONTIER_MAX_ITERATIONS,
    build_i111_frontier_candidates,
)


def test_frontier_candidate_grid_covers_five_research_topics() -> None:
    primary = StrategyConfig(
        type="dual_momentum",
        tickers=[
            "QQQ",
            "SMH",
            "SOXX",
            "IGV",
            "NVDA",
            "AVGO",
            "MSFT",
            "META",
            "AMZN",
            "PLTR",
        ],
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

    baselines, candidates = build_i111_frontier_candidates(config)

    topic_counts = Counter(candidate.primary_topic for candidate in candidates)
    assert sorted(baselines) == ["baseline_primary"]
    assert len(candidates) == DEFAULT_FRONTIER_MAX_ITERATIONS
    assert len({candidate.name for candidate in candidates}) == DEFAULT_FRONTIER_MAX_ITERATIONS
    assert topic_counts == {
        "gated_concentration": 50,
        "dynamic_guard": 50,
        "ai_health_score": 50,
        "crash_onset_mesh": 50,
        "two_model_router": 50,
    }
    assert {candidate.base_strategy.type for candidate in candidates} == {
        "dual_momentum_risk_repair"
    }
