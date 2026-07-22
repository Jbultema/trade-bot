from __future__ import annotations

from trade_bot.config import BotConfig, DataConfig, ExecutionConfig, StrategyConfig
from trade_bot.research.i111_candidates import build_i111_candidates
from trade_bot.research.i111_risk_repair import (
    build_i111_risk_repair_candidates,
    default_i111_repair_specs,
)


def test_i111_candidate_roster_includes_configured_primary_and_upside_variants() -> None:
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
    )
    config = BotConfig(
        data=DataConfig(start="2020-01-01", end="2021-01-01"),
        execution=ExecutionConfig(),
        primary_strategy="i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145",
        universe={"core": ["SPY", "QQQ", "BIL"]},
        strategies={
            "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145": primary,
            "i111_reentry_vol_target_fast_21d_no_trend_vol18_guard12": primary.model_copy(
                update={"min_return": 0.02}
            ),
            "other_strategy": primary,
        },
    )

    candidates = build_i111_candidates(config)
    compatibility_candidates = build_i111_risk_repair_candidates(config)
    names = [candidate.name for candidate in candidates]

    assert "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145" in names
    assert "i111_reentry_vol_target_fast_21d_no_trend_vol18_guard12" in names
    assert "other_strategy" not in names
    assert "r19_min025_vol185_guard17_mult65" in names
    assert [candidate.name for candidate in compatibility_candidates] == names


def test_i111_repair_grid_covers_ai_caps_and_defensive_relief() -> None:
    names = {spec.name for spec in default_i111_repair_specs()}

    assert "ai_cap45_bil" in names
    assert "ai_cap35_bil" in names
    assert "defensive_relief_cap75" in names
    assert "defensive_relief_cap65" in names
    assert "defensive_relief_cap55" in names
    assert "ai_cap45_plus_relief65" in names
    assert len(names) == 9
