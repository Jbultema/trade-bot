from __future__ import annotations

from trade_bot.research.strategy_naming import canonical_strategy_id, strategy_display_name


def test_strategy_display_name_humanizes_legacy_ids() -> None:
    label = strategy_display_name(
        "i43_active_ai_beta_sprint",
        family="active_ai_beta",
        phase="active_trading",
    )

    assert label == "Active AI Beta Sprint"


def test_canonical_strategy_id_builds_reset_era_slug() -> None:
    strategy_id = canonical_strategy_id(
        family="regime pulse",
        behavior="growth liquidity",
        variant="risk on core",
        number=1,
    )

    assert strategy_id == "regime_pulse_growth_liquidity_01_risk_on_core"
