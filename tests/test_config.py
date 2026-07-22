from __future__ import annotations

from pytest import approx, raises

from trade_bot.config import (
    AllocationPolicyConfig,
    StrategyConfig,
    apply_excluded_ticker_policy_to_strategy,
    configured_tickers,
    load_config,
    required_strategy_tickers,
)
from trade_bot.DEFAULTS import DEFAULT_CONFIG_PATH, DEFAULT_REBALANCE, DEFAULT_SIGNAL_LAG_DAYS


def test_allocation_utility_profiles_set_distinct_tail_tolerances() -> None:
    growth = AllocationPolicyConfig(utility_profile="growth")
    balanced = AllocationPolicyConfig(utility_profile="balanced_asymmetric")
    preservation = AllocationPolicyConfig(utility_profile="capital_preservation")

    assert growth.normal_tail_loss_limit > balanced.normal_tail_loss_limit
    assert balanced.normal_tail_loss_limit > preservation.normal_tail_loss_limit
    assert growth.catastrophic_stress_loss_limit > balanced.catastrophic_stress_loss_limit
    assert balanced.catastrophic_stress_loss_limit > preservation.catastrophic_stress_loss_limit


def test_scenario_authority_fails_closed_without_calibration() -> None:
    with raises(ValueError, match="requires provisional or validated calibration"):
        AllocationPolicyConfig(scenario_sizing_authority=0.1)


def test_risk_timing_authority_fails_closed_without_calibration() -> None:
    with raises(ValueError, match="Risk-timing allocation authority"):
        AllocationPolicyConfig(risk_timing_sizing_authority=0.1)


def test_macro_authority_fails_closed_on_revised_history() -> None:
    with raises(ValueError, match="point-in-time or first-release"):
        AllocationPolicyConfig(
            macro_sizing_authority=1.0,
            macro_calibration_status="validated",
            macro_data_vintage_status="revised_history",
        )


def test_validated_point_in_time_macro_authority_is_allowed() -> None:
    policy = AllocationPolicyConfig(
        macro_sizing_authority=1.0,
        macro_calibration_status="validated",
        macro_data_vintage_status="point_in_time",
    )

    assert policy.macro_sizing_authority == 1.0


def test_baseline_execution_config_matches_default_cadence() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    assert DEFAULT_REBALANCE == "W-WED"
    assert config.execution.rebalance == DEFAULT_REBALANCE
    assert config.execution.signal_lag_days == DEFAULT_SIGNAL_LAG_DAYS


def test_default_operable_momentum_strategies_include_global_equity_choices() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    expected_global_choices = {"VEA", "VWO", "VGK", "EWJ"}

    for strategy_name in [
        "dual_momentum_core",
        "vol_target_dual_momentum",
        "drawdown_managed_dual_momentum",
    ]:
        strategy = config.strategies[strategy_name]
        assert expected_global_choices.issubset(set(strategy.tickers))


def test_configured_tickers_include_native_risk_repair_dependencies() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    strategy = config.strategies["i111_native_risk_repair_guard17_relief85_ai85_div"]

    assert set(required_strategy_tickers(strategy)).issubset(configured_tickers(config))


def test_load_config_applies_hard_ticker_exclusions(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        data:
          start: "2020-01-01"
        execution: {}
        universe:
          ai_beta:
            - SPY
            - ORCL
            - QQQ
        strategies:
          fixed:
            type: fixed_allocation
            tickers:
              - SPY
              - ORCL
            allocation_weights:
              SPY: 0.5
              ORCL: 0.5
          cycle:
            type: ai_risk_cycle_overlay
            tickers:
              - SPY
              - ORCL
              - QQQ
            satellite_tickers:
              - ORCL
              - QQQ
            defensive_ticker: BIL
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert "ORCL" not in config.universe["ai_beta"]
    assert "ORCL" not in config.strategies["fixed"].tickers
    assert "ORCL" not in config.strategies["cycle"].tickers
    assert "ORCL" not in config.strategies["cycle"].satellite_tickers
    assert config.strategies["fixed"].allocation_weights == {"SPY": approx(1.0)}
    assert "ORCL" not in configured_tickers(config)


def test_hard_ticker_exclusions_apply_to_risk_repair_diversifiers() -> None:
    strategy = StrategyConfig(
        type="dual_momentum_risk_repair",
        tickers=["QQQ", "SMH"],
        defensive_ticker="BIL",
        risk_repair_ai_diversifier_tickers=["SPY", "ORCL", "GLD"],
    )

    filtered = apply_excluded_ticker_policy_to_strategy(strategy)

    assert filtered.risk_repair_ai_diversifier_tickers == ["SPY", "GLD"]
    assert "ORCL" not in required_strategy_tickers(filtered)
