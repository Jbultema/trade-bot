from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.research.forward_simulation import (
    REGIME_BUCKETS,
    ForwardSimulationConfig,
    build_regime_return_library,
    scenario_bucket_probabilities,
    simulate_regime_conditioned_paths,
    summarize_forward_simulation,
)


def test_scenario_bucket_probabilities_normalize_current_rollup() -> None:
    scenario_outlook = pd.DataFrame(
        [
            {"risk_bucket": "risk_off_then_relief", "probability": 0.20},
            {"risk_bucket": "transition", "probability": 0.25},
            {"risk_bucket": "risk_on_fragile", "probability": 0.15},
            {"risk_bucket": "risk_on", "probability": 0.40},
        ]
    )

    probabilities = scenario_bucket_probabilities(scenario_outlook)

    assert set(probabilities.index) == set(REGIME_BUCKETS)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert probabilities["risk_off"] == pytest.approx(0.20)
    assert probabilities["transition"] == pytest.approx(0.25)
    assert probabilities["risk_on_fragile"] == pytest.approx(0.15)
    assert probabilities["risk_on"] == pytest.approx(0.40)


def test_regime_return_library_labels_known_buckets() -> None:
    returns = pd.Series(
        [0.002] * 70
        + [-0.025] * 15
        + [0.004] * 70
        + [-0.012, 0.018] * 25
        + [0.001] * 60
    )
    config = ForwardSimulationConfig(min_regime_observations=1)

    library = build_regime_return_library(returns, config=config)

    assert not library.empty
    assert set(library["regime"]).issubset(set(REGIME_BUCKETS))
    assert {"risk_off", "risk_on"}.issubset(set(library["regime"]))


def test_simulate_regime_conditioned_paths_returns_distribution() -> None:
    returns = pd.Series(
        [0.003] * 80
        + [-0.02] * 20
        + [0.004] * 80
        + [-0.01, 0.02] * 30
        + [0.001] * 60
    )
    scenario_outlook = pd.DataFrame(
        [
            {"risk_bucket": "risk_off", "probability": 0.30},
            {"risk_bucket": "transition", "probability": 0.35},
            {"risk_bucket": "risk_on_fragile", "probability": 0.10},
            {"risk_bucket": "risk_on", "probability": 0.25},
        ]
    )
    config = ForwardSimulationConfig(
        horizon_years=1,
        trading_days_per_year=20,
        paths=25,
        block_days=5,
        starting_account_value=100.0,
        annual_contribution=10.0,
        random_seed=7,
        min_regime_observations=1,
    )

    paths = simulate_regime_conditioned_paths(
        returns,
        scenario_outlook=scenario_outlook,
        config=config,
    )
    summary = summarize_forward_simulation(paths, config=config)

    assert len(paths) == 25
    assert paths["terminal_wealth"].notna().all()
    assert paths["max_drawdown"].le(0).all()
    assert summary["paths"] == 25
    assert summary["terminal_wealth_p50"] is not None
    assert summary["severe_drawdown_probability"] is not None


def test_forward_simulation_empty_inputs_return_empty_summary() -> None:
    paths = simulate_regime_conditioned_paths(pd.Series(dtype=float))
    summary = summarize_forward_simulation(paths)

    assert paths.empty
    assert summary["paths"] == 0
    assert summary["terminal_wealth_p50"] is None
