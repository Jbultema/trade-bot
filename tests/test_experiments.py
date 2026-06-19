from __future__ import annotations

import json

import pandas as pd

from trade_bot.config import StrategyConfig
from trade_bot.research.experiments import (
    DecisionSanityConfig,
    ScenarioSizingConfig,
    apply_decision_sanity_overlay,
    apply_scenario_position_sizing,
    build_experiment_scorecard,
    generate_iteration_candidates,
)
from trade_bot.strategies.momentum import build_strategy_weights


def test_iteration_one_has_bounded_candidate_batch() -> None:
    candidates = generate_iteration_candidates(1)

    assert 3 <= len(candidates) <= 10
    assert len({candidate.name for candidate in candidates}) == len(candidates)


def test_broad_iterations_cover_multiple_research_families() -> None:
    candidates = generate_iteration_candidates(2)

    assert 3 <= len(candidates) <= 10
    assert {"risk_adjusted_momentum", "off_ramp", "ai_infrastructure"}.issubset(
        {candidate.family for candidate in candidates}
    )


def test_later_iterations_evolve_previous_candidate_configs() -> None:
    strategy = StrategyConfig(
        type="dual_momentum",
        tickers=["SPY", "QQQ", "IWM", "GLD", "TLT"],
        lookback_days=126,
        skip_days=21,
        top_n=2,
        defensive_ticker="BIL",
        min_return=0.0,
    )
    previous_scorecards = pd.DataFrame(
        {
            "iteration": [3],
            "strategy": ["parent_strategy"],
            "family": ["core_cross_asset"],
            "role": ["candidate_core"],
            "promotion_decision": ["evolve_next_iteration"],
            "promotion_score": [0.7],
        }
    )
    previous_candidates = pd.DataFrame(
        {
            "strategy": ["parent_strategy"],
            "strategy_json": [json.dumps(strategy.model_dump(mode="json"))],
        }
    )

    candidates = generate_iteration_candidates(
        4,
        previous_scorecards=previous_scorecards,
        previous_candidates=previous_candidates,
    )

    assert 3 <= len(candidates) <= 10
    assert {candidate.parent for candidate in candidates} == {"parent_strategy"}
    assert all(candidate.phase == "deep" for candidate in candidates)


def test_experiment_scorecard_marks_left_tail_rejects() -> None:
    candidates = generate_iteration_candidates(1)
    names = [candidate.name for candidate in candidates]
    metrics = pd.DataFrame(
        {
            "cagr": [0.10 for _ in names],
            "sharpe": [0.8 for _ in names],
            "sortino": [1.1 for _ in names],
            "max_drawdown": [-0.20 for _ in names],
            "calmar": [0.5 for _ in names],
            "average_turnover": [0.05 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )
    metrics.loc[names[0], "max_drawdown"] = -0.50
    window_summary = pd.DataFrame(
        {
            "worst_cagr": [-0.01 for _ in names for _ in range(3)],
            "positive_window_rate": [0.8 for _ in names for _ in range(3)],
        },
        index=pd.MultiIndex.from_product(
            [names, ["1y", "3y", "5y"]],
            names=["strategy", "window"],
        ),
    )

    benchmark_metrics = pd.DataFrame(
        {
            "cagr": [0.08, 0.12],
            "max_drawdown": [-0.30, -0.40],
            "calmar": [0.27, 0.30],
        },
        index=["benchmark_spy", "benchmark_qqq"],
    )

    scorecard = build_experiment_scorecard(
        candidates,
        metrics,
        window_summary,
        benchmark_metrics=benchmark_metrics,
    )

    assert scorecard.loc[names[0], "promotion_decision"] == "reject_left_tail"
    assert "promotion_score" in scorecard.columns
    assert "family" in scorecard.columns
    assert round(float(scorecard.loc[names[1], "excess_cagr_vs_spy"]), 6) == 0.02


def test_scorecard_includes_walk_forward_and_regime_robustness() -> None:
    candidates = generate_iteration_candidates(1)
    names = [candidate.name for candidate in candidates]
    metrics = pd.DataFrame(
        {
            "cagr": [0.10 for _ in names],
            "sharpe": [0.8 for _ in names],
            "sortino": [1.1 for _ in names],
            "max_drawdown": [-0.20 for _ in names],
            "calmar": [0.5 for _ in names],
            "average_turnover": [0.05 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )
    window_summary = pd.DataFrame(
        {
            "worst_cagr": [-0.01 for _ in names for _ in range(3)],
            "positive_window_rate": [0.8 for _ in names for _ in range(3)],
        },
        index=pd.MultiIndex.from_product(
            [names, ["1y", "3y", "5y"]],
            names=["strategy", "window"],
        ),
    )
    regime_summary = pd.DataFrame(
        {
            "worst_regime_cagr": [-0.08 for _ in names],
            "worst_regime_return": [-0.05 for _ in names],
            "left_tail_regime_cagr": [-0.06 for _ in names],
            "left_tail_regime_return": [-0.04 for _ in names],
            "transition_regime_hit_rate": [0.6 for _ in names],
            "regime_positive_rate": [0.7 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )
    walk_forward_summary = pd.DataFrame(
        {
            "walk_forward_median_cagr": [0.09 for _ in names],
            "walk_forward_worst_cagr": [-0.04 for _ in names],
            "walk_forward_positive_rate": [0.7 for _ in names],
            "walk_forward_median_calmar": [0.45 for _ in names],
        },
        index=pd.Index(names, name="name"),
    )
    operability_metrics = pd.DataFrame(
        {
            "material_trade_days_per_year": [10.0 for _ in names],
            "mean_days_between_material_trades": [25.0 for _ in names],
            "median_material_turnover": [0.08 for _ in names],
            "max_single_day_turnover": [0.40 for _ in names],
            "operability_score": [0.85 for _ in names],
            "operability_label": ["paper_operable" for _ in names],
        },
        index=pd.Index(names, name="strategy"),
    )
    transition_metrics = pd.DataFrame(
        {
            "average_risk_weight": [0.75 for _ in names],
            "min_risk_weight": [0.25 for _ in names],
            "latest_risk_weight": [0.80 for _ in names],
            "low_risk_day_rate": [0.10 for _ in names],
            "median_reentry_days": [18.0 for _ in names],
            "reentry_cycles": [3 for _ in names],
            "reentry_score": [0.75 for _ in names],
            "risk_cycle_label": ["risk_off_then_reenters" for _ in names],
        },
        index=pd.Index(names, name="strategy"),
    )

    scorecard = build_experiment_scorecard(
        candidates,
        metrics,
        window_summary,
        regime_summary=regime_summary,
        walk_forward_summary=walk_forward_summary,
        operability_metrics=operability_metrics,
        transition_metrics=transition_metrics,
    )

    assert "robustness_score" in scorecard.columns
    assert "walk_forward_positive_rate" in scorecard.columns
    assert "left_tail_regime_return" in scorecard.columns
    assert "left_tail_regime_cagr" in scorecard.columns
    assert "monitoring_readiness_score" in scorecard.columns
    assert "operability_score" in scorecard.columns
    assert "risk_cycle_label" in scorecard.columns
    assert set(scorecard["operability_label"]) == {"paper_operable"}


def test_scenario_position_sizing_moves_stress_residual_to_defensive_ticker() -> None:
    index = pd.bdate_range("2026-01-01", periods=90)
    prices = pd.DataFrame(
        {
            "SPY": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "QQQ": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "RSP": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "HYG": pd.Series(range(100, 10, -1), index=index, dtype=float),
            "LQD": pd.Series(range(100, 190), index=index, dtype=float),
            "VIXY": pd.Series(range(50, 140), index=index, dtype=float),
            "UUP": pd.Series(range(50, 140), index=index, dtype=float),
            "BIL": pd.Series(100.0, index=index),
        }
    )
    target_weights = pd.DataFrame(
        {"SPY": 1.0, "BIL": 0.0},
        index=index,
    )

    adjusted = apply_scenario_position_sizing(
        target_weights,
        prices,
        ScenarioSizingConfig(profile="test", stress_multiplier=0.2, min_multiplier=0.1),
        defensive_ticker="BIL",
    )

    assert adjusted["SPY"].iloc[-1] < 0.8
    assert adjusted["BIL"].iloc[-1] > 0.2
    assert round(float(adjusted.iloc[-1].sum()), 8) == 1.0



def test_decision_sanity_overlay_caps_unconfirmed_defensive_add() -> None:
    index = pd.bdate_range("2026-01-01", periods=90)
    prices = _sanity_prices(index, confirmed_break=False)
    base = pd.DataFrame({"SPY": 1.0, "BIL": 0.0}, index=index)
    adjusted = pd.DataFrame({"SPY": 0.20, "BIL": 0.80}, index=index)

    capped = apply_decision_sanity_overlay(
        base,
        adjusted,
        prices,
        DecisionSanityConfig(profile="test", max_defensive_add=0.25),
        defensive_ticker="BIL",
    )

    assert capped["BIL"].iloc[-1] <= 0.25 + 1e-9
    assert capped["SPY"].iloc[-1] >= 0.75 - 1e-9
    assert round(float(capped.iloc[-1].sum()), 8) == 1.0


def test_decision_sanity_overlay_allows_confirmed_defensive_add() -> None:
    index = pd.bdate_range("2026-01-01", periods=90)
    prices = _sanity_prices(index, confirmed_break=True)
    base = pd.DataFrame({"SPY": 1.0, "BIL": 0.0}, index=index)
    adjusted = pd.DataFrame({"SPY": 0.20, "BIL": 0.80}, index=index)

    capped = apply_decision_sanity_overlay(
        base,
        adjusted,
        prices,
        DecisionSanityConfig(profile="test", max_defensive_add=0.25),
        defensive_ticker="BIL",
    )

    assert capped["BIL"].iloc[-1] == adjusted["BIL"].iloc[-1]
    assert capped["SPY"].iloc[-1] == adjusted["SPY"].iloc[-1]


def test_decision_sanity_ablation_iteration_has_raw_and_capped_pairs() -> None:
    candidates = generate_iteration_candidates(77)

    assert len(candidates) == 8
    assert {candidate.role for candidate in candidates} == {"sanity_ablation"}
    assert any(candidate.decision_sanity is None for candidate in candidates)
    assert any(candidate.decision_sanity is not None for candidate in candidates)
    capped = [candidate for candidate in candidates if candidate.decision_sanity is not None]
    assert all(candidate.scenario_sizing is not None for candidate in candidates)
    assert all(candidate.parent.startswith("i77_sanity_raw_") for candidate in capped)


def test_decision_sanity_tuning_iteration_has_multiple_profiles() -> None:
    candidates = generate_iteration_candidates(78)

    assert len(candidates) == 18
    raw = [candidate for candidate in candidates if candidate.decision_sanity is None]
    tuned = [candidate for candidate in candidates if candidate.decision_sanity is not None]
    assert len(raw) == 3
    assert len(tuned) == 15
    assert {candidate.decision_sanity.profile for candidate in tuned if candidate.decision_sanity} == {
        "modest_cap",
        "confirmation_cap",
        "wide_cap",
        "strict_gate",
        "loose_gate",
    }
    assert all(candidate.parent.startswith("i78_sanity_raw_") for candidate in tuned)


def test_paper_readiness_iteration_tests_low_churn_and_metered_reentry() -> None:
    candidates = generate_iteration_candidates(79)

    assert len(candidates) == 10
    assert {candidate.role for candidate in candidates} == {"paper_readiness_candidate"}
    assert {candidate.phase for candidate in candidates} == {"paper_readiness_tuning"}
    assert any(candidate.name.endswith("low_churn") for candidate in candidates)
    assert any(candidate.name.endswith("metered_reentry") for candidate in candidates)
    assert all(candidate.parent for candidate in candidates)


def _sanity_prices(index: pd.DatetimeIndex, *, confirmed_break: bool) -> pd.DataFrame:
    trend = pd.Series(range(len(index)), index=index, dtype=float)
    if not confirmed_break:
        return pd.DataFrame(
            {
                "SPY": 100.0 + trend,
                "QQQ": 100.0 + trend,
                "RSP": 100.0 + trend,
                "HYG": 100.0 + trend * 0.5,
                "LQD": 100.0 + trend * 0.4,
                "VIXY": 100.0 - trend * 0.2,
                "UUP": 100.0,
                "BIL": 100.0,
            },
            index=index,
        )
    return pd.DataFrame(
        {
            "SPY": 120.0 - trend * 0.8,
            "QQQ": 125.0 - trend,
            "RSP": 120.0 - trend * 1.2,
            "HYG": 110.0 - trend * 0.7,
            "LQD": 100.0 + trend * 0.2,
            "VIXY": 80.0 + trend * 1.5,
            "UUP": 100.0 + trend * 0.3,
            "BIL": 100.0,
        },
        index=index,
    )


def test_macro_reset_iterations_use_human_readable_strategy_names() -> None:
    reset_families: set[str] = set()
    reset_names: list[str] = []

    for iteration in range(101, 106):
        candidates = generate_iteration_candidates(iteration)
        assert len(candidates) == 4
        assert all(not candidate.name[1:4].isdigit() for candidate in candidates)
        assert all(candidate.scenario_sizing is not None for candidate in candidates)
        reset_families.update(candidate.family for candidate in candidates)
        reset_names.extend(candidate.name for candidate in candidates)

    assert len(reset_names) == len(set(reset_names))
    assert {
        "regime_pulse_growth_liquidity",
        "growth_inflation_rotation",
        "positioning_crowding",
        "exposure_alignment_long_only",
        "integrated_operating_system",
    }.issubset(reset_families)
    assert "integrated_operating_system_01_retirement_core" in reset_names

def test_reference_portfolio_iteration_includes_explicit_policy_sizing() -> None:
    candidates = generate_iteration_candidates(41)

    assert len(candidates) == 9
    assert {candidate.family for candidate in candidates} == {"reference_portfolio"}
    assert {candidate.role for candidate in candidates} == {"reference_portfolio"}
    assert all(candidate.strategy.type == "fixed_allocation" for candidate in candidates)

    allocation_lookup = {
        candidate.name: candidate.strategy.allocation_weights for candidate in candidates
    }
    assert allocation_lookup["i41_ref_us_60_40"] == {"SPY": 0.60, "AGG": 0.40}
    assert allocation_lookup["i41_ref_all_weather"] == {
        "SPY": 0.30,
        "TLT": 0.40,
        "IEF": 0.15,
        "GLD": 0.075,
        "DBC": 0.075,
    }


def test_active_trading_iterations_are_bounded_and_short_horizon() -> None:
    for iteration in range(42, 50):
        candidates = generate_iteration_candidates(iteration)

        assert 3 <= len(candidates) <= 10
        assert len({candidate.name for candidate in candidates}) == len(candidates)
        assert all(
            candidate.name.startswith(f"i{iteration:02d}_active_") for candidate in candidates
        )
        assert {candidate.phase for candidate in candidates} == {"active_trading"}
        assert {candidate.role for candidate in candidates} == {"active_candidate"}
        assert all(candidate.strategy.defensive_ticker == "BIL" for candidate in candidates)
        assert all(
            candidate.strategy.type in {"dual_momentum", "absolute_momentum"}
            for candidate in candidates
        )
        assert any(
            candidate.strategy.lookback_days <= 42 or candidate.strategy.moving_average_days <= 63
            for candidate in candidates
        )
        assert any(candidate.scenario_sizing is not None for candidate in candidates)


def test_final_deep_wide_iterations_are_curated_and_human_executable() -> None:
    cached_universe = {
        "AAPL",
        "AGG",
        "AMZN",
        "ARCC",
        "ARKK",
        "AVGO",
        "BIL",
        "BITB",
        "BIZD",
        "BKLN",
        "BNO",
        "BOTZ",
        "BXSL",
        "CCJ",
        "CEG",
        "CLOU",
        "COWZ",
        "DBC",
        "EEM",
        "EFA",
        "ETHE",
        "EWC",
        "EWJ",
        "EWZ",
        "ETN",
        "FBTC",
        "FXE",
        "FXF",
        "FXY",
        "GEV",
        "GLD",
        "HYG",
        "IAU",
        "IBIT",
        "IEF",
        "IGV",
        "INDA",
        "IWM",
        "JAAA",
        "JBBB",
        "JNK",
        "KRE",
        "LQD",
        "MAIN",
        "META",
        "MOAT",
        "MSFT",
        "MTUM",
        "MUB",
        "NVDA",
        "NRG",
        "OBDC",
        "OIH",
        "ORCL",
        "PLTR",
        "PWR",
        "QQQ",
        "QUAL",
        "ROBO",
        "RSP",
        "SCHD",
        "SGOV",
        "SHY",
        "SKYY",
        "SMH",
        "SOXX",
        "SPHB",
        "SPLV",
        "SPY",
        "SRLN",
        "SVXY",
        "TAN",
        "TIP",
        "TLT",
        "TSLA",
        "UUP",
        "USFR",
        "USMV",
        "USO",
        "VCIT",
        "VCSH",
        "VEA",
        "VGK",
        "VIG",
        "VRT",
        "VTV",
        "VWO",
        "XBI",
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XOP",
    }

    for iteration in range(50, 55):
        candidates = generate_iteration_candidates(iteration)

        assert 3 <= len(candidates) <= 10
        assert len({candidate.name for candidate in candidates}) == len(candidates)
        assert all(
            candidate.name.startswith(f"i{iteration:02d}_final_") for candidate in candidates
        )
        assert {candidate.phase for candidate in candidates} == {"final_deep_dive"}
        assert {candidate.role for candidate in candidates} == {"final_candidate"}
        assert all(candidate.strategy.defensive_ticker == "BIL" for candidate in candidates)
        assert all(
            candidate.strategy.type in {"dual_momentum", "absolute_momentum"}
            for candidate in candidates
        )
        assert any(candidate.scenario_sizing is not None for candidate in candidates)
        assert all(
            set(candidate.strategy.tickers).issubset(cached_universe) for candidate in candidates
        )


def test_dip_reentry_strategy_meters_cash_into_confirmed_discount() -> None:
    index = pd.bdate_range("2020-01-01", periods=160)
    falling = list(range(100, 70, -1))
    basing = [70.0 for _ in range(30)]
    repairing = [70.0 + i * 0.8 for i in range(100)]
    spy_path = pd.Series(falling + basing + repairing, index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": spy_path,
            "QQQ": spy_path * 1.02,
            "RSP": spy_path * 1.01,
            "HYG": pd.Series(falling + basing + repairing, index=index, dtype=float),
            "LQD": pd.Series([100.0 for _ in index], index=index),
            "BIL": pd.Series([100.0 for _ in index], index=index),
        }
    )
    strategy = StrategyConfig(
        type="dip_reentry",
        tickers=["SPY", "QQQ", "RSP", "HYG", "LQD"],
        defensive_ticker="BIL",
        top_n=3,
        weighting="risk_adjusted_score",
        max_asset_weight=0.40,
        dip_lookback_days=63,
        dip_trigger_drawdown=-0.10,
        dip_deep_drawdown=-0.25,
        dip_recovery_days=10,
        dip_confirmation_days=3,
        dip_min_recovery_return=0.01,
        dip_starter_weight=0.20,
        dip_step_weight=0.20,
        dip_max_risk_weight=0.70,
        dip_volatility_ceiling=1.00,
    )

    weights = build_strategy_weights(prices, strategy)
    risk_weight = weights.drop(columns=["BIL"]).sum(axis=1)

    assert risk_weight.iloc[20] == 0.0
    assert risk_weight.max() > 0.15
    assert risk_weight.max() <= 0.70
    assert weights.loc[risk_weight.idxmax(), "BIL"] < 0.85
    assert round(float(weights.loc[risk_weight.idxmax()].sum()), 8) == 1.0


def test_dip_reentry_overlay_replaces_cash_after_confirmed_discount() -> None:
    index = pd.bdate_range("2020-01-01", periods=180)
    falling = list(range(100, 64, -1))
    basing = [64.0 for _ in range(44)]
    repairing = [64.0 + i * 0.9 for i in range(100)]
    spy_path = pd.Series(falling + basing + repairing, index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": spy_path,
            "QQQ": spy_path * 1.03,
            "RSP": spy_path * 1.01,
            "HYG": pd.Series(falling + basing + repairing, index=index, dtype=float),
            "LQD": pd.Series([100.0 for _ in index], index=index),
            "BIL": pd.Series([100.0 for _ in index], index=index),
        }
    )
    strategy = StrategyConfig(
        type="dip_reentry_overlay",
        tickers=["SPY", "QQQ", "RSP", "HYG", "LQD"],
        defensive_ticker="BIL",
        lookback_days=42,
        skip_days=5,
        top_n=2,
        min_return=0.04,
        ranking_metric="risk_adjusted_return",
        weighting="risk_adjusted_score",
        max_asset_weight=0.40,
        dip_lookback_days=63,
        dip_trigger_drawdown=-0.10,
        dip_deep_drawdown=-0.25,
        dip_recovery_days=10,
        dip_confirmation_days=3,
        dip_min_recovery_return=0.01,
        dip_starter_weight=0.22,
        dip_step_weight=0.20,
        dip_max_risk_weight=0.70,
        dip_volatility_ceiling=1.00,
    )

    weights = build_strategy_weights(prices, strategy)
    risk_weight = weights.drop(columns=["BIL"]).sum(axis=1)

    assert risk_weight.iloc[25] == 0.0
    assert risk_weight.max() > 0.20
    assert risk_weight.max() <= 1.0
    assert weights.loc[risk_weight.idxmax(), "BIL"] < 0.80
    assert round(float(weights.loc[risk_weight.idxmax()].sum()), 8) == 1.0


def test_dip_reentry_iterations_are_bounded_and_use_cached_tradeable_universe() -> None:
    cached_universe = {
        "AAPL",
        "AMZN",
        "ARCC",
        "ARKK",
        "AVGO",
        "BIL",
        "BIZD",
        "BKLN",
        "BOTZ",
        "BRK-B",
        "BXSL",
        "CCJ",
        "CEG",
        "COWZ",
        "DBC",
        "EEM",
        "EFA",
        "ETN",
        "EWC",
        "EWJ",
        "EWZ",
        "FBTC",
        "GEV",
        "GLD",
        "GOOGL",
        "HYG",
        "IAU",
        "IBIT",
        "IEF",
        "IGV",
        "INDA",
        "IWD",
        "IWM",
        "IYT",
        "JAAA",
        "JBBB",
        "JNK",
        "JPM",
        "KRE",
        "LQD",
        "MAIN",
        "MDY",
        "META",
        "MOAT",
        "MSFT",
        "MTUM",
        "NVDA",
        "NRG",
        "OBDC",
        "PWR",
        "QQQ",
        "QVAL",
        "QUAL",
        "RSP",
        "SCHD",
        "SGOV",
        "SHY",
        "SMH",
        "SOXX",
        "SPHB",
        "SPLV",
        "SPY",
        "SRLN",
        "SVXY",
        "TAN",
        "TIP",
        "TLT",
        "TSLA",
        "UUP",
        "USFR",
        "USMV",
        "USO",
        "VCIT",
        "VCSH",
        "VEA",
        "VFQY",
        "VGK",
        "VIG",
        "VRT",
        "VTV",
        "VWO",
        "XBI",
        "XHB",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
        "XRT",
    }

    for iteration in range(55, 61):
        candidates = generate_iteration_candidates(iteration)

        assert 3 <= len(candidates) <= 10
        assert len({candidate.name for candidate in candidates}) == len(candidates)
        assert all(candidate.name.startswith(f"i{iteration:02d}_dip_") for candidate in candidates)
        assert {candidate.phase for candidate in candidates} == {"dip_reentry"}
        assert {candidate.role for candidate in candidates} == {"reentry_candidate"}
        assert all(candidate.strategy.type == "dip_reentry" for candidate in candidates)
        assert all(candidate.strategy.defensive_ticker == "BIL" for candidate in candidates)
        assert all(
            set(candidate.strategy.tickers).issubset(cached_universe) for candidate in candidates
        )
        assert any(candidate.scenario_sizing is not None for candidate in candidates)



def test_dip_reentry_overlay_iterations_are_bounded_and_cash_redeployment_focused() -> None:
    cached_universe = {
        "AAPL",
        "AMZN",
        "ARCC",
        "ARKK",
        "AVGO",
        "BIL",
        "BIZD",
        "BKLN",
        "BOTZ",
        "BRK-B",
        "BXSL",
        "CCJ",
        "CEG",
        "COWZ",
        "DBC",
        "EEM",
        "EFA",
        "ETN",
        "EWC",
        "EWJ",
        "EWZ",
        "FBTC",
        "GEV",
        "GLD",
        "GOOGL",
        "HYG",
        "IAU",
        "IBIT",
        "IEF",
        "IGV",
        "INDA",
        "IWD",
        "IWM",
        "IYT",
        "JAAA",
        "JBBB",
        "JNK",
        "JPM",
        "KRE",
        "LQD",
        "MAIN",
        "MDY",
        "META",
        "MOAT",
        "MSFT",
        "MTUM",
        "NVDA",
        "NRG",
        "OBDC",
        "PWR",
        "QQQ",
        "QVAL",
        "QUAL",
        "RSP",
        "SCHD",
        "SGOV",
        "SHY",
        "SMH",
        "SOXX",
        "SPHB",
        "SPLV",
        "SPY",
        "SRLN",
        "SVXY",
        "TAN",
        "TIP",
        "TLT",
        "TSLA",
        "UUP",
        "USFR",
        "USMV",
        "USO",
        "VCIT",
        "VCSH",
        "VEA",
        "VFQY",
        "VGK",
        "VIG",
        "VRT",
        "VTV",
        "VWO",
        "XBI",
        "XHB",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
        "XRT",
    }

    for iteration in range(61, 66):
        candidates = generate_iteration_candidates(iteration)

        assert 3 <= len(candidates) <= 10
        assert len({candidate.name for candidate in candidates}) == len(candidates)
        assert all(
            candidate.name.startswith(f"i{iteration:02d}_dip_overlay_")
            for candidate in candidates
        )
        assert {candidate.phase for candidate in candidates} == {"dip_reentry_overlay"}
        assert {candidate.role for candidate in candidates} == {"reentry_overlay_candidate"}
        assert all(candidate.strategy.type == "dip_reentry_overlay" for candidate in candidates)
        assert all(candidate.strategy.defensive_ticker == "BIL" for candidate in candidates)
        assert all(
            set(candidate.strategy.tickers).issubset(cached_universe) for candidate in candidates
        )
        assert any(candidate.scenario_sizing is not None for candidate in candidates)




def test_sector_regime_rotation_routes_from_stress_to_sector_reentry() -> None:
    index = pd.bdate_range("2020-01-01", periods=240)
    base = pd.Series(
        [100.0] * 45
        + list(range(100, 70, -1))
        + [70.0] * 45
        + [70.0 + i * 0.42 for i in range(120)],
        index=index,
        dtype=float,
    )
    ai = pd.Series(
        [110.0] * 45
        + list(range(110, 50, -2))
        + [50.0] * 45
        + [50.0 + i * 0.90 for i in range(120)],
        index=index,
        dtype=float,
    )
    defensive = pd.Series([100.0 + i * 0.03 for i in range(240)], index=index, dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": base,
            "RSP": base * 1.01,
            "HYG": base,
            "LQD": defensive,
            "QQQ": ai,
            "SMH": ai * 1.04,
            "XLK": ai * 0.98,
            "XLE": base * 0.95,
            "XLI": base * 1.02,
            "XLF": base * 0.97,
            "XLV": defensive * 1.01,
            "XLU": defensive * 1.02,
            "TLT": defensive * 1.03,
            "IEF": defensive,
            "SHY": defensive * 0.99,
            "DBC": base * 0.90,
            "GLD": defensive * 1.04,
            "BIL": pd.Series([100.0 for _ in index], index=index),
        }
    )
    strategy = StrategyConfig(
        type="sector_regime_rotation",
        tickers=[
            "SPY",
            "RSP",
            "QQQ",
            "SMH",
            "XLK",
            "XLE",
            "XLI",
            "XLF",
            "XLV",
            "XLU",
            "GLD",
            "TLT",
            "IEF",
            "HYG",
            "LQD",
            "DBC",
        ],
        defensive_ticker="BIL",
        lookback_days=42,
        skip_days=5,
        top_n=4,
        min_return=0.0,
        ranking_metric="risk_adjusted_return",
        weighting="risk_adjusted_score",
        volatility_lookback_days=21,
        trend_filter_days=42,
        max_asset_weight=0.35,
        dip_lookback_days=63,
        dip_trigger_drawdown=-0.12,
        dip_deep_drawdown=-0.30,
        dip_recovery_days=10,
        dip_confirmation_days=3,
        dip_min_recovery_return=0.015,
        dip_starter_weight=0.25,
        dip_step_weight=0.35,
        dip_max_risk_weight=0.90,
        dip_volatility_ceiling=1.20,
        cycle_min_rebalance_change=0.02,
        cycle_max_step_change=0.40,
    )

    weights = build_strategy_weights(prices, strategy)
    equity_theme_weight = weights[["QQQ", "SMH", "XLK", "XLE", "XLI", "XLF"]].sum(axis=1)
    defensive_weight = weights[["BIL", "GLD", "TLT", "IEF", "LQD"]].sum(axis=1)

    assert round(float(weights.iloc[-1].sum()), 8) == 1.0
    assert defensive_weight.iloc[80] > equity_theme_weight.iloc[80]
    assert equity_theme_weight.iloc[-1] > equity_theme_weight.iloc[80]
    assert equity_theme_weight.max() <= 0.90


def test_sector_regime_iterations_are_diverse_and_operable() -> None:
    cached_universe = {
        "AAPL",
        "AGG",
        "ARCC",
        "ARKK",
        "BIL",
        "BIZD",
        "BKLN",
        "BNO",
        "BOTZ",
        "BXSL",
        "CCJ",
        "CEG",
        "CLOU",
        "COWZ",
        "DBA",
        "DBC",
        "EEM",
        "EFA",
        "ETN",
        "EWC",
        "EWJ",
        "EWZ",
        "GEV",
        "GLD",
        "HYG",
        "IAU",
        "IEF",
        "IGV",
        "INDA",
        "IWD",
        "IWF",
        "IWM",
        "IYT",
        "JAAA",
        "JBBB",
        "JNK",
        "KRE",
        "LQD",
        "MAIN",
        "MOAT",
        "MTUM",
        "NRG",
        "OBDC",
        "PWR",
        "QQQ",
        "QUAL",
        "ROBO",
        "RSP",
        "SCHD",
        "SGOV",
        "SHY",
        "SKYY",
        "SMH",
        "SOXX",
        "SPY",
        "SPLV",
        "SRLN",
        "TIP",
        "TLT",
        "UUP",
        "URA",
        "USFR",
        "USMV",
        "USO",
        "VCIT",
        "VCSH",
        "VEA",
        "VGK",
        "VIG",
        "VRT",
        "VTV",
        "VUG",
        "VWO",
        "XBI",
        "XHB",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
        "XME",
        "XOP",
        "XRT",
    }

    families: set[str] = set()
    for iteration in range(72, 77):
        candidates = generate_iteration_candidates(iteration)
        families.update(candidate.family for candidate in candidates)

        assert len(candidates) == 6
        assert len({candidate.name for candidate in candidates}) == len(candidates)
        assert all(candidate.name.startswith(f"i{iteration:02d}_sector_regime_") for candidate in candidates)
        assert {candidate.phase for candidate in candidates} == {"sector_regime_rotation"}
        assert {candidate.role for candidate in candidates} == {"sector_regime_candidate"}
        assert all(candidate.strategy.type == "sector_regime_rotation" for candidate in candidates)
        assert all(candidate.strategy.defensive_ticker == "BIL" for candidate in candidates)
        assert all(candidate.strategy.cycle_min_rebalance_change >= 0.04 for candidate in candidates)
        assert all(candidate.strategy.cycle_max_step_change <= 0.35 for candidate in candidates)
        assert all(
            set(candidate.strategy.tickers).issubset(cached_universe) for candidate in candidates
        )
        assert any(candidate.scenario_sizing is not None for candidate in candidates)

    assert len(families) >= 20

def test_ai_risk_cycle_overlay_reenters_ai_after_confirmed_repair() -> None:
    index = pd.bdate_range("2020-01-01", periods=220)
    core_path = pd.Series(
        [100.0] * 40 + list(range(100, 70, -1)) + [70.0] * 40 + [70.0 + i * 0.45 for i in range(110)],
        index=index,
        dtype=float,
    )
    ai_path = pd.Series(
        [120.0] * 40 + list(range(120, 60, -2)) + [60.0] * 40 + [60.0 + i * 1.0 for i in range(110)],
        index=index,
        dtype=float,
    )
    prices = pd.DataFrame(
        {
            "SPY": core_path,
            "RSP": core_path * 1.01,
            "HYG": core_path,
            "LQD": pd.Series([100.0 for _ in index], index=index),
            "QQQ": ai_path,
            "SMH": ai_path * 1.03,
            "MSFT": ai_path * 0.97,
            "BIL": pd.Series([100.0 for _ in index], index=index),
        }
    )
    strategy = StrategyConfig(
        type="ai_risk_cycle_overlay",
        tickers=["SPY", "RSP", "HYG", "LQD", "QQQ", "SMH", "MSFT"],
        satellite_tickers=["QQQ", "SMH", "MSFT"],
        defensive_ticker="BIL",
        lookback_days=42,
        skip_days=5,
        top_n=2,
        min_return=0.04,
        ranking_metric="risk_adjusted_return",
        weighting="risk_adjusted_score",
        max_asset_weight=0.45,
        dip_lookback_days=63,
        dip_trigger_drawdown=-0.15,
        dip_deep_drawdown=-0.35,
        dip_recovery_days=10,
        dip_confirmation_days=3,
        dip_min_recovery_return=0.015,
        dip_starter_weight=0.20,
        dip_step_weight=0.22,
        dip_max_risk_weight=0.80,
        dip_volatility_ceiling=1.20,
        cycle_satellite_max_weight=0.55,
        cycle_satellite_risk_on_weight=0.35,
        cycle_satellite_reentry_weight=0.70,
        cycle_min_rebalance_change=0.02,
        cycle_max_step_change=0.35,
    )

    weights = build_strategy_weights(prices, strategy)
    satellite_weight = weights[["QQQ", "SMH", "MSFT"]].sum(axis=1)

    assert satellite_weight.iloc[45] == 0.0
    assert satellite_weight.max() > 0.15
    assert satellite_weight.max() <= 0.55
    assert weights.loc[satellite_weight.idxmax(), "BIL"] < 0.85
    assert round(float(weights.loc[satellite_weight.idxmax()].sum()), 8) == 1.0


def test_ai_risk_cycle_iterations_are_diverse_and_human_operable() -> None:
    cached_universe = {
        "AAPL",
        "AMZN",
        "ARCC",
        "ARKK",
        "AVGO",
        "BIL",
        "BIZD",
        "BKLN",
        "BNO",
        "BOTZ",
        "BRK-B",
        "BXSL",
        "CCJ",
        "CEG",
        "COWZ",
        "DBC",
        "EEM",
        "EFA",
        "ETN",
        "EWC",
        "EWJ",
        "EWZ",
        "FBTC",
        "GEV",
        "GLD",
        "GOOGL",
        "HYG",
        "IAU",
        "IBIT",
        "IEF",
        "IGV",
        "INDA",
        "IWD",
        "IWM",
        "IYT",
        "JAAA",
        "JBBB",
        "JNK",
        "KRE",
        "LQD",
        "MAIN",
        "MDY",
        "META",
        "MOAT",
        "MSFT",
        "MTUM",
        "NVDA",
        "NRG",
        "OBDC",
        "PWR",
        "QQQ",
        "QUAL",
        "RSP",
        "SCHD",
        "SGOV",
        "SHY",
        "SMH",
        "SOXX",
        "SPHB",
        "SPLV",
        "SPY",
        "SRLN",
        "SVXY",
        "TAN",
        "TIP",
        "TLT",
        "TSLA",
        "UUP",
        "USFR",
        "USMV",
        "USO",
        "VCIT",
        "VCSH",
        "VEA",
        "VGK",
        "VIG",
        "VRT",
        "VTV",
        "VWO",
        "XBI",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
        "XOP",
    }

    families: set[str] = set()
    for iteration in range(66, 72):
        candidates = generate_iteration_candidates(iteration)
        families.update(candidate.family for candidate in candidates)

        assert len(candidates) == 6
        assert len({candidate.name for candidate in candidates}) == len(candidates)
        assert all(candidate.name.startswith(f"i{iteration:02d}_cycle_") for candidate in candidates)
        assert {candidate.phase for candidate in candidates} == {"ai_risk_cycle"}
        assert {candidate.role for candidate in candidates} == {"risk_cycle_candidate"}
        assert all(candidate.strategy.type == "ai_risk_cycle_overlay" for candidate in candidates)
        assert all(candidate.strategy.defensive_ticker == "BIL" for candidate in candidates)
        assert all(candidate.strategy.satellite_tickers for candidate in candidates)
        assert all(candidate.strategy.cycle_min_rebalance_change >= 0.025 for candidate in candidates)
        assert all(candidate.strategy.cycle_max_step_change <= 0.48 for candidate in candidates)
        assert all(
            set(candidate.strategy.tickers).issubset(cached_universe) for candidate in candidates
        )
        assert any(candidate.scenario_sizing is not None for candidate in candidates)

    assert len(families) >= 25
