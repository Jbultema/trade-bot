from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import load_config
from trade_bot.dashboard.monitoring import (
    _monitoring_display_frame,
    _monitoring_drift_envelope_frame,
    _monitoring_start_cohorts,
)
from trade_bot.dashboard.research_lab import (
    _factor_contribution_waterfall_figure,
    _make_decision_timeline_figure,
    _signal_ablation_heatmap_frame,
)
from trade_bot.dashboard.risk_scenarios import (
    _scenario_bucket_history_figure,
    _scenario_driver_score_figure,
    _scenario_history_from_lattice,
    _scenario_history_insights,
    _scenario_history_scope,
    _scenario_horizon_differentiation_frame,
    _scenario_horizon_differentiation_read,
    _scenario_named_history_figure,
    _scenario_probability_heatmap_figure,
    _scenario_probability_stack_figure,
)
from trade_bot.dashboard.strategy_candidates import (
    outcome_candidate_scorecards,
    outcome_strategy_option_frame,
)
from trade_bot.DEFAULTS import DEFAULT_CONFIG_PATH
from trade_bot.research.evaluation_contract import (
    build_strategy_evaluation_contract,
    evaluation_contract_sha256,
)


def test_signal_ablation_heatmap_flips_turnover_direction() -> None:
    marginal_tests = pd.DataFrame(
        [
            {
                "signal_label": "Credit",
                "delta_cagr": 0.02,
                "delta_max_drawdown": 0.03,
                "delta_calmar": 0.10,
                "delta_reentry_score": 0.20,
                "delta_average_turnover": 0.04,
                "delta_left_tail_regime_return": 0.05,
            },
            {
                "signal_label": "Credit",
                "delta_cagr": 0.04,
                "delta_max_drawdown": 0.01,
                "delta_calmar": 0.20,
                "delta_reentry_score": 0.10,
                "delta_average_turnover": 0.02,
                "delta_left_tail_regime_return": 0.03,
            },
        ]
    )

    heatmap = _signal_ablation_heatmap_frame(marginal_tests)

    assert float(heatmap.loc["Credit", "delta_cagr"]) == 0.03
    assert float(heatmap.loc["Credit", "delta_average_turnover"]) == -0.03


def test_factor_contribution_waterfall_uses_factor_return_contributions() -> None:
    factor_view = pd.DataFrame(
        [
            {"label": "Market beta", "return_contribution": 0.08},
            {"label": "Residual strategy behavior", "return_contribution": 0.04},
            {"label": "Rates beta", "return_contribution": -0.01},
        ]
    )

    figure = _factor_contribution_waterfall_figure(factor_view)

    assert len(figure.data) == 1
    assert "Explained + residual" in list(figure.data[0].x)
    assert list(figure.data[0].y)[-1] == 0.11


def test_outcome_frontier_includes_latest_runtime_snapshot_candidates() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0], "QQQ": [100.0, 102.0], "BIL": [100.0, 100.01]},
        index=pd.bdate_range("2026-01-02", periods=2),
    )
    baseline_run = SimpleNamespace(
        prices=prices,
        metrics=pd.DataFrame(
            [
                {
                    "name": "runtime_frontier_candidate",
                    "cagr": 0.22,
                    "max_drawdown": -0.20,
                    "calmar": 1.10,
                    "average_turnover": 0.05,
                },
                {
                    "name": "buy_hold_spy",
                    "cagr": 0.10,
                    "max_drawdown": -0.55,
                    "calmar": 0.18,
                    "average_turnover": 0.0,
                },
                {
                    "name": "buy_hold_qqq",
                    "cagr": 0.15,
                    "max_drawdown": -0.53,
                    "calmar": 0.28,
                    "average_turnover": 0.0,
                },
            ]
        ).set_index("name"),
        window_summary=pd.DataFrame(
            [
                {
                    "name": "runtime_frontier_candidate",
                    "window": "1y",
                    "worst_cagr": -0.18,
                    "positive_window_rate": 0.72,
                },
                {
                    "name": "runtime_frontier_candidate",
                    "window": "3y",
                    "worst_cagr": 0.02,
                    "positive_window_rate": 0.90,
                },
                {
                    "name": "buy_hold_spy",
                    "window": "1y",
                    "worst_cagr": -0.45,
                    "positive_window_rate": 0.65,
                },
                {
                    "name": "buy_hold_qqq",
                    "window": "1y",
                    "worst_cagr": -0.42,
                    "positive_window_rate": 0.68,
                },
            ]
        ).set_index(["name", "window"]),
    )
    configured = load_config(DEFAULT_CONFIG_PATH)
    bot_config = SimpleNamespace(
        strategies={"runtime_frontier_candidate": object()},
        execution=configured.execution,
        data=configured.data,
    )
    contract_hash = evaluation_contract_sha256(
        build_strategy_evaluation_contract(bot_config, prices)
    )
    experiment_scorecards = pd.DataFrame(
        [
            {
                "strategy": "runtime_frontier_candidate",
                "source": "canonical_experiment_replay",
                "cagr": 0.22,
                "max_drawdown": -0.18,
                "worst_1y_cagr": -0.18,
                "worst_3y_cagr": 0.02,
                "evaluation_contract_sha256": contract_hash,
            },
            {
                "strategy": "older_experiment_candidate",
                "source": "experiment_scorecard",
                "cagr": 0.14,
                "max_drawdown": -0.20,
                "evaluation_contract_sha256": contract_hash,
            }
        ]
    )

    scorecards = outcome_candidate_scorecards(
        baseline_run=baseline_run,
        bot_config=bot_config,
        experiment_scorecards=experiment_scorecards,
    )

    assert set(scorecards["strategy"]) == {
        "runtime_frontier_candidate",
        "older_experiment_candidate",
    }
    runtime = scorecards[scorecards["strategy"].eq("runtime_frontier_candidate")].iloc[0]
    assert runtime["source"] == "canonical_experiment_replay"
    assert runtime["worst_1y_cagr"] == pytest.approx(-0.18)
    assert runtime["worst_3y_cagr"] == pytest.approx(0.02)

    option_frame = outcome_strategy_option_frame(
        baseline_run=baseline_run,
        bot_config=bot_config,
        experiment_scorecards=experiment_scorecards,
    )
    option_runtime = option_frame[
        option_frame["strategy"].eq("runtime_frontier_candidate")
    ].iloc[0]
    assert option_runtime["wealth_multiple_vs_spy"] > 1.0
    assert option_runtime["wealth_multiple_vs_qqq"] > 1.0
    assert option_runtime["worst_1y_cagr"] == pytest.approx(-0.18)
    assert option_runtime["worst_3y_cagr"] == pytest.approx(0.02)

    stale = experiment_scorecards.copy()
    stale.loc[stale.index[-1], "evaluation_contract_sha256"] = "stale-contract"
    with pytest.raises(ValueError, match="Mixed-version strategy metrics"):
        outcome_candidate_scorecards(
            baseline_run=baseline_run,
            bot_config=bot_config,
            experiment_scorecards=stale,
        )


def test_decision_timeline_figure_marks_key_allocation_events() -> None:
    dates = pd.bdate_range("2026-01-02", periods=8)
    equity = pd.Series([100, 102, 101, 99, 103, 104, 106, 105], index=dates)
    weights = pd.DataFrame(
        {
            "QQQ": [0.8, 0.8, 0.4, 0.4, 0.7, 0.7, 0.7, 0.7],
            "BIL": [0.2, 0.2, 0.6, 0.6, 0.3, 0.3, 0.3, 0.3],
        },
        index=dates,
    )
    result = BacktestResult(
        name="candidate",
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        gross_returns=equity.pct_change().fillna(0.0),
        weights=weights,
        target_weights=weights,
        turnover=weights.diff().abs().sum(axis=1).fillna(0.0),
        transaction_costs=pd.Series(0.0, index=dates),
    )
    events = pd.DataFrame(
        [
            {
                "event": "De-risking move",
                "date": dates[2].date().isoformat(),
                "signal": "De-risking move: risk -40.0%, defensive +40.0%, total move 80.0%.",
                "inferred_driver": "Inferred off-ramp from reconstructed weights.",
                "total_change": 0.8,
                "risk_weight_change": -0.4,
                "defensive_weight_change": 0.4,
                "top_adds": "BIL +40%",
                "top_reductions": "QQQ -40%",
                "forward_return_1m": 0.04,
                "risk_weight_at_event": 0.4,
                "drawdown_at_event": -0.02,
                "forward_return_3m": 0.06,
            }
        ]
    )

    figure = _make_decision_timeline_figure(
        result,
        events,
        landmark_frame=pd.DataFrame(),
        defensive_ticker="BIL",
        start=dates[0],
        end=dates[-1],
    )

    assert any(trace.name == "decision events" for trace in figure.data)


def test_scenario_probability_and_driver_figures_render_from_current_state_frames() -> None:
    scenario_lattice = pd.DataFrame(
        [
            {"horizon": "1m", "risk_bucket": "risk_on", "probability": 0.55},
            {"horizon": "1m", "risk_bucket": "risk_off", "probability": 0.25},
            {"horizon": "1m", "risk_bucket": "transition", "probability": 0.20},
            {"horizon": "3m", "risk_bucket": "risk_on", "probability": 0.40},
            {"horizon": "3m", "risk_bucket": "transition", "probability": 0.60},
        ]
    )
    scenario_drivers = pd.DataFrame(
        [
            {"driver": "AI leadership", "score": 0.75, "evidence": "QQQ/SPY firm"},
            {"driver": "Credit stress", "score": -0.20, "evidence": "HYG/LQD soft"},
        ]
    )

    probability_figure = _scenario_probability_stack_figure(scenario_lattice)
    driver_figure = _scenario_driver_score_figure(scenario_drivers)

    assert len(probability_figure.data) == 3
    assert probability_figure.layout.legend.y < 0
    assert probability_figure.layout.margin.b >= 90
    assert len(driver_figure.data) == 1
    assert list(driver_figure.data[0].y)[0] == "AI leadership"


def test_scenario_horizon_audit_flags_flat_bucket_probabilities() -> None:
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "scenario": "Broad risk-on",
                "risk_bucket": "risk_on",
                "probability": risk_on_probability,
            }
            for horizon, risk_on_probability in [
                ("1w", 0.40),
                ("1m", 0.41),
                ("3m", 0.40),
                ("6m", 0.42),
            ]
        ]
        + [
            {
                "horizon": horizon,
                "scenario": "Choppy transition",
                "risk_bucket": "transition",
                "probability": 1.0 - risk_on_probability,
            }
            for horizon, risk_on_probability in [
                ("1w", 0.40),
                ("1m", 0.41),
                ("3m", 0.40),
                ("6m", 0.42),
            ]
        ]
    )

    audit = _scenario_horizon_differentiation_frame(scenario_lattice)
    read = _scenario_horizon_differentiation_read(scenario_lattice)

    assert not audit.empty
    assert float(audit["horizon_spread"].max()) < 0.03
    assert "nearly flat" in read


def test_scenario_horizon_heatmap_shows_named_scenario_movement() -> None:
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": "1w",
                "scenario": "Policy whipsaw",
                "risk_bucket": "transition",
                "probability": 0.50,
            },
            {
                "horizon": "6m",
                "scenario": "Policy whipsaw",
                "risk_bucket": "transition",
                "probability": 0.10,
            },
            {
                "horizon": "1w",
                "scenario": "Credit repair",
                "risk_bucket": "risk_on",
                "probability": 0.10,
            },
            {
                "horizon": "6m",
                "scenario": "Credit repair",
                "risk_bucket": "risk_on",
                "probability": 0.50,
            },
        ]
    )

    audit = _scenario_horizon_differentiation_frame(scenario_lattice).set_index("risk_bucket")
    heatmap = _scenario_probability_heatmap_figure(scenario_lattice)

    assert float(audit.loc["transition", "horizon_spread"]) == 0.40
    assert len(heatmap.data) == 1
    assert "Policy whipsaw" in list(heatmap.data[0].y)


def test_scenario_history_helpers_surface_risk_pressure_changes() -> None:
    first_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "scenario": "Broad risk-on",
                "risk_bucket": "risk_on",
                "probability": 0.50,
            },
            {
                "horizon": "1m",
                "scenario": "Choppy transition",
                "risk_bucket": "transition",
                "probability": 0.30,
            },
            {
                "horizon": "1m",
                "scenario": "Oil shock",
                "risk_bucket": "risk_off",
                "probability": 0.20,
            },
        ]
    )
    second_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "scenario": "Broad risk-on",
                "risk_bucket": "risk_on",
                "probability": 0.35,
            },
            {
                "horizon": "1m",
                "scenario": "Choppy transition",
                "risk_bucket": "transition",
                "probability": 0.35,
            },
            {
                "horizon": "1m",
                "scenario": "Oil shock",
                "risk_bucket": "risk_off",
                "probability": 0.30,
            },
        ]
    )
    history = pd.concat(
        [
            _scenario_history_from_lattice(
                first_lattice,
                market_date="2026-07-01",
                created_at_utc="2026-07-01T14:00:00+00:00",
                run_id="first",
            ),
            _scenario_history_from_lattice(
                second_lattice,
                market_date="2026-07-02",
                created_at_utc="2026-07-02T14:00:00+00:00",
                run_id="second",
            ),
        ],
        ignore_index=True,
    )

    scoped = _scenario_history_scope(history, "Latest per market date")
    insights = _scenario_history_insights(scoped, "1m")

    assert scoped["history_time"].nunique() == 2
    assert "Risk pressure rising" in set(insights["read"])
    assert any("risk-off changed +10.0%" in detail for detail in insights["detail"])


def test_scenario_history_figures_render_bucket_and_named_views() -> None:
    history = pd.concat(
        [
            _scenario_history_from_lattice(
                pd.DataFrame(
                    [
                        {
                            "horizon": "1m",
                            "scenario": "Risk-on",
                            "risk_bucket": "risk_on",
                            "probability": 0.60,
                        },
                        {
                            "horizon": "1m",
                            "scenario": "Transition",
                            "risk_bucket": "transition",
                            "probability": 0.25,
                        },
                        {
                            "horizon": "1m",
                            "scenario": "Risk-off",
                            "risk_bucket": "risk_off",
                            "probability": 0.15,
                        },
                    ]
                ),
                market_date="2026-07-01",
                created_at_utc="2026-07-01T14:00:00+00:00",
                run_id="first",
            ),
            _scenario_history_from_lattice(
                pd.DataFrame(
                    [
                        {
                            "horizon": "1m",
                            "scenario": "Risk-on",
                            "risk_bucket": "risk_on",
                            "probability": 0.45,
                        },
                        {
                            "horizon": "1m",
                            "scenario": "Transition",
                            "risk_bucket": "transition",
                            "probability": 0.35,
                        },
                        {
                            "horizon": "1m",
                            "scenario": "Risk-off",
                            "risk_bucket": "risk_off",
                            "probability": 0.20,
                        },
                    ]
                ),
                market_date="2026-07-02",
                created_at_utc="2026-07-02T14:00:00+00:00",
                run_id="second",
            ),
        ],
        ignore_index=True,
    )
    scoped = _scenario_history_scope(history, "Latest per market date")

    bucket_figure = _scenario_bucket_history_figure(scoped)
    named_figure = _scenario_named_history_figure(scoped, top_n=2)

    assert len(bucket_figure.data) == 3
    assert len(named_figure.data) == 2


def test_monitoring_drift_envelope_classifies_forward_drawdown_status() -> None:
    frame = pd.DataFrame(
        [
            {
                "window_role": "champion",
                "start_date": "2026-01-01",
                "monitoring_days": 180,
                "strategy_name": "steady",
                "forward_status": "in_line",
                "valuation_date": "2026-06-30",
                "drawdown": -0.02,
                "snapshot_max_drawdown": -0.20,
            },
            {
                "window_role": "challenger",
                "start_date": "2026-06-01",
                "monitoring_days": 29,
                "strategy_name": "stressed",
                "forward_status": "behind_benchmark",
                "valuation_date": "2026-06-30",
                "drawdown": -0.18,
                "snapshot_max_drawdown": -0.20,
            },
            {
                "window_role": "challenger",
                "start_date": "2026-06-01",
                "monitoring_days": 29,
                "strategy_name": "breached",
                "forward_status": "behind_benchmark",
                "valuation_date": "2026-06-30",
                "drawdown": -0.22,
                "snapshot_max_drawdown": -0.20,
            },
        ]
    )

    envelope = _monitoring_drift_envelope_frame(frame).set_index("strategy_name")

    assert envelope.loc["steady", "envelope_status"] == "inside"
    assert envelope.loc["stressed", "envelope_status"] == "review"
    assert envelope.loc["breached", "envelope_status"] == "breach"
    assert envelope.loc["steady", "start_date"] == "2026-01-01"
    assert envelope.loc["stressed", "monitoring_days"] == 29


def test_monitoring_display_frame_preserves_duplicate_strategy_start_cohorts() -> None:
    frame = pd.DataFrame(
        [
            {
                "window_role": "challenger",
                "strategy_name": "same_strategy",
                "start_date": "2026-01-01",
                "valuation_date": "2026-07-07",
                "account": "paper",
            },
            {
                "window_role": "challenger",
                "strategy_name": "same_strategy",
                "start_date": "2026-06-01",
                "valuation_date": "2026-07-07",
                "account": "paper",
            },
            {
                "window_role": "champion",
                "strategy_name": "champion_strategy",
                "start_date": "2026-06-01",
                "valuation_date": "2026-07-07",
                "account": "paper",
            },
        ]
    )

    cohorts = _monitoring_start_cohorts(frame)
    all_rows = _monitoring_display_frame(frame, start_cohort="All starts")
    june_rows = _monitoring_display_frame(frame, start_cohort="2026-06-01")

    assert cohorts == ["2026-01-01", "2026-06-01"]
    assert len(all_rows[all_rows["strategy_name"].eq("same_strategy")]) == 2
    assert set(june_rows["start_date"]) == {"2026-06-01"}
    assert set(june_rows["strategy_name"]) == {"same_strategy", "champion_strategy"}
    assert june_rows["monitoring_days"].notna().all()
