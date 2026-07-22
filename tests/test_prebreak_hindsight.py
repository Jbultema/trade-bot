from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trade_bot.research.prebreak_hindsight import (
    DEFAULT_PREBREAK_LOOKBACK_DAYS,
    BubbleBreakWindow,
    _add_market_health_metrics,
    build_late_trigger_mesh,
    build_prebreak_snapshot_plan,
    current_best_signal_readout,
    deduplicate_prebreak_snapshot_plan,
    evaluate_staged_policy_variants,
    rank_predictive_signals,
    snapshot_signal_row,
    summarize_action_timing,
    summarize_hard_defense_attribution,
    summarize_staged_risk_behavior,
)


def test_market_health_metrics_exclude_structural_vixy_lifetime_drawdown() -> None:
    frame = pd.DataFrame(
        {
            "return_1m": {"SPY": 0.02, "VIXY": -0.10},
            "return_3m": {"SPY": 0.04, "VIXY": -0.25},
            "drawdown": {"SPY": -0.05, "VIXY": -0.9999},
        }
    )
    row: dict[str, object] = {}

    _add_market_health_metrics(row, frame)

    assert row["health_spy_drawdown"] == -0.05
    assert row["health_vixy_return_1m"] == -0.10
    assert row["health_vixy_return_3m"] == -0.25
    assert "health_vixy_drawdown" not in row


def test_prebreak_snapshot_plan_selects_weekly_dates_before_break() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-19")
    windows = (
        BubbleBreakWindow(
            name="test_break",
            break_date="2020-02-19",
            family="test",
            description="Synthetic break.",
        ),
    )

    plan = build_prebreak_snapshot_plan(
        dates,
        windows=windows,
        lookback_days=30,
        postbreak_days=0,
        weekly_frequency="W-WED",
    )

    assert not plan.empty
    assert plan["event_name"].unique().tolist() == ["test_break"]
    assert plan["market_date"].iloc[-1] == "2020-02-19"
    assert plan["days_to_break"].iloc[-1] == 0
    assert pd.to_datetime(plan["market_date"]).dt.to_period("W-WED").nunique() == len(plan)
    assert pd.to_datetime(plan["market_date"]).min() >= pd.Timestamp("2020-01-20")


def test_prebreak_snapshot_plan_can_include_first_month_after_break() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-03-20")
    windows = (
        BubbleBreakWindow(
            name="test_break",
            break_date="2020-02-19",
            family="test",
            description="Synthetic break.",
        ),
    )

    plan = build_prebreak_snapshot_plan(
        dates,
        windows=windows,
        lookback_days=14,
        postbreak_days=31,
        weekly_frequency="W-WED",
    )

    assert not plan.empty
    assert pd.to_datetime(plan["market_date"]).max() > pd.Timestamp("2020-02-19")
    assert plan["postbreak_snapshot"].any()
    assert plan["days_to_break"].min() < 0
    assert pd.to_datetime(plan["market_date"]).max() <= pd.Timestamp("2020-03-21")


def test_prebreak_snapshot_plan_preserves_non_weekly_break_session() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-03-20")
    windows = (
        BubbleBreakWindow(
            name="friday_break",
            break_date="2020-02-21",
            family="test",
            description="Synthetic Friday break.",
        ),
    )

    plan = build_prebreak_snapshot_plan(
        dates,
        windows=windows,
        lookback_days=14,
        postbreak_days=7,
        weekly_frequency="W-WED",
    )

    assert "2020-02-21" in set(plan["market_date"])
    break_row = plan[plan["market_date"].eq("2020-02-21")].iloc[0]
    assert break_row["days_to_break"] == 0


def test_prebreak_snapshot_plan_deduplicates_overlapping_event_dates() -> None:
    plan = pd.DataFrame(
        [
            {
                "event_name": "event_a",
                "family": "bubble",
                "break_date": "2020-02-19",
                "market_date": "2020-02-12",
                "days_to_break": 7,
                "postbreak_snapshot": False,
            },
            {
                "event_name": "event_b",
                "family": "crisis",
                "break_date": "2020-02-20",
                "market_date": "2020-02-12",
                "days_to_break": 8,
                "postbreak_snapshot": False,
            },
        ]
    )

    deduplicated = deduplicate_prebreak_snapshot_plan(plan)

    assert len(deduplicated) == 1
    assert deduplicated.iloc[0]["event_name"] == "event_a | event_b"
    assert deduplicated.iloc[0]["event_count"] == 2


def test_prebreak_default_lookback_extends_to_one_year() -> None:
    dates = pd.bdate_range("2019-02-01", "2020-02-19")
    windows = (
        BubbleBreakWindow(
            name="test_break",
            break_date="2020-02-19",
            family="test",
            description="Synthetic break.",
        ),
    )

    plan = build_prebreak_snapshot_plan(dates, windows=windows)

    assert DEFAULT_PREBREAK_LOOKBACK_DAYS == 365
    assert pd.to_datetime(plan["market_date"]).min() <= pd.Timestamp("2019-02-20")
    assert plan["days_to_break"].max() >= 360


def test_rank_predictive_signals_finds_monotonic_break_signal() -> None:
    frame = pd.DataFrame(
        {
            "market_date": pd.bdate_range("2024-01-01", periods=20).astype(str),
            "risk_signal": list(range(20)),
            "noise_signal": [1, 0] * 10,
            "break_severity_3m": [value / 20 for value in range(20)],
            "forward_break_label_3m": [False] * 10 + [True] * 10,
            "forward_major_break_label_3m": [False] * 20,
        }
    )

    rankings = rank_predictive_signals(frame, min_observations=10)

    assert rankings.iloc[0]["signal"] == "risk_signal"
    assert rankings.iloc[0]["risk_direction"] == "higher_is_riskier"
    assert rankings.iloc[0]["spearman_to_break_severity"] > 0.95


def test_rank_predictive_signals_handles_boolean_operational_flags() -> None:
    frame = pd.DataFrame(
        {
            "market_date": pd.bdate_range("2024-01-01", periods=20).astype(str),
            "hard_defensive_action_flag": [False] * 10 + [True] * 10,
            "break_severity_3m": [0.0] * 10 + [0.2] * 10,
            "forward_break_label_3m": [False] * 10 + [True] * 10,
            "forward_major_break_label_3m": [False] * 20,
        }
    )

    rankings = rank_predictive_signals(frame, min_observations=10)

    assert "hard_defensive_action_flag" in rankings["signal"].tolist()


def test_rank_predictive_signals_excludes_postbreak_metadata() -> None:
    frame = pd.DataFrame(
        {
            "market_date": pd.bdate_range("2024-01-01", periods=20).astype(str),
            "postbreak_snapshot": [False] * 10 + [True] * 10,
            "risk_signal": list(range(20)),
            "forward_upside_proxy_3m": [value / 20 for value in range(20)],
            "break_severity_3m": [value / 20 for value in range(20)],
            "forward_break_label_3m": [False] * 10 + [True] * 10,
            "forward_major_break_label_3m": [False] * 20,
        }
    )

    rankings = rank_predictive_signals(frame, min_observations=10)

    assert "postbreak_snapshot" not in rankings["signal"].tolist()
    assert "forward_upside_proxy_3m" not in rankings["signal"].tolist()
    assert rankings.iloc[0]["signal"] == "risk_signal"


def test_current_readout_marks_high_risk_percentile() -> None:
    frame = pd.DataFrame(
        {
            "market_date": pd.bdate_range("2024-01-01", periods=20).astype(str),
            "risk_signal": list(range(20)),
            "break_severity_3m": [value / 20 for value in range(20)],
            "forward_break_label_3m": [False] * 10 + [True] * 10,
            "forward_major_break_label_3m": [False] * 20,
        }
    )
    rankings = rank_predictive_signals(frame, min_observations=10)

    readout = current_best_signal_readout(frame, rankings, top_n=1)

    assert readout.iloc[0]["signal"] == "risk_signal"
    assert readout.iloc[0]["current_risk_read"] == "high_risk"


def test_action_timing_summarizes_first_defensive_snapshot() -> None:
    frame = pd.DataFrame(
        {
            "event_name": ["test_break", "test_break", "test_break"],
            "event_break_date": ["2024-04-01"] * 3,
            "market_date": ["2024-01-31", "2024-02-07", "2024-02-14"],
            "days_to_break": [61, 54, 47],
            "defensive_action_flag": [False, True, True],
            "hard_defensive_action_flag": [False, False, True],
            "forward_break_label_3m": [True, True, True],
            "hindsight_action_aligned": [False, True, True],
            "risk_budget_multiplier": [0.80, 0.60, 0.50],
        }
    )

    timing = summarize_action_timing(frame)
    event = timing[timing["event_name"].eq("test_break")].iloc[0]

    assert event["first_defensive_market_date"] == "2024-02-07"
    assert event["first_defensive_days_before_break"] == 54
    assert event["aligned_when_severe_share"] == 2 / 3
    assert event["first_hard_defensive_market_date"] == "2024-02-14"
    assert event["first_hard_defensive_days_before_break"] == 47
    assert event["hard_aligned_when_severe_share"] == 1 / 3


def test_staged_risk_behavior_scores_early_hard_false_alarms() -> None:
    frame = pd.DataFrame(
        {
            "event_name": ["test_break", "test_break", "test_break", "test_break"],
            "event_family": ["test"] * 4,
            "event_break_date": ["2024-04-01"] * 4,
            "days_to_break": [150, 90, 30, 5],
            "prebreak_stage": [
                "long_lead_context",
                "early_watch",
                "confirmed_prebreak",
                "break_unwind",
            ],
            "prebreak_stage_order": [0, 1, 3, 4],
            "target_staged_risk_budget_multiplier": [1.0, 0.75, 0.35, 0.20],
            "risk_budget_multiplier": [0.40, 0.40, 0.35, 0.20],
            "over_defensive_gap_to_stage_target": [0.60, 0.35, 0.0, 0.0],
            "under_defensive_gap_to_stage_target": [0.0, 0.0, 0.0, 0.0],
            "defensive_action_flag": [True, True, True, True],
            "hard_defensive_action_flag": [True, True, True, True],
            "forward_break_label_3m": [False, False, True, True],
            "forward_major_break_label_3m": [False, False, False, True],
            "forward_spy_return_3m": [0.08, 0.07, -0.02, -0.10],
            "forward_qqq_return_3m": [0.10, 0.09, -0.04, -0.12],
            "forward_smh_return_3m": [0.12, 0.11, -0.05, -0.20],
            "forward_min_max_drawdown_3m": [-0.02, -0.03, -0.12, -0.20],
            "early_hard_false_alarm_flag": [True, True, False, False],
        }
    )

    staged = summarize_staged_risk_behavior(frame)
    early = staged[
        staged["event_name"].eq("test_break") & staged["prebreak_stage"].eq("early_watch")
    ].iloc[0]

    assert early["target_staged_risk_budget_multiplier"] == 0.75
    assert early["mean_candidate_risk_budget_lift"] == 0.35
    assert early["early_hard_false_alarm_share"] == 1.0
    assert early["median_forward_return_when_early_hard_false_alarm"] == 0.11


def test_late_trigger_mesh_quantifies_gated_hard_defense_tradeoff() -> None:
    frame = pd.DataFrame(
        {
            "event_name": ["test_break"] * 5,
            "days_to_break": [90, 45, 30, 15, 5],
            "prebreak_stage": [
                "early_watch",
                "confirmed_prebreak",
                "confirmed_prebreak",
                "confirmed_prebreak",
                "break_unwind",
            ],
            "target_staged_risk_budget_multiplier": [0.75, 0.35, 0.35, 0.35, 0.20],
            "risk_budget_multiplier": [0.40, 0.35, 0.35, 0.35, 0.20],
            "hard_defensive_action_flag": [True, True, True, True, True],
            "forward_break_label_3m": [False, False, True, True, True],
            "forward_spy_return_3m": [0.05, 0.01, -0.03, -0.04, -0.08],
            "forward_qqq_return_3m": [0.07, 0.02, -0.04, -0.06, -0.12],
            "forward_smh_return_3m": [0.09, 0.03, -0.05, -0.08, -0.15],
            "forward_min_max_drawdown_3m": [-0.02, -0.04, -0.12, -0.13, -0.15],
        }
    )

    mesh = build_late_trigger_mesh(frame, trigger_days=(15,))
    row = mesh.iloc[0]

    assert row["actual_first_hard_defensive_days_before_break"] == 90
    assert row["hard_defense_lead_cut_days"] == 75
    assert row["missed_severe_label_share_if_gated"] == 1 / 3
    assert row["pre_trigger_false_alarm_share"] == 1 / 3
    assert row["mesh_read"] == "too_late"


def test_hard_defense_attribution_identifies_base_strategy_stickiness() -> None:
    frame = pd.DataFrame(
        {
            "event_name": ["test_break", "test_break", "test_break"],
            "prebreak_stage": ["early_watch", "early_watch", "confirmed_prebreak"],
            "prebreak_stage_order": [1, 1, 3],
            "hard_defensive_action_flag": [True, True, True],
            "early_hard_false_alarm_flag": [True, False, False],
            "risk_budget_multiplier": [0.35, 0.45, 0.40],
            "current_risk_asset_weight": [0.40, 0.80, 0.80],
            "target_risk_asset_weight": [0.30, 0.35, 0.35],
            "scenario_event_macro_multiplier": [0.90, 0.65, 0.65],
            "portfolio_risk_multiplier": [0.95, 0.95, 0.95],
            "risk_status": ["yellow", "yellow", "orange"],
            "recommended_action": ["HOLD", "REVIEW_REDUCE_RISK", "REDUCE_RISK"],
        }
    )

    attribution = summarize_hard_defense_attribution(frame)
    early = attribution[
        attribution["event_name"].eq("test_break") & attribution["prebreak_stage"].eq("early_watch")
    ]

    assert {
        "base_strategy_already_defensive",
        "scenario_event_macro_overlay",
    }.issubset(set(early["hard_defense_source"]))
    base = early[early["hard_defense_source"].eq("base_strategy_already_defensive")].iloc[0]
    assert base["early_hard_false_alarm_share"] == 1.0


def test_hard_defense_attribution_prefers_persisted_causal_layer() -> None:
    frame = pd.DataFrame(
        {
            "event_name": ["test_break"],
            "prebreak_stage": ["early_watch"],
            "prebreak_stage_order": [1],
            "hard_defensive_action_flag": [True],
            "early_hard_false_alarm_flag": [False],
            "risk_budget_multiplier": [0.55],
            "current_risk_asset_weight": [0.60],
            "target_risk_asset_weight": [0.45],
            "scenario_event_macro_multiplier": [0.55],
            "portfolio_risk_multiplier": [1.0],
            "risk_status": ["orange"],
            "recommended_action": ["REDUCE_RISK"],
            "attribution_base_market_strategy_defensive_weight": [0.40],
            "attribution_quantitative_risk_status_defensive_add_pp": [15.0],
            "attribution_scenario_probabilities_defensive_add_pp": [0.0],
            "attribution_news_event_pressure_defensive_add_pp": [0.0],
            "attribution_macro_quantitative_defensive_add_pp": [0.0],
            "attribution_portfolio_absolute_risk_defensive_add_pp": [0.0],
            "attribution_decision_sanity_defensive_add_pp": [0.0],
        }
    )

    attribution = summarize_hard_defense_attribution(frame)

    assert attribution.iloc[0]["hard_defense_source"] == "quantitative_risk_status"


def test_policy_variants_score_added_false_alarm_upside_against_crash_exposure() -> None:
    frame = pd.DataFrame(
        {
            "event_name": ["test_break"] * 4,
            "days_to_break": [100, 80, 35, 10],
            "prebreak_stage": [
                "early_watch",
                "early_watch",
                "confirmed_prebreak",
                "break_unwind",
            ],
            "target_staged_risk_budget_multiplier": [0.75, 0.75, 0.35, 0.20],
            "risk_budget_multiplier": [0.35, 0.35, 0.35, 0.20],
            "hard_defensive_action_flag": [True, True, True, True],
            "current_risk_asset_weight": [0.90, 0.90, 0.90, 0.90],
            "target_risk_asset_weight": [0.35, 0.35, 0.35, 0.20],
            "scenario_event_macro_multiplier": [0.90, 0.90, 0.90, 0.90],
            "portfolio_risk_multiplier": [0.35, 0.35, 0.35, 0.90],
            "portfolio_constraints": [
                "expected_shortfall",
                "expected_shortfall",
                "expected_shortfall",
                "none",
            ],
            "decision_sanity_break_count": [0, 0, 1, 2],
            "risk_status": ["yellow", "yellow", "yellow", "orange"],
            "recommended_action": ["REVIEW_REDUCE_RISK"] * 4,
            "forward_break_label_3m": [False, False, True, True],
            "forward_spy_return_3m": [0.06, 0.04, -0.04, -0.08],
            "forward_qqq_return_3m": [0.08, 0.05, -0.06, -0.11],
            "forward_smh_return_3m": [0.10, 0.06, -0.07, -0.14],
            "forward_min_max_drawdown_3m": [-0.02, -0.03, -0.12, -0.18],
        }
    )

    variants = evaluate_staged_policy_variants(frame)
    stage_floor = variants[
        variants["event_name"].eq("ALL_EVENTS") & variants["policy_name"].eq("stage_floor")
    ].iloc[0]
    actual = variants[
        variants["event_name"].eq("ALL_EVENTS") & variants["policy_name"].eq("actual")
    ].iloc[0]

    assert stage_floor["mean_false_alarm_risk_budget_lift"] == 0.40
    assert stage_floor["mean_severe_label_risk_budget_lift"] == 0.0
    assert stage_floor["candidate_score"] > actual["candidate_score"]
    portfolio = variants[
        variants["event_name"].eq("ALL_EVENTS")
        & variants["policy_name"].eq("portfolio_watch_floor")
    ].iloc[0]
    assert portfolio["mean_false_alarm_risk_budget_lift"] == 0.40
    assert "portfolio_confirm_30d_conservative" in variants["policy_name"].tolist()


def test_snapshot_signal_row_keeps_incomplete_forward_label_missing() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0], "QQQ": [100.0], "SMH": [100.0]},
        index=pd.DatetimeIndex(["2024-01-31"]),
    )
    current_state = SimpleNamespace(
        market_date="2024-01-31",
        risk_status="red",
        risk_score=0.9,
        scenario_drivers=pd.DataFrame(),
        confirmation_matrix=pd.DataFrame(),
        market_health=pd.DataFrame(),
        regime_instability=pd.DataFrame(),
        regime_instability_components=pd.DataFrame(),
    )
    run = SimpleNamespace(
        current_state=current_state,
        prices=prices,
        trade_decision=SimpleNamespace(summary=pd.DataFrame()),
    )

    row = snapshot_signal_row(
        run,
        run_id="test",
        created_at_utc="2024-01-31T22:00:00Z",
        reference_prices=prices,
    )

    assert pd.isna(row["forward_break_label_3m"])
    assert pd.isna(row["break_severity_3m"])
    assert row["hindsight_action_aligned"] is False
