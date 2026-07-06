from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import trade_bot.research.baselines as baselines_module
import trade_bot.storage.run_store as run_store_module
import trade_bot.trading.journal as journal_module
from trade_bot.backtest.engine import BacktestResult
from trade_bot.dashboard.market_brief import build_market_brief_report
from trade_bot.research.action_headline import build_action_headline
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.event_risk import EventRiskRun
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.signal_inclusion import SignalInclusionRun
from trade_bot.research.trade_decision import TradeDecisionRun
from trade_bot.trading.book_alignment import BookAlignmentRun


def test_dashboard_app_renders_action_headline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        baselines_module,
        "run_configured_baselines",
        lambda *args, **kwargs: _baseline_run(),
    )
    monkeypatch.setattr(run_store_module, "RunStore", _FakeRunStore)
    monkeypatch.setattr(journal_module, "DEFAULT_JOURNAL_PATH", tmp_path / "journal.sqlite")

    app = AppTest.from_file("src/trade_bot/dashboard/app.py", default_timeout=20)
    app.run(timeout=20)

    assert not app.exception
    assert any("brand-masthead" in markdown.value for markdown in app.markdown)
    assert any("Trade Bot Operations" in markdown.value for markdown in app.markdown)
    assert any("Regime Research Lab" in markdown.value for markdown in app.markdown)
    assert any("freshness-strip" in markdown.value for markdown in app.markdown)
    assert any("Latest update" in markdown.value for markdown in app.markdown)
    assert any("Market date 2026-06-17" in markdown.value for markdown in app.markdown)
    assert any("Term Lookup" in markdown.value for markdown in app.markdown)
    assert any("ticket field" in markdown.value for markdown in app.markdown)
    assert any("metric-info-rail" in markdown.value for markdown in app.markdown)
    assert any(button.label == "Run Full Daily Update" for button in app.button)
    assert any(button.label == "Migrate Warehouse" for button in app.button)
    assert any(button.label == "Run Paper Valuation" for button in app.button)
    assert any(button.label == "Seed Monitoring Windows" for button in app.button)
    assert any(button.label == "Run ML Diagnostics" for button in app.button)
    assert any("dashboard-primary-nav-label" in markdown.value for markdown in app.markdown)
    assert not any("Daily Market Brief" in markdown.value for markdown in app.markdown)
    assert any("Action Headline" in markdown.value for markdown in app.markdown)
    assert any("stMetricValue" in markdown.value for markdown in app.markdown)
    assert any("Small Actions" in markdown.value for markdown in app.markdown)
    assert any(subheader.value == "Operating Brief" for subheader in app.subheader)
    assert any("Risk Constraints" in markdown.value for markdown in app.markdown)
    assert any("Bias Check" in markdown.value for markdown in app.markdown)
    assert any(subheader.value == "Book Alignment" for subheader in app.subheader)
    operating_html = [
        markdown.value
        for markdown in app.markdown
        if '<div class="operating-grid">' in markdown.value
    ]
    assert operating_html
    assert '</div><div class="operating-card' in operating_html[0]
    assert '\n    <div class="operating-card' not in operating_html[0]
    assert any(subheader.value == "Decision Brief" for subheader in app.subheader)
    brief_html = [
        markdown.value for markdown in app.markdown if '<div class="brief-grid">' in markdown.value
    ]
    assert brief_html
    assert '</div><div class="brief-card' in brief_html[0]
    assert '\n    <div class="brief-card' not in brief_html[0]
    assert any(pills.label == "Dashboard section" for pills in app.pills)
    assert any(subheader.value == "Current State" for subheader in app.subheader)
    assert any(subheader.value == "Trade Decision" for subheader in app.subheader)

    dashboard_section = next(pills for pills in app.pills if pills.label == "Dashboard section")
    dashboard_section.set_value("Risk & Scenarios").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Portfolio Risk Engine" for subheader in app.subheader)

    dashboard_section = next(pills for pills in app.pills if pills.label == "Dashboard section")
    dashboard_section.set_value("Simulation Lab").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Simulation Lab" for subheader in app.subheader)
    assert any("Future-State Simulation Map" in markdown.value for markdown in app.markdown)
    assert any(selectbox.label == "Strategy to simulate" for selectbox in app.selectbox)

    dashboard_section = next(pills for pills in app.pills if pills.label == "Dashboard section")
    dashboard_section.set_value("Research Lab").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Experiment Monitor" for subheader in app.subheader)
    assert any("Outcome Frontier" in markdown.value for markdown in app.markdown)
    assert any(radio.label == "Approach set" for radio in app.radio)
    assert any(selectbox.label == "Approach to inspect" for selectbox in app.selectbox)
    assert not any(subheader.value == "Approach Explorer" for subheader in app.subheader)

    dashboard_section = next(pills for pills in app.pills if pills.label == "Dashboard section")
    dashboard_section.set_value("Forward Test").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Forward Test / Trade Journal" for subheader in app.subheader)

    dashboard_section = next(pills for pills in app.pills if pills.label == "Dashboard section")
    dashboard_section.set_value("Performance").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Windowed Performance" for subheader in app.subheader)


def test_market_brief_report_summarizes_market_news_and_scenarios() -> None:
    baseline_run = _baseline_run()
    headline = build_action_headline(
        current_state=baseline_run.current_state,
        trade_decision=baseline_run.trade_decision,
        news_monitor=baseline_run.news_monitor,
        open_ticket_count=0,
    )

    report = build_market_brief_report(
        baseline_run=baseline_run,
        headline=headline,
        open_ticket_count=0,
    )

    assert report.tone == "warning"
    assert "YELLOW risk" in report.title
    assert "1-month scenario map" in report.summary
    assert len(report.paragraphs) == 4
    assert "Current posture: as of 2026-06-17" in report.paragraphs[0]
    assert "Choppy factor rotation" in report.paragraphs[0]
    assert "no prior snapshot" in report.paragraphs[1].lower()
    assert "Daily delta" in report.paragraphs[1]
    assert "Still true: the driver stack" in report.paragraphs[2]
    assert "explainer/research-only" in report.paragraphs[2]
    assert "Action read-through" in report.paragraphs[3]
    assert "What would change this" in report.paragraphs[3]
    assert [card.label for card in report.daily_delta_cards] == [
        "What Changed Today",
        "Still True",
    ]
    assert [card.label for card in report.cards] == [
        "Change Since Prior",
        "Driver Stack",
        "Scenario Map",
        "Risk Budget / Action",
        "Decision Sanity",
    ]
    scenario_card = next(card for card in report.cards if card.label == "Scenario Map")
    assert "Choppy factor rotation" in scenario_card.detail
    assert "explainer_research_only" in report.detail_rows.to_string()
    latest_table_names = [label for label, _frame in report.latest_input_tables]
    assert latest_table_names == [
        "Current operating numbers",
        "Latest news and events",
        "Scenario probabilities",
        "Macro pressure groups",
        "Confirmation matrix",
        "Position plan",
        "Cross-source diagnostics",
    ]
    numbers_frame = dict(report.latest_input_tables)["Current operating numbers"]
    assert "risk_score" in set(numbers_frame["input"])
    assert "target_posture" in set(numbers_frame["input"])
    assert set(report.detail_rows["topic"]) >= {
        "market_state",
        "scenario_map",
        "change_since_prior",
        "news_events",
        "macro_stack",
        "regime_pulse",
        "risk_budget",
    }
    role_lookup = report.detail_rows.set_index("topic")["model_role"].to_dict()
    assert role_lookup["risk_budget"] == "allocation_driver"
    assert role_lookup["cross_source_signals"] == "explainer_research_only"


def test_market_brief_report_compares_current_run_to_prior_snapshot() -> None:
    baseline_run = _baseline_run()
    previous_run = _previous_baseline_run()
    headline = build_action_headline(
        current_state=baseline_run.current_state,
        trade_decision=baseline_run.trade_decision,
        news_monitor=baseline_run.news_monitor,
        open_ticket_count=0,
    )

    report = build_market_brief_report(
        baseline_run=baseline_run,
        headline=headline,
        open_ticket_count=0,
        previous_run=previous_run,
    )

    change_card = next(card for card in report.cards if card.label == "Change Since Prior")
    assert change_card.answer == "Action changed to Review Reduce Risk"
    assert change_card.tone in {"warning", "critical"}
    assert "prior stored posture" in report.paragraphs[1]
    assert "action Hold -> Review Reduce Risk" in report.paragraphs[1]
    assert "1M risk-off probability 2% -> 12% (+10pp)" in report.paragraphs[1]
    change_rows = report.detail_rows[report.detail_rows["topic"] == "change_since_prior"]
    assert not change_rows.empty
    assert "2026-06-16 GREEN risk" in str(change_rows.iloc[0]["previous_read"])


def test_market_brief_report_uses_book_alignment_for_execution_readthrough() -> None:
    baseline_run = _baseline_run()
    book_alignment = _book_alignment_run()
    headline = build_action_headline(
        current_state=baseline_run.current_state,
        trade_decision=baseline_run.trade_decision,
        news_monitor=baseline_run.news_monitor,
        open_ticket_count=0,
        position_plan=book_alignment.position_plan,
    )

    report = build_market_brief_report(
        baseline_run=baseline_run,
        headline=headline,
        open_ticket_count=0,
        book_alignment=book_alignment,
    )

    assert "default paper book is currently QQQ 36%, IWM 36%, BIL 28%" in report.paragraphs[0]
    assert "latest target is QQQ 37%, IWM 37%, BIL 25%" in report.paragraphs[0]
    assert "reduce BIL by 3.00%" in report.paragraphs[3]
    assert "add bil by 10.00%" not in report.paragraphs[3].lower()


class _FakeRunStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def load_latest_snapshot(self, *args: object, **kwargs: object) -> None:
        return None

    def list_jobs(self, *, limit: int = 8) -> pd.DataFrame:
        return pd.DataFrame()

    def list_snapshots(self, *, limit: int = 50) -> pd.DataFrame:
        return pd.DataFrame()

    def load_snapshot(self, run_id: str) -> object:
        raise FileNotFoundError(run_id)

    def start_snapshot_build_job(self, *args: object, **kwargs: object) -> object:
        return type("Job", (), {"job_id": "job-test"})()


def _baseline_run() -> BaselineRun:
    index = pd.bdate_range("2026-06-01", periods=8)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "QQQ": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0],
            "IWM": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5],
            "BIL": [100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07],
        },
        index=index,
    )
    weights = pd.DataFrame(
        {
            "QQQ": [0.5 for _ in index],
            "IWM": [0.5 for _ in index],
        },
        index=index,
    )
    returns = pd.Series([0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], index=index)
    equity = 100.0 * (1.0 + returns).cumprod()
    result = BacktestResult(
        name="drawdown_managed_dual_momentum",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )
    metrics = pd.DataFrame(
        {
            "cagr": [0.10],
            "sharpe": [0.8],
            "sortino": [1.2],
            "max_drawdown": [-0.15],
            "calmar": [0.67],
            "average_turnover": [0.03],
        },
        index=pd.Index(["drawdown_managed_dual_momentum"], name="name"),
    )
    window_summary = pd.DataFrame(
        {
            "median_cagr": [0.08],
            "worst_cagr": [-0.05],
            "worst_drawdown": [-0.15],
            "positive_window_rate": [0.8],
            "median_calmar": [0.5],
        },
        index=pd.MultiIndex.from_tuples(
            [("drawdown_managed_dual_momentum", "1y")],
            names=["name", "window"],
        ),
    )
    calendar_returns = pd.DataFrame(
        {"drawdown_managed_dual_momentum": [0.08]},
        index=pd.Index(["2026"], name="window"),
    )
    current_state = _current_state()
    trade_decision = _trade_decision()
    return BaselineRun(
        prices=prices,
        macro_data=pd.DataFrame(),
        macro_catalog=(),
        results={"drawdown_managed_dual_momentum": result},
        metrics=metrics,
        rolling_windows=pd.DataFrame(),
        window_summary=window_summary,
        calendar_metrics=pd.DataFrame(),
        calendar_returns=calendar_returns,
        current_state=current_state,
        event_risk=EventRiskRun(
            events=(),
            asset_event_returns=pd.DataFrame(),
            strategy_event_returns=pd.DataFrame(),
            event_summary=pd.DataFrame(),
            scenario_playbook=pd.DataFrame(),
            current_event_scenarios=pd.DataFrame(),
        ),
        news_monitor=NewsMonitorRun(
            items=(),
            triage=pd.DataFrame(),
            source_health=pd.DataFrame(),
            activated_events=(),
            activation_threshold=0.8,
            lookback_days=7,
        ),
        signal_inclusion=SignalInclusionRun(
            summary=pd.DataFrame(),
            pressure=pd.DataFrame(),
            results={},
            metrics=pd.DataFrame(),
            window_summary=pd.DataFrame(),
        ),
        trade_decision=trade_decision,
    )


def _previous_baseline_run() -> BaselineRun:
    run = _baseline_run()
    previous_state = replace(
        run.current_state,
        market_date="2026-06-16",
        risk_score=0.25,
        risk_status="green",
        risk_summary="Risk status was GREEN with score 0.25.",
    )
    previous_summary = run.trade_decision.summary.copy()
    previous_summary.loc[0, "recommended_action"] = "HOLD"
    previous_summary.loc[0, "risk_status"] = "green"
    previous_summary.loc[0, "risk_score"] = 0.25
    previous_summary.loc[0, "risk_budget_multiplier"] = 1.0
    previous_summary.loc[0, "one_month_risk_off_probability"] = 0.02
    previous_summary.loc[0, "one_month_transition_probability"] = 0.10
    previous_summary.loc[0, "one_month_risk_on_probability"] = 0.80
    previous_summary.loc[0, "event_pressure"] = 0.0
    previous_summary.loc[0, "scenario_adjusted_position"] = "QQQ 50%, IWM 50%"
    previous_links = pd.DataFrame(
        [
            {
                "rank": 1,
                "scenario": "Broad risk-on",
                "probability": 0.80,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Hold risk.",
                "preferred_exposure": "QQQ",
                "avoid_exposure": "None",
                "confirmation": "Trend holds.",
                "off_ramp": "Trend breaks.",
            }
        ]
    )
    previous_decision = replace(
        run.trade_decision,
        summary=previous_summary,
        scenario_links=previous_links,
    )
    return replace(run, current_state=previous_state, trade_decision=previous_decision)


def _book_alignment_run() -> BookAlignmentRun:
    return BookAlignmentRun(
        summary=pd.DataFrame(
            [
                {
                    "mode": "paper",
                    "account": "default_paper_account",
                    "strategy_name": "scenario_adjusted_trade_decision",
                    "alignment_status": "small_drift",
                    "recommended_action": "SMALL_REBALANCE",
                    "current_position": "QQQ 36%, IWM 36%, BIL 28%",
                    "target_position": "QQQ 37%, IWM 37%, BIL 25%",
                    "max_abs_delta": 0.03,
                    "material_trade_count": 1,
                    "has_executions": True,
                }
            ]
        ),
        position_plan=pd.DataFrame(
            [
                {
                    "ticker": "BIL",
                    "current_weight": 0.28,
                    "scenario_adjusted_weight": 0.25,
                    "target_weight": 0.25,
                    "delta_weight": -0.03,
                    "action": "REDUCE",
                },
                {
                    "ticker": "QQQ",
                    "current_weight": 0.36,
                    "scenario_adjusted_weight": 0.37,
                    "target_weight": 0.37,
                    "delta_weight": 0.01,
                    "action": "HOLD",
                },
            ]
        ),
        holdings=pd.DataFrame(
            [
                {
                    "mode": "paper",
                    "account": "default_paper_account",
                    "ticker": "QQQ",
                    "net_quantity": 1.0,
                    "current_notional": 3600.0,
                }
            ]
        ),
    )


def _current_state() -> CurrentStateRun:
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "rank": 1,
                "scenario": "Choppy factor rotation",
                "probability": 0.25,
                "risk_bucket": "transition",
                "expected_bot_posture": "Keep risk smaller.",
                "preferred_exposure": "Quality",
                "avoid_exposure": "Over-trading",
                "confirmation": "Breadth remains mixed.",
                "off_ramp": "Move defensive if credit breaks.",
            },
            {
                "horizon": "1w",
                "rank": 1,
                "scenario": "Broad risk-on",
                "probability": 1.0,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Hold risk.",
                "preferred_exposure": "SPY",
                "avoid_exposure": "None",
                "confirmation": "Trend holds.",
                "off_ramp": "Trend breaks.",
            },
            {
                "horizon": "3m",
                "rank": 1,
                "scenario": "Defensive grind",
                "probability": 1.0,
                "risk_bucket": "transition",
                "expected_bot_posture": "Modest risk.",
                "preferred_exposure": "BIL",
                "avoid_exposure": "High beta",
                "confirmation": "Vol remains elevated.",
                "off_ramp": "Breadth improves.",
            },
            {
                "horizon": "6m",
                "rank": 1,
                "scenario": "Risk recovery",
                "probability": 1.0,
                "risk_bucket": "risk_on",
                "expected_bot_posture": "Add risk.",
                "preferred_exposure": "QQQ",
                "avoid_exposure": "None",
                "confirmation": "Credit improves.",
                "off_ramp": "Credit weakens.",
            },
        ]
    )
    return CurrentStateRun(
        market_date="2026-06-17",
        risk_score=0.43,
        risk_status="yellow",
        risk_summary="Risk status is YELLOW with score 0.43.",
        market_health=pd.DataFrame({"metric": ["trend"], "value": [1.0]}),
        momentum_state=pd.DataFrame({"ticker": ["QQQ"], "momentum_state_label": ["bullish"]}),
        confirmation_matrix=pd.DataFrame({"signal": ["trend"], "state": ["bullish"]}),
        strategy_alerts=pd.DataFrame(
            {
                "strategy": ["drawdown_managed_dual_momentum"],
                "priority": ["primary"],
                "action": ["HOLD"],
                "latest_position": ["QQQ 50%, IWM 50%"],
                "trade_alert": ["No material trade."],
            }
        ),
        scenario_outlook=pd.DataFrame(
            {
                "horizon": ["1m"],
                "top_scenario": ["Choppy factor rotation"],
                "risk_status": ["yellow"],
                "probability": [0.25],
            }
        ),
        scenario_lattice=scenario_lattice,
        scenario_drivers=pd.DataFrame({"driver": ["breadth"], "score": [0.2]}),
        macro_signals=pd.DataFrame(),
        macro_category_summary=pd.DataFrame(),
        signal_coverage=pd.DataFrame({"coverage_area": ["market"], "status": ["implemented"]}),
        data_quality=pd.DataFrame({"ticker": ["QQQ"], "usable_share": [1.0]}),
    )


def _trade_decision() -> TradeDecisionRun:
    return TradeDecisionRun(
        summary=pd.DataFrame(
            [
                {
                    "strategy": "drawdown_managed_dual_momentum",
                    "recommended_action": "REVIEW_REDUCE_RISK",
                    "decision_authority": "scenario_event_review",
                    "base_position": "QQQ 50%, IWM 50%",
                    "scenario_adjusted_position": "QQQ 40%, IWM 50%, BIL 10%",
                    "risk_budget_multiplier": 0.8,
                    "risk_status": "yellow",
                    "risk_score": 0.43,
                    "one_month_risk_off_probability": 0.12,
                    "one_month_transition_probability": 0.25,
                    "one_month_fragile_upside_probability": 0.08,
                    "one_month_risk_on_probability": 0.55,
                    "constructive_scenario_probability": 0.59,
                    "scenario_event_macro_multiplier": 0.8,
                    "portfolio_risk_multiplier": 1.0,
                    "event_pressure": 0.04,
                    "macro_pressure": 0.0,
                    "posture_calibration_status": "opportunity_cost_watch",
                    "posture_calibration_signal": "Opportunity-cost watch",
                    "posture_calibration_note": "Constructive evidence still deserves review.",
                    "current_risk_asset_weight": 1.0,
                    "target_risk_asset_weight": 0.9,
                    "target_defensive_weight": 0.1,
                    "opportunity_pressure": 0.47,
                    "human_explanation": "Because risk is yellow, review reducing risk.",
                }
            ]
        ),
        position_plan=pd.DataFrame(
            [
                {
                    "ticker": "QQQ",
                    "current_weight": 0.5,
                    "scenario_adjusted_weight": 0.4,
                    "delta_weight": -0.1,
                    "action": "REDUCE",
                },
                {
                    "ticker": "BIL",
                    "current_weight": 0.0,
                    "scenario_adjusted_weight": 0.1,
                    "delta_weight": 0.1,
                    "action": "ADD",
                },
            ]
        ),
        evidence=pd.DataFrame(
            [{"evidence_type": "risk", "signal": "YELLOW", "impact": "reduce risk"}]
        ),
        scenario_links=pd.DataFrame(
            [
                {
                    "rank": 1,
                    "scenario": "Choppy factor rotation",
                    "probability": 0.25,
                    "risk_bucket": "transition",
                    "expected_bot_posture": "Keep smaller.",
                    "preferred_exposure": "Quality",
                    "avoid_exposure": "High beta",
                    "confirmation": "Mixed breadth.",
                    "off_ramp": "Credit break.",
                }
            ]
        ),
    )
