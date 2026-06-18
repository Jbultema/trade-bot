from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import trade_bot.research.baselines as baselines_module
import trade_bot.storage.run_store as run_store_module
import trade_bot.trading.journal as journal_module
from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.baselines import BaselineRun
from trade_bot.research.current_state import CurrentStateRun
from trade_bot.research.event_risk import EventRiskRun
from trade_bot.research.news_monitor import NewsMonitorRun
from trade_bot.research.signal_inclusion import SignalInclusionRun
from trade_bot.research.trade_decision import TradeDecisionRun


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
    assert any(title.value == "Trade Bot Operations" for title in app.title)
    assert any("Action Headline" in markdown.value for markdown in app.markdown)
    assert any("stMetricValue" in markdown.value for markdown in app.markdown)
    assert any("Small Actions" in markdown.value for markdown in app.markdown)
    assert any(subheader.value == "Operating Brief" for subheader in app.subheader)
    assert any("Scenario Incorporation" in markdown.value for markdown in app.markdown)
    assert any(subheader.value == "Decision Brief" for subheader in app.subheader)
    assert any(radio.label == "Dashboard section" for radio in app.radio)
    assert any(subheader.value == "Current State" for subheader in app.subheader)
    assert any(subheader.value == "Trade Decision" for subheader in app.subheader)

    dashboard_section = next(radio for radio in app.radio if radio.label == "Dashboard section")
    dashboard_section.set_value("Risk & Scenarios").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Portfolio Risk Engine" for subheader in app.subheader)

    dashboard_section = next(radio for radio in app.radio if radio.label == "Dashboard section")
    dashboard_section.set_value("Research Lab").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Approach Explorer" for subheader in app.subheader)

    dashboard_section = next(radio for radio in app.radio if radio.label == "Dashboard section")
    dashboard_section.set_value("Forward Test").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Forward Test / Trade Journal" for subheader in app.subheader)

    dashboard_section = next(radio for radio in app.radio if radio.label == "Dashboard section")
    dashboard_section.set_value("Performance").run(timeout=20)
    assert not app.exception
    assert any(subheader.value == "Windowed Performance" for subheader in app.subheader)


class _FakeRunStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def load_latest_snapshot(self, *args: object, **kwargs: object) -> None:
        return None

    def list_jobs(self, *, limit: int = 8) -> pd.DataFrame:
        return pd.DataFrame()

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
        vams=pd.DataFrame({"ticker": ["QQQ"], "vams_state": ["bullish"]}),
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
                    "event_pressure": 0.04,
                    "macro_pressure": 0.0,
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
