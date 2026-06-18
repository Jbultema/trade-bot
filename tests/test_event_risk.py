from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.research.event_risk import (
    MarketEvent,
    classify_news_text,
    run_event_risk_study,
)


def test_event_risk_study_calculates_asset_and_strategy_windows() -> None:
    index = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame(
        {
            "SPY": [100.0 + value for value in range(30)],
            "QQQ": [100.0 + value * 1.5 for value in range(30)],
            "USO": [50.0 + value * 0.5 for value in range(30)],
            "BIL": [100.0 for _ in range(30)],
        },
        index=index,
    )
    returns = prices["SPY"].pct_change(fill_method=None).fillna(0.0)
    equity = 100.0 * (1.0 + returns).cumprod()
    weights = pd.DataFrame({"SPY": 1.0}, index=index)
    result = BacktestResult(
        name="demo",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )
    event = MarketEvent(
        event_id="demo_event",
        name="Demo event",
        date=pd.Timestamp("2024-01-10"),
        category="oil_chokepoint",
        direction="deescalation",
        description="test",
        current=True,
    )

    study = run_event_risk_study(
        prices,
        {"demo": result},
        (event,),
        windows=(-3, 1, 5),
        asset_proxies=("SPY", "QQQ", "USO", "BIL"),
        primary_strategy="demo",
    )

    spy_post_5d = study.asset_event_returns[
        (study.asset_event_returns["ticker"] == "SPY")
        & (study.asset_event_returns["window"] == "post_5d")
    ].iloc[0]

    assert spy_post_5d["complete"]
    assert spy_post_5d["return"] == pytest.approx(5.0 / 107.0)
    assert not study.event_summary.empty
    assert not study.current_event_scenarios.empty


def test_classify_news_text_maps_hormuz_deal_to_oil_chokepoint_relief() -> None:
    classification = classify_news_text(
        "The Iran deal would reopen the Strait of Hormuz and let oil flow again."
    )

    assert classification.category == "oil_chokepoint"
    assert classification.direction == "deescalation"
    assert "USO" in classification.candidate_proxies
    assert "oil" in classification.risk_channels


def test_classify_news_text_maps_openai_financials_to_ai_unit_economics() -> None:
    classification = classify_news_text(
        "Exclusive audited OpenAI financials showed 2025 losses, revenue growth, "
        "AI capex spending, and compute costs."
    )

    assert classification.category == "ai_unit_economics"
    assert classification.direction == "escalation"
    assert classification.phase == "leading_warning"
    assert "SMH" in classification.candidate_proxies
    assert "SOXX" in classification.candidate_proxies
    assert "market_concentration" in classification.risk_channels


def test_classify_news_text_maps_private_credit_to_credit_liquidity_risk() -> None:
    classification = classify_news_text(
        "New data revealed private credit and direct lending stress in middle market loans."
    )

    assert classification.category == "private_credit"
    assert classification.direction == "escalation"
    assert classification.phase == "leading_warning"
    assert "BKLN" in classification.candidate_proxies
    assert "KRE" in classification.candidate_proxies
    assert "liquidity" in classification.risk_channels


def test_classify_news_text_does_not_map_close_or_cloud_to_private_credit() -> None:
    classification = classify_news_text(
        "CoreWeave announced new cloud AI infrastructure results for Nvidia GPU clusters."
    )

    assert classification.category == "ai_infrastructure"
    assert classification.category != "private_credit"


def test_classify_news_text_maps_oil_inventory_news_to_energy_supply() -> None:
    classification = classify_news_text(
        "EIA new data showed US crude oil and gasoline inventories still falling."
    )

    assert classification.category == "energy_supply"
    assert classification.direction == "escalation"
    assert "USO" in classification.candidate_proxies
    assert "inflation" in classification.risk_channels


def test_classify_news_text_does_not_map_warsh_to_military_escalation() -> None:
    classification = classify_news_text(
        "Kevin Warsh said the Federal Reserve should change how it handles inflation."
    )

    assert classification.category != "military_escalation"


def test_event_risk_study_surfaces_event_phase_in_scenario_playbook() -> None:
    index = pd.bdate_range("2026-06-01", periods=20)
    prices = pd.DataFrame(
        {
            "SPY": [100.0 + value for value in range(20)],
            "QQQ": [100.0 + value for value in range(20)],
            "SMH": [100.0 + value for value in range(20)],
            "BIL": [100.0 for _ in range(20)],
        },
        index=index,
    )
    returns = prices["SPY"].pct_change(fill_method=None).fillna(0.0)
    equity = 100.0 * (1.0 + returns).cumprod()
    weights = pd.DataFrame({"SPY": 1.0}, index=index)
    result = BacktestResult(
        name="demo",
        equity=equity,
        returns=returns,
        gross_returns=returns,
        weights=weights,
        target_weights=weights,
        turnover=pd.Series(0.0, index=index),
        transaction_costs=pd.Series(0.0, index=index),
    )
    event = MarketEvent(
        event_id="openai_financials",
        name="OpenAI financials",
        date=pd.Timestamp("2026-06-10"),
        category="ai_unit_economics",
        direction="escalation",
        description="test",
        current=True,
        phase="leading_warning",
        phase_reason="Audited disclosure may lead public-market repricing.",
        confirmation_window="Watch 1d/5d/21d confirmation.",
    )

    study = run_event_risk_study(
        prices,
        {"demo": result},
        (event,),
        windows=(1, 5),
        asset_proxies=("SPY", "QQQ", "SMH", "BIL"),
        primary_strategy="demo",
    )

    assert set(study.current_event_scenarios["event_phase"]) == {"leading_warning"}
    assert "AI capex repricing starts" in set(study.current_event_scenarios["scenario"])
