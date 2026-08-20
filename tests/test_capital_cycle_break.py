from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.research.capital_cycle_break import (
    _capital_cycle_state,
    classify_derivative_state,
    run_capital_cycle_break_monitor,
    score_capital_cycle_events,
)


def test_derivative_state_distinguishes_direction_and_acceleration() -> None:
    assert classify_derivative_state(0.10, 0.08, 0.06) == "worsening_fast"
    assert classify_derivative_state(0.04, 0.02, 0.00) == "worsening"
    assert classify_derivative_state(-0.10, -0.08, -0.06) == "improving_fast"
    assert classify_derivative_state(0.00, 0.01, -0.01) == "flat"
    assert classify_derivative_state(np.nan, np.nan, np.nan) == "unavailable"


def test_structural_vulnerability_cannot_declare_rollover_without_sequence() -> None:
    assert _capital_cycle_state(0.90, 0.20, 0.80, 0.80, 0, 0) == "buildout_fragile"
    assert _capital_cycle_state(0.85, 0.70, 0.80, 0.80, 0, 2) == "prebreak_pressure"
    assert _capital_cycle_state(0.85, 0.70, 0.80, 0.80, 1, 1) == "rollover_confirming"
    assert _capital_cycle_state(0.85, 0.82, 0.80, 0.80, 1, 2) == "break_confirmed"


def test_structured_events_decay_and_exclude_future_information() -> None:
    events = pd.DataFrame(
        [
            {
                "event_date": "2026-06-01",
                "event_type": "failed_or_repriced_financing",
                "entity": "sample",
                "direction": "risk",
                "severity": 0.8,
                "confidence": 1.0,
                "source_quality": "A",
                "cycle_family": "financing_reflexivity",
                "decay_half_life_days": 30,
            },
            {
                "event_date": "2026-08-20",
                "event_type": "future_event",
                "entity": "sample",
                "direction": "risk",
                "severity": 1.0,
                "confidence": 1.0,
                "source_quality": "A",
                "cycle_family": "financing_reflexivity",
                "decay_half_life_days": 90,
            },
        ]
    )

    scored = score_capital_cycle_events(events, as_of="2026-07-01")

    assert scored["event_type"].tolist() == ["failed_or_repriced_financing"]
    assert scored.iloc[0]["decay_weight"] == 0.5
    assert scored.iloc[0]["decayed_contribution"] == 0.4
    assert scored.iloc[0]["allocation_authority"] == 0.0


def test_monitor_reports_partial_coverage_and_missing_upstream_evidence() -> None:
    dates = pd.bdate_range("2021-01-01", periods=900)
    trend = np.exp(np.linspace(0.0, 1.0, len(dates)))
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * trend,
            "QQQ": 100.0 * trend**1.15,
            "RSP": 100.0 * trend**0.90,
            "SMH": 100.0 * trend**1.25,
            "NVDA": 100.0 * trend**1.35,
            "AVGO": 100.0 * trend**1.20,
            "MSFT": 100.0 * trend**1.10,
            "GOOGL": 100.0 * trend**1.05,
            "AMZN": 100.0 * trend**1.08,
            "META": 100.0 * trend**1.12,
            "HYG": 100.0 * trend**0.20,
            "LQD": 100.0 * trend**0.18,
        },
        index=dates,
    )
    months = pd.date_range("2021-01-31", periods=68, freq="ME")
    macro = pd.DataFrame(
        {
            "DFII10": np.linspace(-0.5, 2.0, len(months)),
            "DGS30": np.linspace(2.0, 4.5, len(months)),
            "BAMLC0A0CM": np.linspace(0.8, 1.0, len(months)),
            "BAMLH0A0HYM2": np.linspace(3.0, 3.5, len(months)),
            "AMTMNO": np.linspace(90.0, 130.0, len(months)),
            "DGORDER": np.linspace(200.0, 260.0, len(months)),
        },
        index=months,
    )

    result = run_capital_cycle_break_monitor(prices=prices, macro_data=macro)
    summary = result.summary.iloc[0]

    assert summary["allocation_authority"] == 0.0
    assert summary["model_role"] == "explainer_research_only"
    assert summary["capital_cycle_state"] == "insufficient_data"
    assert summary["direct_real_economy_families"] == 0
    assert summary["vulnerability_coverage"] == 0.0
    assert summary["rollover_coverage"] == 0.0
    assert summary["direction"] == "unavailable"
    assert summary["upstream_direct_families"] == 0
    assert summary["upstream_composite_policy"] == "direct_upstream_only"
    assert summary["existing_tradebot_signals_role"] == "confirmation_only"
    assert not result.families["included_in_upstream_composite"].any()
    assert set(result.families["module_role"]) == {
        "upstream_break_evidence",
        "existing_tradebot_confirmation",
    }
    assert {
        "Hyperscaler capex revisions",
        "AI supplier orders / backlog",
        "GPU / cloud compute pricing",
        "AI-specific credit spreads",
        "FINRA margin-debt rollover",
        "Frontier-lab revenue, margin, and cash burn",
    }.issubset(set(result.missing_confirmations["confirmation"]))
    assert result.signals["allocation_authority"].eq(0.0).all()


def test_fundamentals_are_filtered_by_information_availability_date() -> None:
    dates = pd.bdate_range("2025-01-01", periods=300)
    prices = pd.DataFrame(
        {"SPY": np.linspace(100, 120, len(dates)), "QQQ": np.linspace(100, 130, len(dates))},
        index=dates,
    )
    fundamentals = pd.DataFrame(
        [
            {
                "available_date": "2025-06-01",
                "period_end": "2025-03-31",
                "entity": "MSFT",
                "metric": "capex_ocf",
                "value": 0.50,
                "source_quality": "A",
            },
            {
                "available_date": "2026-01-15",
                "period_end": "2025-12-31",
                "entity": "MSFT",
                "metric": "capex_ocf",
                "value": 0.95,
                "source_quality": "A",
            },
        ]
    )

    result = run_capital_cycle_break_monitor(
        prices=prices,
        fundamentals=fundamentals,
        as_of="2025-12-31",
    )
    signal = result.signals[result.signals["signal"].eq("capex_ocf")].iloc[0]

    assert signal["available"]
    assert signal["latest_value"] == 0.50
