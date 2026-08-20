from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

CAPITAL_CYCLE_FAMILIES: Final[tuple[str, ...]] = (
    "capital_intensity_overbuild",
    "financing_reflexivity",
    "monetization_efficiency",
    "infrastructure_demand_rollover",
    "theme_market_break",
    "margin_leverage",
    "rates_capital_competition",
    "equity_concentration",
)

# This module adds evidence that the rest of Trade-bot does not already own.
# Market prices, broad credit, rates, leverage, and concentration remain visible
# as confirmation, but cannot manufacture an "upstream" read by being recombined.
UPSTREAM_BREAK_FAMILIES: Final[tuple[str, ...]] = (
    "capital_intensity_overbuild",
    "financing_reflexivity",
    "monetization_efficiency",
    "infrastructure_demand_rollover",
)
EXISTING_TRADEBOT_CONFIRMATION_FAMILIES: Final[tuple[str, ...]] = (
    "theme_market_break",
    "margin_leverage",
    "rates_capital_competition",
    "equity_concentration",
)

VULNERABILITY_WEIGHTS: Final[dict[str, float]] = {
    "capital_intensity_overbuild": 0.40,
    "financing_reflexivity": 0.25,
    "monetization_efficiency": 0.35,
}
ROLLOVER_WEIGHTS: Final[dict[str, float]] = {
    "infrastructure_demand_rollover": 0.55,
    "financing_reflexivity": 0.25,
    "monetization_efficiency": 0.20,
}

FAMILY_METADATA: Final[dict[str, dict[str, str]]] = {
    "capital_intensity_overbuild": {
        "dimension": "vulnerability",
        "role": "structural vulnerability",
        "typical_lead": "12-36m hypothesis",
        "description": "Capital commitments relative to cash flow and revenue, including their first and second derivatives.",
    },
    "financing_reflexivity": {
        "dimension": "both",
        "role": "vulnerability / rollover",
        "typical_lead": "0-18m hypothesis",
        "description": "Funding demand, broad credit price, and structured-finance event pressure.",
    },
    "monetization_efficiency": {
        "dimension": "vulnerability",
        "role": "economic vulnerability",
        "typical_lead": "6-36m hypothesis",
        "description": "Revenue, losses, compute commitments, and model price-performance economics.",
    },
    "infrastructure_demand_rollover": {
        "dimension": "rollover",
        "role": "real-economy rollover",
        "typical_lead": "0-9m hypothesis",
        "description": "Capex revisions, orders, backlog, utilization, pricing, and supplier deceleration.",
    },
    "theme_market_break": {
        "dimension": "rollover",
        "role": "market recognition",
        "typical_lead": "0-6m hypothesis",
        "description": "AI leadership, supplier breadth, drawdown, and relative-trend deterioration.",
    },
    "margin_leverage": {
        "dimension": "both",
        "role": "leverage rollover",
        "typical_lead": "0-9m hypothesis",
        "description": "Margin-debt level, growth, momentum, and drawdown from peak.",
    },
    "rates_capital_competition": {
        "dimension": "vulnerability",
        "role": "funding-cost pressure",
        "typical_lead": "3-18m hypothesis",
        "description": "Real yields and long-duration funding costs while capital requirements remain high.",
    },
    "equity_concentration": {
        "dimension": "vulnerability",
        "role": "market vulnerability",
        "typical_lead": "6-24m hypothesis",
        "description": "Leadership concentration and narrow participation; not break evidence by itself.",
    },
}

EVENT_RULES: Final[tuple[tuple[str, str, tuple[str, ...], int], ...]] = (
    (
        "failed_or_repriced_financing",
        "financing_reflexivity",
        ("failed financing", "repriced financing", "refinancing failed", "financing pulled"),
        90,
    ),
    (
        "vendor_or_supplier_financing",
        "financing_reflexivity",
        ("vendor financing", "supplier financing", "supplier backstop", "backstop financing"),
        120,
    ),
    (
        "gpu_collateralized_financing",
        "financing_reflexivity",
        ("gpu collateral", "gpu-backed", "gpu backed"),
        120,
    ),
    (
        "large_datacenter_lease",
        "capital_intensity_overbuild",
        ("data center lease", "datacenter lease", "long-term lease"),
        180,
    ),
    (
        "project_cancellation",
        "infrastructure_demand_rollover",
        ("project cancelled", "project canceled", "data center cancelled", "datacenter canceled"),
        180,
    ),
    (
        "contract_renegotiation",
        "infrastructure_demand_rollover",
        ("contract renegotiat", "commitment renegotiat", "capacity renegotiat"),
        120,
    ),
    (
        "capex_guidance_reduction",
        "infrastructure_demand_rollover",
        ("capex cut", "capex reduction", "lowered capex", "reduced capital spending"),
        120,
    ),
    (
        "frontier_monetization_deterioration",
        "monetization_efficiency",
        ("cash burn", "operating loss", "missed revenue", "price war", "model price cut"),
        120,
    ),
)


@dataclass(frozen=True)
class CapitalCycleBreakRun:
    summary: pd.DataFrame
    families: pd.DataFrame
    signals: pd.DataFrame
    events: pd.DataFrame
    missing_confirmations: pd.DataFrame
    lead_times: pd.DataFrame

    @property
    def confirmations(self) -> pd.DataFrame:
        return self.missing_confirmations


def run_capital_cycle_break_monitor(
    *,
    prices: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    news_triage: pd.DataFrame | None = None,
    as_of: object | None = None,
) -> CapitalCycleBreakRun:
    """Canonical Cycle Tracker entry point for the research-only monitor."""

    prepared_fundamentals = _prepare_point_in_time_fundamentals(fundamentals, as_of=as_of)
    structured_events = score_capital_cycle_events(events, as_of=as_of)
    base = build_capital_cycle_break_monitor(
        prices=prices,
        macro_data=macro_data,
        fundamentals=prepared_fundamentals,
        news_triage=news_triage,
        as_of=as_of,
    )
    event_frames = [frame for frame in (base.events, structured_events) if not frame.empty]
    combined_events = (
        pd.concat(event_frames, ignore_index=True, sort=False)
        if event_frames
        else structured_events.copy()
    ).drop_duplicates(
        subset=["event_date", "entity", "event_type", "cycle_family"], keep="last"
    )
    signals = _signal_audit_frame(
        base.families,
        prepared_fundamentals,
        combined_events,
    )
    families = _family_audit_frame(base.families, signals)
    summary = _summary_from_family_audit(
        base.summary,
        families,
        prices=prices,
        macro_data=macro_data,
        fundamentals=prepared_fundamentals,
    )
    missing = base.missing_confirmations.copy()
    implication = _implication(
        str(summary.iloc[0].get("capital_cycle_state", "insufficient_data")),
        _safe_float(summary.iloc[0].get("capital_vulnerability"), np.nan),
        _safe_float(summary.iloc[0].get("rollover_impulse"), np.nan),
        missing,
    )
    summary.loc[:, "current_implication"] = implication
    summary.loc[:, "model_role"] = "explainer_research_only"
    summary.loc[:, "promotion_status"] = "explainer_research_only"
    summary.loc[:, "allocation_authority"] = 0.0
    return CapitalCycleBreakRun(
        summary=summary,
        families=families,
        signals=signals,
        events=combined_events,
        missing_confirmations=missing,
        lead_times=_lead_time_ladder(),
    )


def _signal_audit_frame(
    families: pd.DataFrame,
    fundamentals: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not fundamentals.empty:
        numeric_columns = [
            column
            for column in fundamentals.columns
            if column not in {"period_end", "entity", "ticker", "source", "available_date"}
        ]
        grouped = _fundamental_totals(fundamentals)
        for column in numeric_columns:
            series = pd.to_numeric(grouped.get(column), errors="coerce").dropna()
            score_series = _fundamental_signal_score(column, series)
            derivative = derivative_snapshot(score_series)
            mapped_family = _fundamental_signal_family(column)
            rows.append(
                {
                    "signal": column,
                    "cycle_family": mapped_family or "unmapped_direct_evidence",
                    "available": not series.empty,
                    "latest_value": float(series.iloc[-1]) if not series.empty else np.nan,
                    "score": derivative["score"],
                    "change_1m": derivative["change_1m"],
                    "change_3m": derivative["change_3m"],
                    "acceleration": derivative["acceleration"],
                    "derivative_state": derivative["derivative_state"],
                    "direct": mapped_family is not None,
                    "latest_period": (
                        pd.Timestamp(series.index[-1]).date().isoformat() if not series.empty else ""
                    ),
                    "source_type": (
                        "point_in_time_fundamental"
                        if mapped_family is not None
                        else "unmapped_point_in_time_evidence"
                    ),
                    "allocation_authority": 0.0,
                }
            )
    for _, row in families.iterrows():
        rows.append(
            {
                "signal": f"family::{row.get('family')}",
                "cycle_family": row.get("family"),
                "available": str(row.get("data_status")) != "missing",
                "latest_value": row.get("score"),
                "score": row.get("score"),
                "change_1m": row.get("change_1m"),
                "change_3m": row.get("change_3m"),
                "acceleration": row.get("acceleration"),
                "derivative_state": row.get("derivative_state"),
                "direct": str(row.get("data_status")) == "direct",
                "latest_period": row.get("as_of_date", ""),
                "source_type": row.get("source_type", "unavailable"),
                "allocation_authority": 0.0,
            }
        )
    for family, group in events.groupby("cycle_family") if not events.empty else []:
        rows.append(
            {
                "signal": f"event_pressure::{family}",
                "cycle_family": family,
                "available": True,
                "latest_value": float(
                    pd.to_numeric(group["decayed_contribution"], errors="coerce").sum()
                ),
                "score": float(
                    pd.to_numeric(group["decayed_contribution"], errors="coerce").sum()
                ),
                "change_1m": np.nan,
                "change_3m": np.nan,
                "acceleration": np.nan,
                "derivative_state": "event_pressure",
                "direct": False,
                "latest_period": str(group["event_date"].max()),
                "source_type": "governed_structured_events",
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _fundamental_signal_family(signal: str) -> str | None:
    normalized = signal.lower()
    if any(token in normalized for token in ("backlog", "order", "guidance", "utilization")):
        return "infrastructure_demand_rollover"
    if any(token in normalized for token in ("loss", "compute", "revenue")):
        return "monetization_efficiency"
    if signal in {
        "capex",
        "operating_cash_flow",
        "capex_ocf",
        "capex_revenue",
        "purchase_commitments",
        "lease_commitments",
        "purchase_commitments_ocf",
    }:
        return "capital_intensity_overbuild"
    return None


def _fundamental_signal_score(signal: str, series: pd.Series) -> pd.Series:
    thresholds: dict[str, tuple[float, float]] = {
        "capex_ocf": (0.35, 1.25),
        "capex_revenue": (0.12, 0.45),
        "purchase_commitments_ocf": (0.50, 3.00),
        "operating_loss_revenue": (0.10, 1.50),
        "compute_commitments_revenue": (0.50, 5.00),
        "supplier_backlog_growth": (0.20, -0.20),
        "infrastructure_orders_growth": (0.20, -0.20),
        "capex_guidance_revision": (0.10, -0.20),
        "compute_price_change": (0.05, -0.25),
        "capacity_utilization": (0.90, 0.55),
    }
    calm, stressed = thresholds.get(signal, (0.0, 1.0))
    return _threshold_series(series, calm, stressed)


def _family_audit_frame(families: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    audited = families.copy()
    if audited.empty:
        return audited
    event_rows = signals[signals["signal"].astype(str).str.startswith("event_pressure::")]
    direct_rows = signals[
        signals.get("direct", pd.Series(False, index=signals.index)).fillna(False)
        & ~signals["signal"].astype(str).str.startswith("family::")
    ]
    for family, group in direct_rows.groupby("cycle_family"):
        mask = audited["family"].eq(family)
        scores = pd.to_numeric(group.get("score"), errors="coerce").dropna()
        if not mask.any() or scores.empty:
            continue
        audited.loc[mask, "score"] = _clip01(float(scores.mean()))
        audited.loc[mask, "data_status"] = "direct"
        audited.loc[mask, "source_type"] = "point_in_time_fundamental"
    for _, event in event_rows.iterrows():
        family = str(event.get("cycle_family"))
        mask = audited["family"].eq(family)
        if not mask.any():
            continue
        pressure = _clip01(_safe_float(event.get("latest_value"), 0.0))
        current = _safe_float(audited.loc[mask, "score"].iloc[0], np.nan)
        audited.loc[mask, "score"] = pressure if pd.isna(current) else _clip01(
            0.85 * current + 0.15 * pressure
        )
        if str(audited.loc[mask, "data_status"].iloc[0]) == "missing":
            audited.loc[mask, "data_status"] = "event_only"
            audited.loc[mask, "source_type"] = "governed_structured_events"
    audited["state"] = pd.to_numeric(audited["score"], errors="coerce").map(_score_label)
    audited["available_signals"] = audited["data_status"].ne("missing").astype(int)
    audited["total_signals"] = 1
    audited["required_coverage"] = audited["dimension"].map(
        {"vulnerability": "vulnerability", "rollover": "rollover", "both": "both"}
    )
    audited["direct_signal_available"] = audited["data_status"].eq("direct")
    audited["module_role"] = audited["family"].map(
        lambda family: (
            "upstream_break_evidence"
            if family in UPSTREAM_BREAK_FAMILIES
            else "existing_tradebot_confirmation"
        )
    )
    audited["included_in_upstream_composite"] = (
        audited["family"].isin(UPSTREAM_BREAK_FAMILIES)
        & audited["data_status"].eq("direct")
    )
    audited["conceptual_lead"] = audited["typical_lead"]
    audited["description"] = audited["interpretation"]
    return audited


def _summary_from_family_audit(
    base_summary: pd.DataFrame,
    families: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    macro_data: pd.DataFrame | None,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    vulnerability, vulnerability_coverage = _supported_weighted_score(
        families, VULNERABILITY_WEIGHTS, eligible_statuses={"direct"}
    )
    rollover, rollover_coverage = _supported_weighted_score(
        families, ROLLOVER_WEIGHTS, eligible_statuses={"direct"}
    )
    indexed = families.set_index("family")
    direct_real = sum(
        str(indexed.at[family, "data_status"]) == "direct"
        and _safe_float(indexed.at[family, "score"], 0.0) >= 0.60
        for family in (
            "infrastructure_demand_rollover",
            "capital_intensity_overbuild",
            "monetization_efficiency",
        )
        if family in indexed.index
    )
    direct_real_available = sum(
        str(indexed.at[family, "data_status"]) == "direct"
        for family in (
            "infrastructure_demand_rollover",
            "capital_intensity_overbuild",
            "monetization_efficiency",
        )
        if family in indexed.index
    )
    independent_rollover = sum(
        str(indexed.at[family, "data_status"]) in {"direct", "broad_proxy"}
        and _safe_float(indexed.at[family, "score"], 0.0) >= 0.60
        for family in (
            "infrastructure_demand_rollover",
            "financing_reflexivity",
            "theme_market_break",
            "margin_leverage",
        )
        if family in indexed.index
    )
    state = _capital_cycle_state(
        vulnerability,
        rollover,
        vulnerability_coverage,
        rollover_coverage,
        direct_real,
        independent_rollover,
    )
    if direct_real_available == 0:
        state = "insufficient_data"
    summary = base_summary.copy()
    summary.loc[:, "capital_vulnerability"] = vulnerability
    summary.loc[:, "rollover_impulse"] = rollover
    summary.loc[:, "capital_cycle_state"] = state
    summary.loc[:, "vulnerability_coverage"] = vulnerability_coverage
    summary.loc[:, "rollover_coverage"] = rollover_coverage
    summary.loc[:, "direction"] = _composite_direction(families)
    summary.loc[:, "transition_pressure"] = _transition_pressure(state)
    summary.loc[:, "supported_families"] = int(families["data_status"].ne("missing").sum())
    summary.loc[:, "direct_families"] = int(families["data_status"].eq("direct").sum())
    summary.loc[:, "upstream_direct_families"] = int(
        families["included_in_upstream_composite"].sum()
    )
    summary.loc[:, "real_economy_confirmations"] = direct_real
    summary.loc[:, "direct_real_economy_families"] = direct_real_available
    summary.loc[:, "independent_rollover_confirmations"] = independent_rollover
    summary.loc[:, "upstream_composite_policy"] = "direct_upstream_only"
    summary.loc[:, "existing_tradebot_signals_role"] = "confirmation_only"
    return summary


def _missing_confirmation_audit(
    confirmations: pd.DataFrame,
    families: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    labels = {
        "capex_revisions": "Hyperscaler capex revisions",
        "supplier_orders_backlog": "AI supplier orders / backlog",
        "compute_price_utilization": "GPU / cloud compute pricing",
        "frontier_lab_economics": "Frontier-lab revenue, margin, and cash burn",
        "ai_specific_credit": "AI-specific credit spreads",
        "margin_debt": "FINRA margin-debt rollover",
        "theme_market_break": "AI leadership market break",
        "broad_credit": "Broad credit confirmation",
        "broad_trend": "Broad trend confirmation",
    }
    family_scores = families.set_index("family")["score"].to_dict()
    score_map = {
        "capex_revisions": family_scores.get("infrastructure_demand_rollover"),
        "supplier_orders_backlog": family_scores.get("infrastructure_demand_rollover"),
        "compute_price_utilization": family_scores.get("infrastructure_demand_rollover"),
        "frontier_lab_economics": family_scores.get("monetization_efficiency"),
        "ai_specific_credit": family_scores.get("financing_reflexivity"),
        "margin_debt": family_scores.get("margin_leverage"),
        "theme_market_break": family_scores.get("theme_market_break"),
        "broad_credit": family_scores.get("financing_reflexivity"),
        "broad_trend": np.nan,
    }
    rows = []
    reverse_labels = {label: key for key, label in labels.items()}
    for _, row in confirmations.iterrows():
        key = str(row.get("confirmation"))
        canonical_key = reverse_labels.get(key, key)
        available = bool(row.get("available", False))
        current_read = _safe_float(score_map.get(canonical_key), np.nan)
        status = (
            "missing"
            if not available
            else "confirming"
            if pd.notna(current_read) and current_read >= 0.60
            else "not_confirming"
        )
        rows.append(
            {
                "confirmation": labels.get(
                    canonical_key, key.replace("_", " ").title()
                ),
                "confirmation_type": row.get("confirmation_type", "unclassified"),
                "observed": available,
                "available": available,
                "status": status,
                "current_read": current_read,
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _lead_time_ladder() -> pd.DataFrame:
    rows = [
        ("Capital commitments", "vulnerability", "12-36m"),
        ("Monetization / efficiency", "vulnerability", "6-36m"),
        ("Financing complexity", "vulnerability / rollover", "6-18m"),
        ("Long rates / capital competition", "vulnerability / trigger", "3-18m"),
        ("Margin leverage rollover", "early rollover", "0-9m"),
        ("Capex revisions and supplier orders", "real-economy rollover", "0-9m"),
        ("Theme relative-strength break", "market recognition", "0-6m"),
        ("Broad credit, breadth, trend, volatility", "break confirmation", "coincident"),
    ]
    result = pd.DataFrame(rows, columns=["signal_family", "role", "conceptual_lead"])
    result["empirical_status"] = "conceptual_not_fitted"
    result["allocation_authority"] = 0.0
    return result


def build_capital_cycle_break_monitor(
    *,
    prices: pd.DataFrame,
    macro_data: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    news_triage: pd.DataFrame | None = None,
    as_of: object | None = None,
) -> CapitalCycleBreakRun:
    """Build a transparent, research-only capital-cycle break monitor.

    Missing direct inputs remain missing. Scores are normalized only over supported
    families, and every output has zero allocation authority.
    """

    clean_prices = _clean_frame(prices, as_of=as_of)
    clean_macro = _clean_frame(macro_data, as_of=as_of)
    clean_fundamentals = _clean_fundamentals(fundamentals, as_of=as_of)
    as_of_date = _resolve_as_of(clean_prices, clean_macro, clean_fundamentals, as_of)
    events = build_capital_cycle_events(news_triage, as_of=as_of_date)

    family_rows = [
        _capital_intensity_family(clean_fundamentals),
        _financing_family(clean_macro, events),
        _monetization_family(clean_fundamentals, events),
        _infrastructure_family(clean_prices, clean_fundamentals, events),
        _theme_market_family(clean_prices),
        _margin_leverage_family(clean_macro),
        _rates_family(clean_macro),
        _concentration_family(clean_prices),
    ]
    families = pd.DataFrame(family_rows)
    if families.empty:
        families = _empty_family_frame()
    families.insert(0, "as_of_date", as_of_date)
    families["module_role"] = families["family"].map(
        lambda family: (
            "upstream_break_evidence"
            if family in UPSTREAM_BREAK_FAMILIES
            else "existing_tradebot_confirmation"
        )
    )
    families["included_in_upstream_composite"] = (
        families["family"].isin(UPSTREAM_BREAK_FAMILIES)
        & families["data_status"].eq("direct")
    )
    families["allocation_authority"] = 0.0

    vulnerability, vulnerability_coverage = _supported_weighted_score(
        families, VULNERABILITY_WEIGHTS, eligible_statuses={"direct"}
    )
    rollover, rollover_coverage = _supported_weighted_score(
        families, ROLLOVER_WEIGHTS, eligible_statuses={"direct"}
    )
    state = _capital_cycle_state_from_families(families, vulnerability, rollover)
    transition_pressure = _transition_pressure(state)
    direction = _composite_direction(families)
    confirmations = _confirmation_frame(
        clean_prices,
        clean_macro,
        clean_fundamentals,
        families,
    )
    implication = _implication(state, vulnerability, rollover, confirmations)
    supported = int(families["data_status"].ne("missing").sum())
    direct = int(families["data_status"].eq("direct").sum())
    summary = pd.DataFrame(
        [
            {
                "as_of_date": as_of_date,
                "capital_vulnerability": vulnerability,
                "rollover_impulse": rollover,
                "capital_cycle_state": state,
                "direction": direction,
                "transition_pressure": transition_pressure,
                "supported_families": supported,
                "direct_families": direct,
                "upstream_direct_families": int(
                    families["included_in_upstream_composite"].sum()
                ),
                "total_families": len(CAPITAL_CYCLE_FAMILIES),
                "vulnerability_coverage": vulnerability_coverage,
                "rollover_coverage": rollover_coverage,
                "missing_confirmations": int(confirmations["available"].eq(False).sum()),  # noqa: E712
                "allocation_authority": 0.0,
                "promotion_status": "explainer_research_only",
                "upstream_composite_policy": "direct_upstream_only",
                "existing_tradebot_signals_role": "confirmation_only",
                "implication": implication,
            }
        ]
    )
    return CapitalCycleBreakRun(
        summary=summary,
        families=families,
        signals=_signal_audit_frame(families, clean_fundamentals, events),
        events=events,
        missing_confirmations=_missing_confirmation_audit(
            confirmations,
            families,
            clean_fundamentals,
        ),
        lead_times=_lead_time_ladder(),
    )


def build_capital_cycle_events(
    news_triage: pd.DataFrame | None,
    *,
    as_of: object | None,
) -> pd.DataFrame:
    columns = [
        "event_date",
        "entity",
        "event_type",
        "cycle_family",
        "title",
        "source",
        "severity",
        "confidence",
        "source_quality",
        "decay_half_life_days",
        "age_days",
        "decayed_contribution",
        "allocation_authority",
        "url",
    ]
    if not isinstance(news_triage, pd.DataFrame) or news_triage.empty:
        return pd.DataFrame(columns=columns)
    cutoff = pd.to_datetime(as_of, errors="coerce")
    rows: list[dict[str, object]] = []
    for _, row in news_triage.iterrows():
        event_date = pd.to_datetime(row.get("published_at"), errors="coerce", utc=True)
        if pd.isna(event_date):
            continue
        event_day = event_date.tz_convert(None).normalize()
        if pd.notna(cutoff) and event_day > cutoff.normalize():
            continue
        text = " ".join(
            str(row.get(column, ""))
            for column in ("title", "summary", "category", "topics", "risk_channels")
        ).lower()
        matched = next(
            (
                (event_type, family, half_life)
                for event_type, family, patterns, half_life in EVENT_RULES
                if any(pattern in text for pattern in patterns)
            ),
            None,
        )
        if matched is None:
            continue
        event_type, family, half_life = matched
        severity = _clip01(_safe_float(row.get("urgency_score"), 0.35))
        confidence = _confidence_value(row.get("confidence"))
        age_days = max(0, int((cutoff.normalize() - event_day).days)) if pd.notna(cutoff) else 0
        decayed = severity * confidence * (0.5 ** (age_days / float(half_life)))
        rows.append(
            {
                "event_date": event_day.date().isoformat(),
                "entity": _event_entity(row),
                "event_type": event_type,
                "cycle_family": family,
                "title": str(row.get("title", "")),
                "source": str(row.get("source", "")),
                "severity": severity,
                "confidence": confidence,
                "source_quality": _source_quality(row.get("source_priority")),
                "decay_half_life_days": half_life,
                "age_days": age_days,
                "decayed_contribution": decayed,
                "allocation_authority": 0.0,
                "url": str(row.get("url", "")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["decayed_contribution", "event_date"], ascending=[False, False]
    )


def derivative_snapshot(series: pd.Series) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return {
            "score": np.nan,
            "change_1m": np.nan,
            "change_3m": np.nan,
            "acceleration": np.nan,
            "derivative_state": "unavailable",
        }
    latest_date = pd.to_datetime(clean.index.max())
    latest = float(clean.iloc[-1])
    one_month = _series_value_on_or_before(clean, latest_date - pd.Timedelta(days=31))
    two_month = _series_value_on_or_before(clean, latest_date - pd.Timedelta(days=62))
    three_month = _series_value_on_or_before(clean, latest_date - pd.Timedelta(days=92))
    change_1m = latest - one_month if pd.notna(one_month) else np.nan
    prior_change_1m = (
        one_month - two_month if pd.notna(one_month) and pd.notna(two_month) else np.nan
    )
    acceleration = change_1m - prior_change_1m if pd.notna(prior_change_1m) else np.nan
    change_3m = latest - three_month if pd.notna(three_month) else np.nan
    return {
        "score": latest,
        "change_1m": change_1m,
        "change_3m": change_3m,
        "acceleration": acceleration,
        "derivative_state": _derivative_state(change_1m, acceleration),
    }


def classify_derivative_state(
    short_change: float,
    medium_change: float,
    acceleration: float,
) -> str:
    """Classify direction using transparent, scale-normalized changes."""

    values = (short_change, medium_change, acceleration)
    if all(pd.isna(value) for value in values):
        return "unavailable"
    short = _safe_float(short_change, 0.0)
    medium = _safe_float(medium_change, 0.0)
    accel = _safe_float(acceleration, 0.0)
    if short >= 0.08 and medium >= 0.05 and accel >= 0.03:
        return "worsening_fast"
    if short >= 0.025 and medium >= 0.0:
        return "worsening"
    if short <= -0.08 and medium <= -0.05 and accel <= -0.03:
        return "improving_fast"
    if short <= -0.025 and medium <= 0.0:
        return "improving"
    return "flat"


def score_capital_cycle_events(
    events: pd.DataFrame | None,
    *,
    as_of: object | None,
) -> pd.DataFrame:
    """Score a governed event ledger with point-in-time filtering and decay."""

    columns = [
        "event_date",
        "event_type",
        "entity",
        "direction",
        "severity",
        "confidence",
        "source_quality",
        "cycle_family",
        "decay_half_life_days",
        "age_days",
        "decay_weight",
        "decayed_contribution",
        "allocation_authority",
    ]
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame(columns=columns)
    clean = events.copy()
    clean["event_date"] = pd.to_datetime(clean.get("event_date"), errors="coerce").dt.tz_localize(None)
    clean = clean.dropna(subset=["event_date"])
    cutoff = pd.to_datetime(as_of, errors="coerce")
    if pd.notna(cutoff):
        cutoff = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
        clean = clean[clean["event_date"] <= cutoff]
    elif not clean.empty:
        cutoff = clean["event_date"].max()
    if clean.empty:
        return pd.DataFrame(columns=columns)
    clean["severity"] = pd.to_numeric(clean.get("severity", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    clean["confidence"] = clean.get("confidence", 0.5).map(_confidence_value)
    clean["decay_half_life_days"] = (
        pd.to_numeric(clean.get("decay_half_life_days", 90), errors="coerce")
        .fillna(90.0)
        .clip(lower=1.0)
    )
    clean["age_days"] = (cutoff - clean["event_date"]).dt.days.clip(lower=0)
    clean["decay_weight"] = 0.5 ** (
        clean["age_days"] / clean["decay_half_life_days"]
    )
    direction = clean.get("direction", "risk").astype(str).str.lower()
    sign = np.where(direction.isin({"constructive", "relief", "positive"}), -1.0, 1.0)
    clean["decayed_contribution"] = (
        sign * clean["severity"] * clean["confidence"] * clean["decay_weight"]
    )
    clean["allocation_authority"] = 0.0
    for column in columns:
        if column not in clean:
            clean[column] = "" if column in {"event_type", "entity", "direction", "source_quality", "cycle_family"} else np.nan
    clean["event_date"] = clean["event_date"].dt.date.astype(str)
    return clean[columns].sort_values(
        ["decayed_contribution", "event_date"], ascending=[False, False]
    ).reset_index(drop=True)


def _prepare_point_in_time_fundamentals(
    fundamentals: pd.DataFrame | None,
    *,
    as_of: object | None,
) -> pd.DataFrame:
    if not isinstance(fundamentals, pd.DataFrame) or fundamentals.empty:
        return pd.DataFrame()
    clean = fundamentals.copy()
    cutoff = pd.to_datetime(as_of, errors="coerce")
    if pd.notna(cutoff):
        cutoff = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    if "available_date" in clean:
        clean["available_date"] = pd.to_datetime(
            clean["available_date"], errors="coerce"
        ).dt.tz_localize(None)
        clean = clean.dropna(subset=["available_date"])
        if pd.notna(cutoff):
            clean = clean[clean["available_date"] <= cutoff]
    if "period_end" not in clean:
        return pd.DataFrame()
    clean["period_end"] = pd.to_datetime(clean["period_end"], errors="coerce").dt.tz_localize(None)
    clean = clean.dropna(subset=["period_end"])
    if {"metric", "value"}.issubset(clean.columns):
        index_columns = [column for column in ("period_end", "entity") if column in clean]
        ordered = clean.sort_values(
            [column for column in ("available_date", "period_end") if column in clean]
        )
        wide = ordered.pivot_table(
            index=index_columns,
            columns="metric",
            values="value",
            aggfunc="last",
        ).reset_index()
        wide.columns.name = None
        return wide.sort_values("period_end").reset_index(drop=True)
    return clean.sort_values("period_end").reset_index(drop=True)


def _capital_intensity_family(fundamentals: pd.DataFrame) -> dict[str, object]:
    required = {"capex", "operating_cash_flow", "revenue"}
    if fundamentals.empty or not required.issubset(fundamentals.columns):
        return _missing_family(
            "capital_intensity_overbuild",
            "Quarterly capex, OCF, revenue, purchase commitments, and leases are not ingested.",
        )
    grouped = _fundamental_totals(fundamentals)
    capex = grouped["capex"].abs()
    ocf = grouped["operating_cash_flow"].abs().replace(0.0, np.nan)
    revenue = grouped["revenue"].abs().replace(0.0, np.nan)
    capex_ocf = capex / ocf
    capex_revenue = capex / revenue
    capex_growth = capex.pct_change(4).replace([np.inf, -np.inf], np.nan)
    series = _mean_series(
        [
            _threshold_series(capex_ocf, 0.35, 1.25),
            _threshold_series(capex_revenue, 0.12, 0.45),
            _threshold_series(capex_growth, 0.0, 0.60),
        ]
    )
    if {"purchase_commitments", "lease_commitments"}.intersection(grouped.columns):
        commitments = pd.Series(0.0, index=grouped.index)
        for column in ("purchase_commitments", "lease_commitments"):
            if column in grouped:
                commitments = commitments.add(grouped[column].abs(), fill_value=0.0)
        series = _mean_series([series, _threshold_series(commitments / ocf, 0.50, 3.0)])
    return _family_from_series(
        "capital_intensity_overbuild",
        series,
        data_status="direct",
        source_type="filing_fundamentals",
        interpretation="Direct capex intensity and growth pressure; vulnerability only without rollover confirmation.",
        missing_inputs="Consensus capex revisions remain unavailable unless supplied separately.",
    )


def _financing_family(macro: pd.DataFrame, events: pd.DataFrame) -> dict[str, object]:
    credit_series = []
    for column, calm, stressed in (
        ("BAMLC0A0CM", 0.70, 2.20),
        ("BAMLH0A0HYM2", 2.50, 7.50),
        ("BAMLC0A4CBBB", 1.00, 3.50),
    ):
        if column in macro:
            credit_series.append(_threshold_series(macro[column], calm, stressed))
    event_score = _event_family_score(events, "financing_reflexivity")
    if not credit_series and event_score <= 0:
        return _missing_family(
            "financing_reflexivity",
            "No broad credit-spread series or classified financing events are available.",
        )
    series = _mean_series(credit_series) if credit_series else pd.Series(dtype=float)
    status = "broad_proxy" if credit_series else "event_only"
    source = "broad_credit_spreads" if credit_series else "decaying_news_events"
    row = _family_from_series(
        "financing_reflexivity",
        series,
        data_status=status,
        source_type=source,
        interpretation="Broad credit price and classified financing complexity; not AI-specific credit.",
        missing_inputs="AI-project spreads, technology issuance, converts, and private-credit terms.",
    )
    return _blend_event_score(row, event_score)


def _monetization_family(
    fundamentals: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, object]:
    event_score = _event_family_score(events, "monetization_efficiency")
    columns = {"revenue", "operating_loss", "compute_commitments"}
    if (
        fundamentals.empty
        or "revenue" not in fundamentals
        or not columns.intersection(fundamentals.columns).difference({"revenue"})
    ):
        if event_score <= 0:
            return _missing_family(
                "monetization_efficiency",
                "No standardized frontier-lab revenue, losses, compute commitments, or model-price series.",
            )
        row = _missing_family(
            "monetization_efficiency",
            "Only decaying event evidence is available; direct private-company economics are missing.",
        )
        row.update(
            {
                "score": event_score,
                "data_status": "event_only",
                "source_type": "decaying_news_events",
                "derivative_state": "event_pressure",
            }
        )
        return row
    grouped = _fundamental_totals(fundamentals)
    revenue = grouped["revenue"].abs().replace(0.0, np.nan)
    series_parts = []
    if "operating_loss" in grouped:
        series_parts.append(
            _threshold_series(grouped["operating_loss"].abs() / revenue, 0.10, 1.50)
        )
    if "compute_commitments" in grouped:
        series_parts.append(
            _threshold_series(grouped["compute_commitments"].abs() / revenue, 0.50, 5.0)
        )
    revenue_growth = revenue.pct_change(4)
    series_parts.append(_threshold_series(-revenue_growth.diff(), -0.05, 0.30))
    row = _family_from_series(
        "monetization_efficiency",
        _mean_series(series_parts),
        data_status="direct",
        source_type="filing_fundamentals",
        interpretation="Direct monetization pressure from losses, commitments, and revenue deceleration.",
        missing_inputs="Comparable model price-performance and workload-level utilization series.",
    )
    return _blend_event_score(row, event_score)


def _infrastructure_family(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, object]:
    direct_parts = []
    if not fundamentals.empty:
        grouped = _fundamental_totals(fundamentals)
        for column in ("supplier_backlog", "capex_guidance", "infrastructure_orders"):
            if column in grouped:
                growth = grouped[column].pct_change(4)
                direct_parts.append(_threshold_series(-growth.diff(), -0.05, 0.35))
    proxy = _supplier_rollover_series(prices)
    event_score = _event_family_score(events, "infrastructure_demand_rollover")
    if direct_parts:
        row = _family_from_series(
            "infrastructure_demand_rollover",
            _mean_series(direct_parts),
            data_status="direct",
            source_type="orders_backlog_guidance",
            interpretation="Direct second-derivative pressure in orders, backlog, or capex guidance.",
            missing_inputs="GPU rental pricing, utilization, lead times, and datacenter vacancy.",
        )
    elif not proxy.empty:
        row = _family_from_series(
            "infrastructure_demand_rollover",
            proxy,
            data_status="broad_proxy",
            source_type="supplier_vs_hyperscaler_prices",
            interpretation="Market proxy for supplier deceleration; not direct infrastructure demand.",
            missing_inputs="Capex revisions, supplier orders/backlog, compute pricing, and utilization.",
        )
    elif event_score > 0:
        row = _missing_family(
            "infrastructure_demand_rollover",
            "Only decaying cancellation or guidance-event evidence is available.",
        )
        row.update(
            {
                "score": event_score,
                "data_status": "event_only",
                "source_type": "decaying_news_events",
                "derivative_state": "event_pressure",
            }
        )
    else:
        return _missing_family(
            "infrastructure_demand_rollover",
            "No orders, backlog, capex revisions, compute pricing, utilization, or usable price proxy.",
        )
    return _blend_event_score(row, event_score)


def _theme_market_family(prices: pd.DataFrame) -> dict[str, object]:
    series = _theme_market_break_series(prices)
    if series.empty:
        return _missing_family(
            "theme_market_break",
            "AI supplier, semiconductor, and hyperscaler price baskets are unavailable.",
        )
    return _family_from_series(
        "theme_market_break",
        series,
        data_status="broad_proxy",
        source_type="market_prices",
        interpretation="Leadership-relative trend and drawdown confirmation from observable prices.",
        missing_inputs="AI basket breadth and good-news/bad-price earnings response history.",
    )


def _margin_leverage_family(macro: pd.DataFrame) -> dict[str, object]:
    candidates = [column for column in macro if "MARGIN" in str(column).upper()]
    if not candidates:
        return _missing_family(
            "margin_leverage",
            "Timely FINRA margin-debt history is not present in the macro store.",
        )
    level = pd.to_numeric(macro[candidates[0]], errors="coerce")
    growth = level.pct_change(12 if len(level) < 500 else 252)
    drawdown = level / level.rolling(min(36, max(6, len(level) // 4))).max() - 1.0
    series = _mean_series(
        [
            _threshold_series(growth, 0.0, 0.35),
            _threshold_series(-growth.diff(), -0.02, 0.20),
            _threshold_series(-drawdown, 0.0, 0.20),
        ]
    )
    return _family_from_series(
        "margin_leverage",
        series,
        data_status="direct",
        source_type="finra_margin_debt",
        interpretation="Leverage willingness and rollover from reported margin debt.",
        missing_inputs="Positioning and prime-broker leverage remain unavailable.",
    )


def _rates_family(macro: pd.DataFrame) -> dict[str, object]:
    parts = []
    if "DFII10" in macro:
        real_yield = pd.to_numeric(macro["DFII10"], errors="coerce")
        parts.extend(
            [
                _threshold_series(real_yield, 0.50, 2.75),
                _threshold_series(real_yield.diff(21), -0.20, 0.75),
            ]
        )
    if "DGS30" in macro:
        long_yield = pd.to_numeric(macro["DGS30"], errors="coerce")
        parts.append(_threshold_series(long_yield, 2.50, 6.00))
    if not parts:
        return _missing_family(
            "rates_capital_competition",
            "10Y real-yield and 30Y nominal-yield series are unavailable.",
        )
    return _family_from_series(
        "rates_capital_competition",
        _mean_series(parts),
        data_status="direct",
        source_type="fred_rates",
        interpretation="Long-duration funding-cost pressure; dangerous only when commitments stay elevated.",
        missing_inputs="AI-project financing spreads and technology-sector cost of debt.",
    )


def _concentration_family(prices: pd.DataFrame) -> dict[str, object]:
    if not {"QQQ", "RSP", "SPY"}.issubset(prices.columns):
        return _missing_family(
            "equity_concentration",
            "QQQ, RSP, and SPY are required for the concentration proxy.",
        )
    qqq_rsp = (prices["QQQ"] / prices["RSP"]).pct_change(126)
    qqq_spy = (prices["QQQ"] / prices["SPY"]).pct_change(126)
    series = _mean_series(
        [
            _threshold_series(qqq_rsp, -0.04, 0.20),
            _threshold_series(qqq_spy, -0.04, 0.18),
        ]
    )
    return _family_from_series(
        "equity_concentration",
        series,
        data_status="broad_proxy",
        source_type="market_prices",
        interpretation="Narrow leadership vulnerability; it cannot confirm a break by itself.",
        missing_inputs="Constituent-level AI leadership breadth and issuance concentration.",
    )


def _theme_market_break_series(prices: pd.DataFrame) -> pd.Series:
    if prices.empty or "SPY" not in prices:
        return pd.Series(dtype=float)
    parts = []
    for ticker in ("SMH", "SOXX", "QQQ"):
        if ticker not in prices:
            continue
        relative = prices[ticker] / prices["SPY"]
        parts.extend(
            [
                _threshold_series(-relative.pct_change(21), -0.02, 0.10),
                _threshold_series(-relative.pct_change(63), -0.03, 0.18),
                _threshold_series(
                    -(prices[ticker] / prices[ticker].rolling(252).max() - 1.0), 0.03, 0.25
                ),
            ]
        )
    supplier = _equal_weight_index(
        prices,
        ("NVDA", "AVGO", "ANET", "VRT", "MU", "MRVL", "DELL", "SMCI"),
    )
    hyperscaler = _equal_weight_index(prices, ("MSFT", "AMZN", "GOOGL", "META", "ORCL"))
    if not supplier.empty and not hyperscaler.empty:
        parts.append(_threshold_series(-(supplier / hyperscaler).pct_change(63), -0.03, 0.20))
    return _mean_series(parts)


def _supplier_rollover_series(prices: pd.DataFrame) -> pd.Series:
    supplier = _equal_weight_index(
        prices,
        ("NVDA", "AVGO", "ANET", "VRT", "MU", "MRVL", "DELL", "SMCI"),
    )
    hyperscaler = _equal_weight_index(prices, ("MSFT", "AMZN", "GOOGL", "META", "ORCL"))
    if supplier.empty:
        return pd.Series(dtype=float)
    parts = [
        _threshold_series(-supplier.pct_change(21), -0.03, 0.15),
        _threshold_series(-(supplier.pct_change(21) - supplier.pct_change(63) / 3.0), -0.03, 0.12),
    ]
    if not hyperscaler.empty:
        parts.append(_threshold_series(-(supplier / hyperscaler).pct_change(63), -0.03, 0.20))
    return _mean_series(parts)


def _family_from_series(
    family: str,
    series: pd.Series,
    *,
    data_status: str,
    source_type: str,
    interpretation: str,
    missing_inputs: str,
) -> dict[str, object]:
    snapshot = derivative_snapshot(series)
    metadata = FAMILY_METADATA[family]
    return {
        "family": family,
        "dimension": metadata["dimension"],
        "role": metadata["role"],
        "typical_lead": metadata["typical_lead"],
        "score": snapshot["score"],
        "change_1m": snapshot["change_1m"],
        "change_3m": snapshot["change_3m"],
        "acceleration": snapshot["acceleration"],
        "derivative_state": snapshot["derivative_state"],
        "data_status": data_status,
        "source_type": source_type,
        "interpretation": interpretation,
        "missing_inputs": missing_inputs,
    }


def _missing_family(family: str, missing_inputs: str) -> dict[str, object]:
    metadata = FAMILY_METADATA[family]
    return {
        "family": family,
        "dimension": metadata["dimension"],
        "role": metadata["role"],
        "typical_lead": metadata["typical_lead"],
        "score": np.nan,
        "change_1m": np.nan,
        "change_3m": np.nan,
        "acceleration": np.nan,
        "derivative_state": "unavailable",
        "data_status": "missing",
        "source_type": "unavailable",
        "interpretation": metadata["description"],
        "missing_inputs": missing_inputs,
    }


def _blend_event_score(row: dict[str, object], event_score: float) -> dict[str, object]:
    if event_score <= 0:
        return row
    current = _safe_float(row.get("score"), np.nan)
    row["score"] = event_score if pd.isna(current) else _clip01(0.85 * current + 0.15 * event_score)
    if str(row.get("data_status")) == "missing":
        row["data_status"] = "event_only"
        row["source_type"] = "decaying_news_events"
    row["interpretation"] = (
        f"{row['interpretation']} Decaying event pressure is included with zero sizing authority."
    )
    return row


def _supported_weighted_score(
    families: pd.DataFrame,
    weights: dict[str, float],
    *,
    eligible_statuses: set[str] | None = None,
) -> tuple[float, float]:
    if families.empty:
        return float("nan"), 0.0
    indexed = families.set_index("family")
    scores = indexed["score"]

    def is_supported(family: str) -> bool:
        if pd.isna(scores.get(family)):
            return False
        if eligible_statuses is None:
            return True
        return str(indexed["data_status"].get(family)) in eligible_statuses

    supported_weight = sum(
        weight for family, weight in weights.items() if is_supported(family)
    )
    if supported_weight <= 0:
        return float("nan"), 0.0
    score = (
        sum(
            weight * float(scores[family])
            for family, weight in weights.items()
            if is_supported(family)
        )
        / supported_weight
    )
    return _clip01(score), supported_weight / sum(weights.values())


def _capital_cycle_state_from_families(
    families: pd.DataFrame,
    vulnerability: float,
    rollover: float,
) -> str:
    if pd.isna(vulnerability) and pd.isna(rollover):
        return "insufficient_data"
    indexed = families.set_index("family")
    scores = indexed["score"].to_dict()
    statuses = indexed["data_status"].to_dict()
    high_vulnerability = pd.notna(vulnerability) and vulnerability >= 0.60
    rollover_flags = sum(
        _safe_float(scores.get(family), 0.0) >= 0.45
        for family in (
            "infrastructure_demand_rollover",
            "financing_reflexivity",
            "theme_market_break",
            "margin_leverage",
        )
    )
    real_economy = max(
        (
            _safe_float(scores.get(family), 0.0)
            for family in (
                "infrastructure_demand_rollover",
                "capital_intensity_overbuild",
                "monetization_efficiency",
            )
            if statuses.get(family) == "direct"
        ),
        default=0.0,
    )
    market_confirmation = max(
        _safe_float(scores.get("theme_market_break"), 0.0),
        _safe_float(scores.get("financing_reflexivity"), 0.0),
    )
    market_flags = sum(
        _safe_float(scores.get(family), 0.0) >= 0.70
        for family in ("theme_market_break", "financing_reflexivity")
        if statuses.get(family) in {"direct", "broad_proxy"}
    )
    if real_economy >= 0.70 and market_flags >= 2 and rollover_flags >= 2:
        return "break_confirmed"
    if high_vulnerability and real_economy >= 0.60 and market_confirmation >= 0.50:
        return "rollover_confirming"
    if high_vulnerability and rollover_flags >= 2:
        return "prebreak_pressure"
    if high_vulnerability:
        return "buildout_fragile"
    if pd.notna(rollover) and rollover >= 0.55:
        return "ordinary_correction_or_early_rollover"
    return "normal"


def _capital_cycle_state(
    vulnerability: float,
    rollover: float,
    vulnerability_coverage: float,
    rollover_coverage: float,
    real_economy_confirmations: int,
    independent_rollover_confirmations: int,
) -> str:
    """Apply a sequential state policy without letting fragility imply a break."""

    if vulnerability_coverage < 0.45 or rollover_coverage < 0.45:
        return "insufficient_data"
    high_vulnerability = pd.notna(vulnerability) and vulnerability >= 0.60
    high_rollover = pd.notna(rollover) and rollover >= 0.60
    if (
        high_vulnerability
        and rollover >= 0.75
        and real_economy_confirmations >= 1
        and independent_rollover_confirmations >= 2
    ):
        return "break_confirmed"
    if (
        high_vulnerability
        and high_rollover
        and real_economy_confirmations >= 1
        and independent_rollover_confirmations >= 1
    ):
        return "rollover_confirming"
    if high_vulnerability and high_rollover and independent_rollover_confirmations >= 2:
        return "prebreak_pressure"
    if high_vulnerability:
        return "buildout_fragile"
    if high_rollover:
        return "ordinary_correction_or_early_rollover"
    return "normal"


def _transition_pressure(state: str) -> str:
    return {
        "normal": "none",
        "buildout_fragile": "pre_break watch",
        "prebreak_pressure": "pre_break rising",
        "rollover_confirming": "pre_break to early_unwind moderate/high",
        "break_confirmed": "early_unwind evidence strong; broad confirmation still governs liquidation",
        "ordinary_correction_or_early_rollover": "ambiguous without high structural vulnerability",
        "insufficient_data": "unavailable",
    }.get(state, "unavailable")


def _composite_direction(families: pd.DataFrame) -> str:
    eligible = families
    if "included_in_upstream_composite" in eligible:
        eligible = eligible[eligible["included_in_upstream_composite"].fillna(False)]
    changes = pd.to_numeric(eligible.get("change_1m"), errors="coerce").dropna()
    acceleration = pd.to_numeric(eligible.get("acceleration"), errors="coerce").dropna()
    if changes.empty:
        return "unavailable"
    change = float(changes.mean())
    accel = float(acceleration.mean()) if not acceleration.empty else 0.0
    return _derivative_state(change, accel)


def _confirmation_frame(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    fundamentals: pd.DataFrame,
    families: pd.DataFrame,
) -> pd.DataFrame:
    fundamental_columns = set(fundamentals.columns)
    family_status = families.set_index("family")["data_status"].to_dict()
    rows = [
        (
            "capex_revisions",
            "real_economy",
            "capex_guidance" in fundamental_columns,
            "Hyperscaler capex consensus or guidance revisions",
        ),
        (
            "supplier_orders_backlog",
            "real_economy",
            bool({"supplier_backlog", "infrastructure_orders"}.intersection(fundamental_columns)),
            "Supplier orders, backlog, and book-to-bill",
        ),
        (
            "compute_price_utilization",
            "real_economy",
            bool({"compute_price", "capacity_utilization"}.intersection(fundamental_columns)),
            "GPU/cloud pricing and capacity utilization",
        ),
        (
            "frontier_lab_economics",
            "real_economy",
            bool({"operating_loss", "compute_commitments"}.intersection(fundamental_columns)),
            "Frontier-lab revenue, losses, and compute commitments",
        ),
        (
            "ai_specific_credit",
            "market",
            False,
            "AI-project, neocloud, or technology-specific credit spreads",
        ),
        (
            "margin_debt",
            "positioning",
            family_status.get("margin_leverage") == "direct",
            "FINRA margin debt and rollover",
        ),
        (
            "theme_market_break",
            "market",
            family_status.get("theme_market_break") != "missing",
            "AI leadership relative trend and drawdown",
        ),
        (
            "broad_credit",
            "market",
            any(column in macro for column in ("BAMLC0A0CM", "BAMLH0A0HYM2")),
            "Broad IG/HY spread confirmation",
        ),
        (
            "broad_trend",
            "market",
            {"SPY", "QQQ"}.issubset(prices.columns),
            "Broad equity trend confirmation remains in the native risk engine",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "confirmation": name,
                "confirmation_type": kind,
                "available": bool(available),
                "status": "observable" if available else "missing",
                "description": description,
                "allocation_authority": 0.0,
            }
            for name, kind, available, description in rows
        ]
    )


def _research_readiness_frame(
    families: pd.DataFrame,
    confirmations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in families.iterrows():
        status = str(row.get("data_status", "missing"))
        rows.append(
            {
                "research_item": str(row.get("family")),
                "item_type": "signal_family",
                "availability": status,
                "promotion_eligible": False,
                "next_requirement": (
                    "Build historical analogue series and nested walk-forward incremental-value tests."
                    if status != "missing"
                    else str(row.get("missing_inputs", "Acquire direct source data."))
                ),
            }
        )
    for _, row in confirmations[confirmations["available"].eq(False)].iterrows():  # noqa: E712
        rows.append(
            {
                "research_item": str(row.get("confirmation")),
                "item_type": "missing_confirmation",
                "availability": "missing",
                "promotion_eligible": False,
                "next_requirement": str(row.get("description")),
            }
        )
    return pd.DataFrame(rows)


def _implication(
    state: str,
    vulnerability: float,
    rollover: float,
    confirmations: pd.DataFrame,
) -> str:
    missing_real = confirmations[
        confirmations["confirmation_type"].eq("real_economy") & confirmations["available"].eq(False)  # noqa: E712
    ]["confirmation"].tolist()
    vulnerability_read = _score_label(vulnerability)
    rollover_read = _score_label(rollover)
    missing_text = ", ".join(missing_real[:3]) if missing_real else "none"
    return (
        f"Capital vulnerability is {vulnerability_read}; rollover impulse is {rollover_read}. "
        f"State is {state.replace('_', ' ')}. Missing real-economy confirmation: {missing_text}. "
        "This is research evidence only and cannot change sizing."
    )


def _event_family_score(events: pd.DataFrame, family: str) -> float:
    if events.empty:
        return 0.0
    values = pd.to_numeric(
        events.loc[events["cycle_family"].eq(family), "decayed_contribution"],
        errors="coerce",
    ).dropna()
    return _clip01(float(values.sum())) if not values.empty else 0.0


def _clean_frame(frame: pd.DataFrame | None, *, as_of: object | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    clean.index = pd.to_datetime(clean.index, errors="coerce").tz_localize(None)
    clean = clean[clean.index.notna()].sort_index().dropna(how="all")
    cutoff = pd.to_datetime(as_of, errors="coerce")
    if pd.notna(cutoff):
        clean = clean.loc[: cutoff.tz_localize(None) if cutoff.tzinfo else cutoff]
    return clean


def _clean_fundamentals(frame: pd.DataFrame | None, *, as_of: object | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    period_column = next(
        (column for column in ("period_end", "fiscal_period_end", "date") if column in clean),
        None,
    )
    availability_column = next(
        (
            column
            for column in ("available_date", "filing_date", "published_at", "date")
            if column in clean
        ),
        None,
    )
    if period_column is None:
        return pd.DataFrame()
    clean[period_column] = pd.to_datetime(
        clean[period_column], errors="coerce"
    ).dt.tz_localize(None)
    clean = clean.dropna(subset=[period_column]).sort_values(period_column)
    if availability_column is not None:
        clean[availability_column] = pd.to_datetime(
            clean[availability_column], errors="coerce", utc=True
        ).dt.tz_convert(None)
    cutoff = pd.to_datetime(as_of, errors="coerce")
    if pd.notna(cutoff) and availability_column is not None:
        cutoff = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
        clean = clean[clean[availability_column] <= cutoff]
    elif pd.notna(cutoff):
        cutoff = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
        clean = clean[clean[period_column] <= cutoff]
    return clean.rename(columns={period_column: "period_end"})


def _resolve_as_of(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    fundamentals: pd.DataFrame,
    as_of: object | None,
) -> str:
    requested = pd.to_datetime(as_of, errors="coerce")
    if pd.notna(requested):
        return requested.date().isoformat()
    dates = []
    for frame in (prices, macro):
        if not frame.empty:
            dates.append(pd.to_datetime(frame.index.max()))
    if not fundamentals.empty:
        dates.append(pd.to_datetime(fundamentals["period_end"].max()))
    return max(dates).date().isoformat() if dates else ""


def _fundamental_totals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    numeric = fundamentals.copy()
    excluded = {
        "period_end",
        "available_date",
        "filing_date",
        "published_at",
        "entity",
        "ticker",
        "source",
        "source_quality",
    }
    value_columns = [column for column in numeric if column not in excluded]
    for column in value_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.groupby("period_end", sort=True)[value_columns].sum(min_count=1)


def _equal_weight_index(prices: pd.DataFrame, tickers: tuple[str, ...]) -> pd.Series:
    available = [ticker for ticker in tickers if ticker in prices]
    if not available:
        return pd.Series(dtype=float)
    returns = prices[available].pct_change(fill_method=None)
    return (1.0 + returns.mean(axis=1, skipna=True).fillna(0.0)).cumprod()


def _threshold_series(
    series: pd.Series,
    calm: float,
    stressed: float,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if stressed == calm:
        return pd.Series(np.nan, index=values.index)
    return ((values - calm) / (stressed - calm)).clip(0.0, 1.0)


def _mean_series(series: list[pd.Series]) -> pd.Series:
    populated = [item.rename(index=pd.to_datetime) for item in series if not item.empty]
    if not populated:
        return pd.Series(dtype=float)
    return pd.concat(populated, axis=1).mean(axis=1, skipna=True).dropna().sort_index()


def _series_value_on_or_before(series: pd.Series, target: pd.Timestamp) -> float:
    eligible = series[pd.to_datetime(series.index) <= target]
    return float(eligible.iloc[-1]) if not eligible.empty else float("nan")


def _derivative_state(change: float, acceleration: float) -> str:
    if pd.isna(change):
        return "unavailable"
    adjusted = float(change) + 0.5 * (float(acceleration) if pd.notna(acceleration) else 0.0)
    if adjusted >= 0.12:
        return "worsening_fast"
    if adjusted >= 0.035:
        return "worsening"
    if adjusted <= -0.12:
        return "improving_fast"
    if adjusted <= -0.035:
        return "improving"
    return "flat"


def _confidence_value(value: object) -> float:
    numeric = _safe_float(value, np.nan)
    if pd.notna(numeric):
        return _clip01(numeric)
    return {
        "high": 0.90,
        "medium": 0.65,
        "low": 0.35,
    }.get(str(value).strip().lower(), 0.50)


def _source_quality(value: object) -> str:
    numeric = _safe_float(value, np.nan)
    if pd.isna(numeric):
        return "unclassified"
    if numeric <= 1:
        return "high"
    if numeric <= 2:
        return "medium"
    return "low"


def _event_entity(row: pd.Series) -> str:
    for column in ("entity", "company", "candidate_proxies"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return "unclassified"


def _score_label(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    if value >= 0.75:
        return "extreme"
    if value >= 0.60:
        return "high"
    if value >= 0.40:
        return "moderate"
    return "low"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _empty_family_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "family",
            "dimension",
            "role",
            "typical_lead",
            "score",
            "change_1m",
            "change_3m",
            "acceleration",
            "derivative_state",
            "data_status",
            "source_type",
            "interpretation",
            "missing_inputs",
        ]
    )
