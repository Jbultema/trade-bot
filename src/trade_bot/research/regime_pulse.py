from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimePulseDefinition:
    cycle: str
    horizon: str
    categories: tuple[str, ...]
    stock_weight: float
    bond_weight: float
    dollar_weight: float
    gold_weight: float
    bitcoin_weight: float
    commodity_weight: float


def build_regime_pulse_cycles(
    macro_signals: pd.DataFrame,
    positioning_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [
        _cycle_row(definition, macro_signals)
        for definition in _cycle_definitions()
    ]
    cycle_table = pd.DataFrame(rows)
    if positioning_summary is not None and not positioning_summary.empty:
        cycle_table = _blend_positioning_cycle(cycle_table, positioning_summary)
    return cycle_table.sort_values("cycle")


def build_regime_pulse_asset_table(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return pd.DataFrame()

    assets = {
        "stocks": "stock_weight",
        "bonds": "bond_weight",
        "us_dollar": "dollar_weight",
        "gold": "gold_weight",
        "bitcoin": "bitcoin_weight",
        "commodities": "commodity_weight",
    }
    rows = []
    for asset, weight_column in assets.items():
        weighted = cycles["cycle_tailwind_score"] * cycles[weight_column]
        usable = weighted.dropna()
        score = float(usable.mean()) if not usable.empty else np.nan
        top_tailwind = _top_cycle(cycles, weight_column, positive=True)
        top_headwind = _top_cycle(cycles, weight_column, positive=False)
        rows.append(
            {
                "asset_class": asset,
                "regime_pulse_score": score,
                "regime_pulse_read": _asset_read(score),
                "top_tailwind": top_tailwind,
                "top_headwind": top_headwind,
            }
        )
    return pd.DataFrame(rows).sort_values("regime_pulse_score", ascending=False)


def build_growth_inflation_map(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return pd.DataFrame()

    growth = _cycle_score(cycles, "growth")
    disinflation = _cycle_score(cycles, "inflation")
    inflation = -disinflation
    raw_scores = {
        "Growth-disinflation": growth + disinflation,
        "Reflation": growth + inflation,
        "Inflation": -growth + inflation,
        "Deflation": -growth + disinflation,
    }
    probabilities = _softmax(raw_scores)
    rows = []
    for regime, probability in probabilities.items():
        rows.append(
            {
                "regime": regime,
                "probability": probability,
                "growth_impulse": growth,
                "inflation_impulse": inflation,
                "regime_read": _regime_read(regime),
            }
        )
    return pd.DataFrame(rows).sort_values("probability", ascending=False)


def _cycle_definitions() -> tuple[RegimePulseDefinition, ...]:
    return (
        RegimePulseDefinition(
            cycle="growth",
            horizon="1-3m",
            categories=("growth", "labor", "consumer", "sentiment", "housing"),
            stock_weight=1.00,
            bond_weight=-0.35,
            dollar_weight=0.15,
            gold_weight=-0.10,
            bitcoin_weight=0.80,
            commodity_weight=0.45,
        ),
        RegimePulseDefinition(
            cycle="inflation",
            horizon="1-3m",
            categories=(
                "inflation_market",
                "inflation_realized",
                "inflation_expectations",
                "commodities",
                "wages",
            ),
            stock_weight=0.80,
            bond_weight=1.00,
            dollar_weight=-0.25,
            gold_weight=0.35,
            bitcoin_weight=0.35,
            commodity_weight=-0.80,
        ),
        RegimePulseDefinition(
            cycle="monetary_policy",
            horizon="1-3m",
            categories=("policy_rates", "rates_curve", "real_rates", "housing_rates"),
            stock_weight=0.90,
            bond_weight=0.80,
            dollar_weight=-0.40,
            gold_weight=0.20,
            bitcoin_weight=0.85,
            commodity_weight=0.20,
        ),
        RegimePulseDefinition(
            cycle="fiscal_policy",
            horizon="1-3m",
            categories=("fiscal", "consumer_credit"),
            stock_weight=0.45,
            bond_weight=-0.25,
            dollar_weight=0.10,
            gold_weight=0.10,
            bitcoin_weight=0.35,
            commodity_weight=0.35,
        ),
        RegimePulseDefinition(
            cycle="liquidity",
            horizon="1-3m",
            categories=("liquidity", "financial_conditions", "credit_spreads", "dollar_fx"),
            stock_weight=1.00,
            bond_weight=0.35,
            dollar_weight=-0.60,
            gold_weight=0.35,
            bitcoin_weight=1.00,
            commodity_weight=0.30,
        ),
        RegimePulseDefinition(
            cycle="positioning",
            horizon="<1m",
            categories=("market_index", "volatility"),
            stock_weight=0.75,
            bond_weight=0.20,
            dollar_weight=0.10,
            gold_weight=0.15,
            bitcoin_weight=0.75,
            commodity_weight=0.35,
        ),
    )


def _cycle_row(definition: RegimePulseDefinition, macro_signals: pd.DataFrame) -> dict[str, object]:
    if macro_signals.empty:
        pressure_score = np.nan
        usable = 0
        pressure_groups = ""
    else:
        rows = macro_signals[macro_signals["category"].isin(definition.categories)]
        usable_rows = rows[rows["latest_value"].notna()] if "latest_value" in rows else rows
        usable = int(usable_rows.shape[0])
        pressure_score = (
            float(usable_rows["risk_score"].mean())
            if usable and "risk_score" in usable_rows
            else np.nan
        )
        pressure_groups = _join_unique(usable_rows.get("category", pd.Series(dtype=object)))

    tailwind_score = -pressure_score if pd.notna(pressure_score) else np.nan
    row = {
        "cycle": definition.cycle,
        "horizon": definition.horizon,
        "series_count": usable,
        "source_categories": pressure_groups,
        "risk_pressure_score": pressure_score,
        "cycle_tailwind_score": tailwind_score,
        "cycle_state": _cycle_state(tailwind_score),
        "stock_weight": definition.stock_weight,
        "bond_weight": definition.bond_weight,
        "dollar_weight": definition.dollar_weight,
        "gold_weight": definition.gold_weight,
        "bitcoin_weight": definition.bitcoin_weight,
        "commodity_weight": definition.commodity_weight,
    }
    return row


def _blend_positioning_cycle(cycles: pd.DataFrame, positioning_summary: pd.DataFrame) -> pd.DataFrame:
    output = cycles.copy()
    positioning_rows = positioning_summary[
        positioning_summary["asset_group"].isin(
            ["broad_us_equity", "ai_beta", "us_equity_sectors", "defensive_equity_factor"]
        )
    ]
    if positioning_rows.empty:
        return output

    crowding_pressure = float(positioning_rows["mean_crowding_score"].mean())
    index = output.index[output["cycle"] == "positioning"]
    if index.empty:
        return output

    existing_tailwind = output.loc[index, "cycle_tailwind_score"].iloc[0]
    price_tailwind = -crowding_pressure
    if pd.isna(existing_tailwind):
        blended = price_tailwind
    else:
        blended = 0.50 * float(existing_tailwind) + 0.50 * price_tailwind
    output.loc[index, "cycle_tailwind_score"] = blended
    output.loc[index, "risk_pressure_score"] = -blended
    output.loc[index, "cycle_state"] = _cycle_state(blended)
    output.loc[index, "series_count"] = int(
        output.loc[index, "series_count"].iloc[0] + positioning_rows["tickers"].sum()
    )
    return output


def _cycle_state(score: float) -> str:
    if pd.isna(score):
        return "missing"
    if score >= 0.35:
        return "meaningful_tailwind"
    if score >= 0.12:
        return "modest_tailwind"
    if score <= -0.35:
        return "meaningful_headwind"
    if score <= -0.12:
        return "modest_headwind"
    return "neutral"


def _asset_read(score: float) -> str:
    if pd.isna(score):
        return "missing"
    if score >= 0.25:
        return "macro supports buying or holding"
    if score <= -0.25:
        return "macro argues for reduced exposure"
    return "mixed or neutral"


def _regime_read(regime: str) -> str:
    return {
        "Growth-disinflation": "growth improving while inflation pressure fades",
        "Reflation": "growth and inflation pressure both rising",
        "Inflation": "growth weakening while inflation pressure rises",
        "Deflation": "growth and inflation pressure both weakening",
    }[regime]


def _cycle_score(cycles: pd.DataFrame, cycle: str) -> float:
    rows = cycles[cycles["cycle"] == cycle]
    if rows.empty:
        return 0.0
    value = rows["cycle_tailwind_score"].iloc[0]
    return 0.0 if pd.isna(value) else float(value)


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    values = np.array(list(scores.values()), dtype=float)
    values = values - values.max()
    exp_values = np.exp(values)
    probabilities = exp_values / exp_values.sum()
    return {
        key: float(probability)
        for key, probability in zip(scores.keys(), probabilities, strict=True)
    }


def _top_cycle(cycles: pd.DataFrame, weight_column: str, *, positive: bool) -> str:
    frame = cycles.copy()
    frame["asset_contribution"] = frame["cycle_tailwind_score"] * frame[weight_column]
    frame = frame.dropna(subset=["asset_contribution"])
    if frame.empty:
        return ""
    ordered = frame.sort_values("asset_contribution", ascending=not positive)
    row = ordered.iloc[0]
    return f"{row['cycle']} ({float(row['asset_contribution']):.2f})"


def _join_unique(values: pd.Series) -> str:
    unique_values = [str(value) for value in values.dropna().unique() if str(value)]
    return ", ".join(unique_values)
