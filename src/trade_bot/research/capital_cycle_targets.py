from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from trade_bot.research.capital_cycle_feasibility import (
    audit_point_in_time_observations,
    validate_episode_manifest,
)

DEFAULT_CAPITAL_CYCLE_TARGET_DIR = Path("reports/capital_cycle_targets")
TARGET_COMPONENT_METRIC_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "housing_starts_drawdown": ("housing_starts_saar_thousands",),
    "months_supply_increase": ("housing_months_supply",),
    "mortgage_credit_tightening": ("sloos_mortgage_credit_tightening",),
    "production_or_shipments_decline": (
        "mining_production_index",
        "mining_shipments_index",
        "usgs_copper_reported_refined_consumption",
    ),
    "capex_contraction": ("issuer_aggregate_capex",),
    "cashflow_below_commitment": ("cashflow_commitment_burden_ratio",),
    "rig_count_drawdown": ("shale_rig_count",),
    "completion_or_production_deceleration": (
        "shale_completion_index",
        "shale_production_index",
        "eia_total_shale_oil_production_forecast",
    ),
    "cashflow_below_capex_and_commitments": ("cashflow_capex_commitment_burden_ratio",),
    "cloud_capex_contraction": ("cloud_issuer_aggregate_capex",),
    "cloud_monetization_deterioration": ("cloud_monetization_efficiency",),
    "cloud_commitment_overhang_deterioration": ("cloud_commitment_burden_ratio",),
}


@dataclass(frozen=True)
class CapitalCycleTargetRun:
    component_panel: pd.DataFrame
    event_manifest: pd.DataFrame
    origin_targets: pd.DataFrame
    readiness: pd.DataFrame
    output_paths: dict[str, Path]


def build_objective_economic_targets(
    *,
    episode_manifest: pd.DataFrame,
    observations: pd.DataFrame,
    target_spec: pd.DataFrame,
    output_dir: str | Path = DEFAULT_CAPITAL_CYCLE_TARGET_DIR,
) -> CapitalCycleTargetRun:
    """Generate economic rollover labels only from preregistered PIT rules."""

    episodes = validate_episode_manifest(episode_manifest)
    audited = audit_point_in_time_observations(observations, episodes)
    spec = _validate_target_spec(target_spec, episodes)
    components = _build_component_panel(episodes, audited, spec)
    events = _build_event_manifest(episodes, components, spec)
    targets = _build_origin_targets(episodes, events)
    readiness = _build_target_readiness(episodes, components, events, spec)
    frames = {
        "component_panel": components,
        "event_manifest": events,
        "origin_targets": targets,
        "readiness": readiness,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = destination / f"capital_cycle_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return CapitalCycleTargetRun(output_paths=paths, **frames)


def _validate_target_spec(spec: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "episode",
        "target_component",
        "family",
        "direction",
        "lookback_periods",
        "threshold",
        "persistence_periods",
        "minimum_independent_components",
        "preregistered_at",
    }
    missing = sorted(required.difference(spec.columns))
    if missing:
        raise ValueError(f"Economic target specification is missing columns: {', '.join(missing)}")
    clean = spec.copy()
    clean["preregistered_at"] = pd.to_datetime(clean["preregistered_at"], errors="coerce")
    if clean["preregistered_at"].isna().any():
        raise ValueError("Every economic target component requires a preregistration date.")
    if not set(clean["episode"].astype(str)).issubset(set(episodes["episode"].astype(str))):
        raise ValueError("Economic target specification includes an unregistered episode.")
    if not clean["direction"].isin({"lower_is_rollover", "higher_is_rollover"}).all():
        raise ValueError("Target direction must be lower_is_rollover or higher_is_rollover.")
    return clean


def _build_component_panel(
    episodes: pd.DataFrame,
    observations: pd.DataFrame,
    spec: pd.DataFrame,
) -> pd.DataFrame:
    eligible = observations[observations["eligible_for_model"].eq(True)].copy()
    rows: list[dict[str, object]] = []
    for _, episode in episodes.iterrows():
        episode_name = str(episode["episode"])
        origins = pd.date_range(episode["start_date"], episode["end_date"], freq="MS")
        for _, rule in spec[spec["episode"].astype(str).eq(episode_name)].iterrows():
            component = str(rule["target_component"])
            aliases = TARGET_COMPONENT_METRIC_ALIASES.get(component, (component,))
            source = eligible[
                eligible["episode"].astype(str).eq(episode_name)
                & eligible["metric"].astype(str).isin(aliases)
            ].copy()
            component_rows = _component_history_for_rule(origins, source, rule)
            rows.extend(component_rows)
    return pd.DataFrame(rows)


def _component_history_for_rule(
    origins: pd.DatetimeIndex,
    source: pd.DataFrame,
    rule: pd.Series,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    history_flags: list[bool] = []
    last_evidence_date = pd.NaT
    lookback = max(2, int(rule["lookback_periods"]))
    persistence = max(1, int(rule["persistence_periods"]))
    for origin in origins:
        known = source[source["available_date"].le(origin)].sort_values(
            ["period_end", "available_date"]
        )
        known = known.drop_duplicates("period_end", keep="last")
        values = pd.to_numeric(known["value"], errors="coerce").dropna()
        latest_value = values.iloc[-1] if not values.empty else np.nan
        latest_evidence_date = (
            pd.to_datetime(known["available_date"], errors="coerce").max()
            if not known.empty
            else pd.NaT
        )
        transform = np.nan
        if len(values) >= lookback + 1:
            reference = values.iloc[-(lookback + 1) : -1]
            if str(rule["direction"]) == "lower_is_rollover":
                denominator = float(reference.max())
                transform = latest_value / denominator - 1 if denominator != 0 else np.nan
                raw_flag = bool(transform <= float(rule["threshold"]))
            else:
                denominator = float(reference.median())
                transform = latest_value / denominator - 1 if denominator != 0 else np.nan
                raw_flag = bool(transform >= float(rule["threshold"]))
        else:
            raw_flag = False
        evidence_updated = pd.notna(latest_evidence_date) and (
            pd.isna(last_evidence_date) or latest_evidence_date != last_evidence_date
        )
        if evidence_updated:
            history_flags.append(raw_flag)
            last_evidence_date = latest_evidence_date
        persistent = len(history_flags) >= persistence and all(history_flags[-persistence:])
        rows.append(
            {
                "episode": rule["episode"],
                "origin_date": origin,
                "target_component": rule["target_component"],
                "family": rule["family"],
                "direction": rule["direction"],
                "threshold": rule["threshold"],
                "lookback_periods": lookback,
                "persistence_periods": persistence,
                "source_rows_known": len(known),
                "latest_value": latest_value,
                "latest_evidence_date": latest_evidence_date,
                "evidence_updated_at_origin": evidence_updated,
                "transformed_value": transform,
                "component_data_available": len(values) >= lookback + 1,
                "raw_component_flag": raw_flag,
                "persistent_component_flag": persistent,
                "allocation_authority": 0.0,
            }
        )
    return rows


def _build_event_manifest(
    episodes: pd.DataFrame,
    components: pd.DataFrame,
    spec: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, episode in episodes.iterrows():
        name = str(episode["episode"])
        episode_components = components[components["episode"].astype(str).eq(name)].copy()
        episode_spec = spec[spec["episode"].astype(str).eq(name)]
        target_spec_registered = not episode_spec.empty
        required = (
            int(episode_spec["minimum_independent_components"].max())
            if target_spec_registered
            else 0
        )
        available_components = (
            int(
                episode_components.groupby("target_component")["component_data_available"]
                .max()
                .sum()
            )
            if not episode_components.empty
            else 0
        )
        by_origin = (
            episode_components.groupby("origin_date")["persistent_component_flag"].sum()
            if not episode_components.empty
            else pd.Series(dtype=float)
        )
        triggered = by_origin[by_origin.ge(required)]
        event_date = pd.Timestamp(triggered.index[0]) if not triggered.empty else pd.NaT
        label_resolved = target_spec_registered and available_components >= required
        rows.append(
            {
                "episode": name,
                "episode_type": episode["episode_type"],
                "objective_rollover_date": event_date,
                "event_date": event_date,
                "target": "economic_rollover",
                "required_independent_components": required,
                "components_with_sufficient_history": available_components,
                "target_spec_registered": target_spec_registered,
                "objective_label_resolved": label_resolved,
                "negative_control_passed": (
                    bool(pd.isna(event_date))
                    if label_resolved and episode["episode_type"] == "negative_control"
                    else pd.NA
                ),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_origin_targets(episodes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_index = events.set_index("episode")
    for _, episode in episodes.iterrows():
        name = str(episode["episode"])
        event = event_index.loc[name]
        for origin in pd.date_range(episode["start_date"], episode["end_date"], freq="MS"):
            event_date = event["objective_rollover_date"]
            resolved = bool(event["objective_label_resolved"])
            rows.append(
                {
                    "episode": name,
                    "origin_date": origin,
                    "economic_rollover_2q": (
                        float(origin < event_date <= origin + pd.Timedelta(days=126))
                        if resolved and pd.notna(event_date)
                        else 0.0
                        if resolved
                        else np.nan
                    ),
                    "economic_rollover_4q": (
                        float(origin < event_date <= origin + pd.Timedelta(days=252))
                        if resolved and pd.notna(event_date)
                        else 0.0
                        if resolved
                        else np.nan
                    ),
                    "target_source": "preregistered_point_in_time_component_rule",
                    "allocation_authority": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _build_target_readiness(
    episodes: pd.DataFrame,
    components: pd.DataFrame,
    events: pd.DataFrame,
    spec: pd.DataFrame,
) -> pd.DataFrame:
    resolved = int(events["objective_label_resolved"].sum())
    unresolved = events.loc[~events["objective_label_resolved"], "episode"].astype(str).tolist()
    blockers = []
    if resolved < len(episodes):
        blockers.append("objective_rollover_labels_unresolved")
    controls = events[events["episode_type"].eq("negative_control")]
    if controls["negative_control_passed"].isna().any():
        blockers.append("negative_controls_unresolved")
    return pd.DataFrame(
        [
            {
                "episodes": len(episodes),
                "preregistered_components": len(spec),
                "component_panel_rows": len(components),
                "resolved_episode_labels": resolved,
                "unresolved_episodes": ";".join(unresolved),
                "economic_target_gate_passed": not blockers,
                "economic_forecasting_unlocked": not blockers,
                "equity_timing_unlocked": False,
                "strategy_utility_unlocked": False,
                "blockers": ";".join(blockers),
                "allocation_authority": 0.0,
            }
        ]
    )
