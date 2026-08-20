from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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
    "months_supply_increase": (
        "housing_months_supply",
        "housing_new_home_months_supply",
    ),
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
    "semiconductor_output_drawdown": ("fed_g17_semiconductor_industrial_production",),
    "semiconductor_utilization_drawdown": (
        "fed_g17_semiconductor_capacity_utilization",
    ),
}


@dataclass(frozen=True)
class CapitalCycleTargetRun:
    component_panel: pd.DataFrame
    event_manifest: pd.DataFrame
    origin_targets: pd.DataFrame
    readiness: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    technology_threshold_sensitivity: pd.DataFrame
    oil_shock_sensitivity: pd.DataFrame
    oil_shock_holdout: pd.DataFrame
    output_paths: dict[str, Path]


def build_objective_economic_targets(
    *,
    episode_manifest: pd.DataFrame,
    observations: pd.DataFrame,
    target_spec: pd.DataFrame,
    oil_shock_target_spec: pd.DataFrame | None = None,
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
    threshold_sensitivity = build_housing_threshold_sensitivity(components, episodes)
    technology_threshold_sensitivity = build_technology_threshold_sensitivity(components)
    oil_shock_sensitivity = build_oil_shock_sensitivity(audited)
    oil_shock_holdout = build_oil_shock_holdout(audited, oil_shock_target_spec)
    frames = {
        "component_panel": components,
        "event_manifest": events,
        "origin_targets": targets,
        "readiness": readiness,
        "threshold_sensitivity": threshold_sensitivity,
        "technology_threshold_sensitivity": technology_threshold_sensitivity,
        "oil_shock_sensitivity": oil_shock_sensitivity,
        "oil_shock_holdout": oil_shock_holdout,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = destination / f"capital_cycle_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return CapitalCycleTargetRun(output_paths=paths, **frames)


def build_housing_threshold_sensitivity(
    component_panel: pd.DataFrame,
    episode_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Run a bounded, explicitly exploratory sensitivity audit on the housing pair."""

    break_episode = "housing_mortgage_2001_2011"
    control_episode = "housing_recovery_2012_2019"
    required_episodes = {break_episode, control_episode}
    if component_panel.empty or not required_episodes.issubset(
        set(component_panel.get("episode", pd.Series(dtype=str)).astype(str))
    ):
        return pd.DataFrame()
    thresholds = {
        "housing_starts_drawdown": (-0.15, -0.20, -0.25),
        "months_supply_increase": (0.10, 0.15, 0.20, 0.25, 0.30, 0.35),
        "mortgage_credit_tightening": (0.20,),
    }
    episode_dates = episode_manifest.set_index("episode")
    rows: list[dict[str, object]] = []
    for combination in product(*thresholds.values()):
        selected = dict(zip(thresholds, combination, strict=True))
        dates = {
            episode: _sensitivity_event_date(component_panel, episode, selected)
            for episode in required_episodes
        }
        break_date = dates[break_episode]
        recognition = pd.to_datetime(
            episode_dates.loc[break_episode].get("market_recognition_date"),
            errors="coerce",
        )
        first_deterioration = pd.to_datetime(
            episode_dates.loc[break_episode].get("first_deterioration_date"),
            errors="coerce",
        )
        rows.append(
            {
                **selected,
                "break_event_date": break_date,
                "negative_control_event_date": dates[control_episode],
                "negative_control_passed": pd.isna(dates[control_episode]),
                "lead_days_vs_market_recognition": (
                    int((recognition - break_date).days)
                    if pd.notna(recognition) and pd.notna(break_date)
                    else pd.NA
                ),
                "lag_days_vs_first_deterioration": (
                    int((break_date - first_deterioration).days)
                    if pd.notna(first_deterioration) and pd.notna(break_date)
                    else pd.NA
                ),
                "preregistered_configuration": (
                    selected["housing_starts_drawdown"] == -0.20
                    and selected["months_supply_increase"] == 0.35
                    and selected["mortgage_credit_tightening"] == 0.20
                ),
                "analysis_role": "post_hoc_exploratory_sensitivity_only",
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_technology_threshold_sensitivity(component_panel: pd.DataFrame) -> pd.DataFrame:
    """Audit bounded post-hoc technology thresholds against one same-family control."""

    break_episodes = (
        "dotcom_technology_hardware_1997_2004",
        "semiconductor_inventory_correction_2017_2020",
        "semiconductor_shortage_unwind_2021_2024",
    )
    control_episode = "semiconductor_expansion_2012_2016"
    required_episodes = {*break_episodes, control_episode}
    if component_panel.empty or not required_episodes.issubset(
        set(component_panel.get("episode", pd.Series(dtype=str)).astype(str))
    ):
        return pd.DataFrame()
    thresholds = {
        "semiconductor_output_drawdown": (-0.05, -0.075, -0.10, -0.125),
        "semiconductor_utilization_drawdown": (-0.04, -0.06, -0.08, -0.10),
    }
    rows: list[dict[str, object]] = []
    for combination in product(*thresholds.values()):
        selected = dict(zip(thresholds, combination, strict=True))
        dates = {
            episode: _sensitivity_event_date(component_panel, episode, selected)
            for episode in required_episodes
        }
        detected = sum(pd.notna(dates[episode]) for episode in break_episodes)
        rows.append(
            {
                **selected,
                "dotcom_break_event_date": dates[break_episodes[0]],
                "inventory_correction_event_date": dates[break_episodes[1]],
                "shortage_unwind_event_date": dates[break_episodes[2]],
                "negative_control_event_date": dates[control_episode],
                "breaks_detected": detected,
                "break_detection_rate": detected / len(break_episodes),
                "negative_control_passed": pd.isna(dates[control_episode]),
                "preregistered_configuration": (
                    selected["semiconductor_output_drawdown"] == -0.10
                    and selected["semiconductor_utilization_drawdown"] == -0.08
                ),
                "analysis_role": "post_hoc_exploratory_sensitivity_only",
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_oil_shock_sensitivity(observations: pd.DataFrame) -> pd.DataFrame:
    """Explore producer-downside and consumer-upside oil shock definitions."""

    break_episodes = {
        "producer_downside": "oil_oversupply_collapse_2014_2016",
        "consumer_upside": "oil_geopolitical_shock_2021_2023",
    }
    control_episode = "oil_balanced_market_control_2017_2019"
    required = {*break_episodes.values(), control_episode}
    available = set(observations.get("episode", pd.Series(dtype=str)).astype(str))
    if observations.empty or not required.issubset(available):
        return pd.DataFrame()
    grids = {
        "producer_downside": {
            "price": (-0.20, -0.30, -0.40),
            "inventory": (0.03, 0.05, 0.08),
            "surplus_capacity": (0.10, 0.20, 0.30),
        },
        "consumer_upside": {
            "price": (0.20, 0.30, 0.40),
            "inventory": (-0.03, -0.05, -0.08),
            "surplus_capacity": (-0.20, -0.30, -0.40),
        },
    }
    rows: list[dict[str, object]] = []
    for mechanism, thresholds in grids.items():
        for combination in product(*thresholds.values()):
            selected = dict(zip(thresholds, combination, strict=True))
            for persistence in (1, 2):
                break_date = _oil_shock_event_date(
                    observations,
                    break_episodes[mechanism],
                    mechanism,
                    selected,
                    persistence,
                    quarterly_only=True,
                )
                control_date = _oil_shock_event_date(
                    observations,
                    control_episode,
                    mechanism,
                    selected,
                    persistence,
                    quarterly_only=True,
                )
                rows.append(
                    {
                        "mechanism": mechanism,
                        "price_threshold": selected["price"],
                        "inventory_threshold": selected["inventory"],
                        "surplus_capacity_threshold": selected["surplus_capacity"],
                        "persistence_releases": persistence,
                        "sampling_cadence": "quarterly_archive_releases",
                        "break_event_date": break_date,
                        "negative_control_event_date": control_date,
                        "break_detected": pd.notna(break_date),
                        "negative_control_passed": pd.isna(control_date),
                        "preregistered_configuration": False,
                        "analysis_role": "post_hoc_exploratory_sensitivity_only",
                        "allocation_authority": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def build_oil_shock_holdout(
    observations: pd.DataFrame,
    target_spec: pd.DataFrame | None,
) -> pd.DataFrame:
    """Score frozen oil rules on their development, control, and untouched holdout."""

    if observations.empty or target_spec is None or target_spec.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for spec in target_spec.to_dict("records"):
        thresholds = {
            "price": float(spec["price_threshold"]),
            "inventory": float(spec["inventory_threshold"]),
            "surplus_capacity": float(spec["surplus_capacity_threshold"]),
        }
        mechanism = str(spec["mechanism"])
        persistence = int(spec["persistence_releases"])
        lookback = int(spec["lookback_releases"])
        minimum_components = int(spec["minimum_components"])
        quarterly_only = str(spec.get("sampling_cadence", "")).casefold().startswith(
            "quarterly"
        )
        episode_dates = {
            role: _oil_shock_event_date(
                observations,
                str(spec[column]),
                mechanism,
                thresholds,
                persistence,
                lookback=lookback,
                minimum_components=minimum_components,
                quarterly_only=quarterly_only,
            )
            for role, column in {
                "holdout": "episode",
                "development": "development_episode",
                "negative_control": "negative_control_episode",
            }.items()
        }
        rows.append(
            {
                **spec,
                "holdout_event_date": episode_dates["holdout"],
                "development_event_date": episode_dates["development"],
                "negative_control_event_date": episode_dates["negative_control"],
                "holdout_detected": pd.notna(episode_dates["holdout"]),
                "development_detected": pd.notna(episode_dates["development"]),
                "negative_control_passed": pd.isna(episode_dates["negative_control"]),
                "analysis_role": "frozen_historical_holdout_not_prospective",
                "preregistered_configuration": False,
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _oil_shock_event_date(
    observations: pd.DataFrame,
    episode: str,
    mechanism: str,
    thresholds: dict[str, float],
    persistence: int,
    *,
    lookback: int = 3,
    minimum_components: int = 2,
    quarterly_only: bool = False,
) -> pd.Timestamp | pd.NaT:
    metrics = {
        "price": "eia_steo_wti_spot_price",
        "inventory": "eia_steo_oecd_commercial_inventory",
        "surplus_capacity": "eia_steo_opec_surplus_capacity",
    }
    flags: dict[str, pd.Series] = {}
    for component, metric in metrics.items():
        source = observations[
            observations["episode"].astype(str).eq(episode)
            & observations["metric"].astype(str).eq(metric)
            & observations["eligible_for_model"].eq(True)
        ].copy()
        source["available_date"] = pd.to_datetime(source["available_date"], errors="coerce")
        if quarterly_only and "source_id" in source.columns:
            source = source[source["available_date"].dt.month.isin({1, 4, 7, 10})]
        source = source.sort_values("available_date").drop_duplicates("available_date", keep="last")
        values = pd.to_numeric(source["value"], errors="coerce")
        raw: list[bool] = []
        for index, value in enumerate(values):
            if index < lookback or pd.isna(value):
                raw.append(False)
                continue
            reference = values.iloc[index - lookback : index].dropna()
            if len(reference) < lookback:
                raw.append(False)
                continue
            lower = (mechanism == "producer_downside" and component == "price") or (
                mechanism == "consumer_upside" and component in {"inventory", "surplus_capacity"}
            )
            denominator = float(reference.max() if lower else reference.median())
            transformed = float(value) / denominator - 1 if denominator else np.nan
            raw.append(
                bool(transformed <= thresholds[component])
                if lower
                else bool(transformed >= thresholds[component])
            )
        persistent = [
            index + 1 >= persistence and all(raw[index - persistence + 1 : index + 1])
            for index in range(len(raw))
        ]
        flags[component] = pd.Series(
            persistent,
            index=pd.DatetimeIndex(source["available_date"]),
        )
    if len(flags) < 3:
        return pd.NaT
    confirmations = pd.DataFrame(flags).fillna(False).sum(axis=1)
    triggered = confirmations[confirmations.ge(minimum_components)]
    return pd.Timestamp(triggered.index.min()) if not triggered.empty else pd.NaT


def _sensitivity_event_date(
    component_panel: pd.DataFrame,
    episode: str,
    thresholds: dict[str, float],
) -> pd.Timestamp | pd.NaT:
    flags: dict[str, pd.Series] = {}
    for component, threshold in thresholds.items():
        source = component_panel[
            component_panel["episode"].astype(str).eq(episode)
            & component_panel["target_component"].astype(str).eq(component)
        ].sort_values("origin_date")
        if source.empty:
            continue
        transformed = pd.to_numeric(source["transformed_value"], errors="coerce")
        raw = (
            transformed.le(threshold)
            if str(source.iloc[0]["direction"]) == "lower_is_rollover"
            else transformed.ge(threshold)
        ).fillna(False)
        history: list[bool] = []
        persistent: list[bool] = []
        persistence = max(1, int(source.iloc[0]["persistence_periods"]))
        for updated, flag in zip(source["evidence_updated_at_origin"], raw, strict=True):
            if bool(updated):
                history.append(bool(flag))
            persistent.append(len(history) >= persistence and all(history[-persistence:]))
        flags[component] = pd.Series(
            persistent,
            index=pd.to_datetime(source["origin_date"]),
        )
    if len(flags) < 2:
        return pd.NaT
    confirmations = pd.DataFrame(flags).sum(axis=1)
    triggered = confirmations[confirmations.ge(2)]
    return pd.Timestamp(triggered.index.min()) if not triggered.empty else pd.NaT


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
