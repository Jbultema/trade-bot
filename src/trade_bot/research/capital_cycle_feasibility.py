from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

DEFAULT_CAPITAL_CYCLE_RESEARCH_DIR = Path("research/capital_cycle")
DEFAULT_CAPITAL_CYCLE_FEASIBILITY_DIR = Path("reports/capital_cycle_feasibility")

REQUIRED_DIRECT_FAMILIES: Final[tuple[str, ...]] = (
    "demand",
    "capacity",
    "pricing_utilization",
    "financing",
    "monetization",
    "commitment_overhang",
)
ELIGIBLE_VINTAGE_STATUSES: Final[frozenset[str]] = frozenset(
    {"archived_release", "filing_as_reported", "immutable_transaction_record"}
)
REQUIRED_EPISODE_COLUMNS: Final[tuple[str, ...]] = (
    "episode",
    "episode_family",
    "episode_type",
    "start_date",
    "end_date",
    "theme_ticker",
    "preregistered_at",
)
REQUIRED_OBSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "episode",
    "family",
    "metric",
    "entity",
    "period_end",
    "available_date",
    "value",
    "source_id",
    "source_url",
    "source_tier",
    "vintage_status",
)


@dataclass(frozen=True)
class CapitalCycleFeasibilityRun:
    episode_manifest: pd.DataFrame
    observation_audit: pd.DataFrame
    origin_coverage: pd.DataFrame
    episode_coverage: pd.DataFrame
    family_coverage: pd.DataFrame
    source_gaps: pd.DataFrame
    readiness: pd.DataFrame
    output_paths: dict[str, Path]


def run_capital_cycle_feasibility(
    *,
    episode_manifest: pd.DataFrame,
    observations: pd.DataFrame,
    source_registry: pd.DataFrame | None = None,
    origin_frequency: str = "MS",
    maximum_age_days: int = 550,
    minimum_origin_coverage: float = 0.60,
    minimum_episode_coverage: float = 0.50,
    output_dir: str | Path = DEFAULT_CAPITAL_CYCLE_FEASIBILITY_DIR,
) -> CapitalCycleFeasibilityRun:
    """Audit whether named episodes support credible point-in-time upstream research."""

    episodes = validate_episode_manifest(episode_manifest)
    audited = audit_point_in_time_observations(observations, episodes)
    origin_coverage = build_origin_coverage_panel(
        episodes,
        audited,
        origin_frequency=origin_frequency,
        maximum_age_days=maximum_age_days,
    )
    episode_coverage = summarize_episode_coverage(origin_coverage)
    family_coverage = summarize_family_coverage(origin_coverage)
    source_gaps = build_source_gap_register(
        episodes,
        audited,
        source_registry=source_registry,
    )
    readiness = build_feasibility_readiness(
        episodes,
        audited,
        origin_coverage,
        episode_coverage,
        family_coverage,
        minimum_origin_coverage=minimum_origin_coverage,
        minimum_episode_coverage=minimum_episode_coverage,
    )
    frames = {
        "episode_manifest": episodes,
        "observation_audit": audited,
        "origin_coverage": origin_coverage,
        "episode_coverage": episode_coverage,
        "family_coverage": family_coverage,
        "source_gaps": source_gaps,
        "readiness": readiness,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = destination / f"capital_cycle_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return CapitalCycleFeasibilityRun(output_paths=paths, **frames)


def validate_episode_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_EPISODE_COLUMNS if column not in manifest]
    if missing:
        raise ValueError(f"Episode manifest is missing columns: {', '.join(missing)}")
    clean = manifest.copy()
    for column in (
        "start_date",
        "end_date",
        "accumulation_end",
        "first_deterioration_date",
        "economic_rollover_date",
        "market_recognition_date",
        "preregistered_at",
    ):
        if column in clean:
            clean[column] = pd.to_datetime(clean[column], errors="coerce")
    if clean["episode"].astype(str).duplicated().any():
        raise ValueError("Episode names must be unique.")
    if not bool(clean["episode_type"].astype(str).isin({"break", "negative_control"}).all()):
        raise ValueError("episode_type must be break or negative_control.")
    if clean[["start_date", "end_date", "preregistered_at"]].isna().any().any():
        raise ValueError("Episode start, end, and preregistration dates are required.")
    if clean["start_date"].gt(clean["end_date"]).any():
        raise ValueError("Episode start_date must not follow end_date.")
    clean["allocation_authority"] = 0.0
    return clean.sort_values("start_date").reset_index(drop=True)


def audit_point_in_time_observations(
    observations: pd.DataFrame,
    episode_manifest: pd.DataFrame,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(columns=(*REQUIRED_OBSERVATION_COLUMNS, "eligible_for_model"))
    missing = [column for column in REQUIRED_OBSERVATION_COLUMNS if column not in observations]
    if missing:
        raise ValueError(f"Observation panel is missing columns: {', '.join(missing)}")
    clean = observations.copy()
    clean["period_end"] = pd.to_datetime(
        clean["period_end"], errors="coerce", format="mixed"
    )
    clean["available_date"] = pd.to_datetime(
        clean["available_date"], errors="coerce", format="mixed"
    )
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    episode_names = set(episode_manifest["episode"].astype(str))
    clean["episode_registered"] = clean["episode"].astype(str).isin(episode_names)
    clean["family_registered"] = clean["family"].astype(str).isin(REQUIRED_DIRECT_FAMILIES)
    clean["availability_valid"] = (
        clean["period_end"].notna()
        & clean["available_date"].notna()
        & clean["available_date"].ge(clean["period_end"])
    )
    clean["source_traceable"] = (
        clean["source_id"].fillna("").astype(str).str.len().gt(0)
        & clean["source_url"].fillna("").astype(str).str.startswith("http")
    )
    clean["vintage_eligible"] = clean["vintage_status"].astype(str).isin(
        ELIGIBLE_VINTAGE_STATUSES
    )
    declared = (
        clean["eligible_for_model"].fillna(False).astype(bool)
        if "eligible_for_model" in clean
        else pd.Series(True, index=clean.index)
    )
    clean["eligible_for_model"] = (
        declared
        & clean["episode_registered"]
        & clean["family_registered"]
        & clean["availability_valid"]
        & clean["source_traceable"]
        & clean["vintage_eligible"]
        & clean["value"].notna()
    )
    clean["exclusion_reason"] = clean.apply(_observation_exclusion_reason, axis=1)
    clean["allocation_authority"] = 0.0
    return clean.sort_values(["episode", "available_date", "family", "metric"]).reset_index(
        drop=True
    )


def build_origin_coverage_panel(
    episode_manifest: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    origin_frequency: str,
    maximum_age_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = observations[observations.get("eligible_for_model", False).eq(True)].copy()
    for _, episode in episode_manifest.iterrows():
        origins = pd.date_range(
            pd.Timestamp(episode["start_date"]),
            pd.Timestamp(episode["end_date"]),
            freq=origin_frequency,
        )
        episode_rows = eligible[eligible["episode"].astype(str).eq(str(episode["episode"]))]
        for origin in origins:
            row: dict[str, object] = {
                "episode": episode["episode"],
                "episode_family": episode["episode_family"],
                "episode_type": episode["episode_type"],
                "origin_date": origin,
            }
            supported = 0
            for family in REQUIRED_DIRECT_FAMILIES:
                family_rows = episode_rows[
                    episode_rows["family"].astype(str).eq(family)
                    & episode_rows["available_date"].le(origin)
                ]
                if family_rows.empty:
                    family_available = False
                    latest_date = pd.NaT
                else:
                    latest_date = pd.Timestamp(family_rows["available_date"].max())
                    family_available = (origin - latest_date).days <= maximum_age_days
                row[f"family__{family}"] = float(family_available)
                row[f"latest__{family}"] = latest_date
                supported += int(family_available)
            row["direct_families"] = supported
            row["direct_family_coverage"] = supported / len(REQUIRED_DIRECT_FAMILIES)
            row["allocation_authority"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_episode_coverage(origin_coverage: pd.DataFrame) -> pd.DataFrame:
    if origin_coverage.empty:
        return pd.DataFrame()
    rows = []
    for (episode, episode_family, episode_type), group in origin_coverage.groupby(
        ["episode", "episode_family", "episode_type"]
    ):
        coverage = pd.to_numeric(group["direct_family_coverage"], errors="coerce")
        rows.append(
            {
                "episode": episode,
                "episode_family": episode_family,
                "episode_type": episode_type,
                "origins": len(group),
                "mean_direct_family_coverage": coverage.mean(),
                "median_direct_family_coverage": coverage.median(),
                "origins_at_60pct": float(coverage.ge(0.60).mean()),
                "origins_at_70pct": float(coverage.ge(0.70).mean()),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)


def summarize_family_coverage(origin_coverage: pd.DataFrame) -> pd.DataFrame:
    if origin_coverage.empty:
        return pd.DataFrame()
    rows = []
    for family in REQUIRED_DIRECT_FAMILIES:
        column = f"family__{family}"
        values = pd.to_numeric(origin_coverage[column], errors="coerce").fillna(0.0)
        episodes = origin_coverage.assign(_supported=values).groupby("episode")["_supported"].max()
        rows.append(
            {
                "family": family,
                "origin_coverage": values.mean(),
                "episodes_with_any_coverage": int(episodes.gt(0).sum()),
                "episode_coverage": float(episodes.gt(0).mean()),
                "allocation_authority": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_source_gap_register(
    episode_manifest: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    source_registry: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    registry = source_registry if isinstance(source_registry, pd.DataFrame) else pd.DataFrame()
    for _, episode in episode_manifest.iterrows():
        episode_name = str(episode["episode"])
        for family in REQUIRED_DIRECT_FAMILIES:
            observed = observations[
                observations.get("episode", pd.Series(dtype=str)).astype(str).eq(episode_name)
                & observations.get("family", pd.Series(dtype=str)).astype(str).eq(family)
                & observations.get("eligible_for_model", pd.Series(dtype=bool)).eq(True)
            ]
            candidates = registry.copy()
            if not candidates.empty:
                candidates = candidates[
                    candidates.get("episode", pd.Series(dtype=str)).astype(str).eq(episode_name)
                    & candidates.get("family", pd.Series(dtype=str)).astype(str).eq(family)
                ]
            rows.append(
                {
                    "episode": episode_name,
                    "family": family,
                    "eligible_observations": len(observed),
                    "candidate_sources": len(candidates),
                    "registered_acquisition_status": (
                        ";".join(
                            sorted(
                                set(
                                    candidates.get(
                                        "acquisition_status", pd.Series(dtype=str)
                                    )
                                    .dropna()
                                    .astype(str)
                                )
                            )
                        )
                        if not candidates.empty
                        else ""
                    ),
                    "registered_providers": (
                        ";".join(
                            sorted(
                                set(
                                    candidates.get("provider", pd.Series(dtype=str))
                                    .dropna()
                                    .astype(str)
                                )
                            )
                        )
                        if not candidates.empty
                        else ""
                    ),
                    "acquisition_status": (
                        "extracted_point_in_time"
                        if not observed.empty
                        else "source_identified_not_extracted"
                        if not candidates.empty
                        else "source_not_identified"
                    ),
                    "allocation_authority": 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_feasibility_readiness(
    episodes: pd.DataFrame,
    observations: pd.DataFrame,
    origin_coverage: pd.DataFrame,
    episode_coverage: pd.DataFrame,
    family_coverage: pd.DataFrame,
    *,
    minimum_origin_coverage: float,
    minimum_episode_coverage: float,
) -> pd.DataFrame:
    blockers: list[str] = []
    break_count = int(episodes["episode_type"].eq("break").sum())
    control_count = int(episodes["episode_type"].eq("negative_control").sum())
    if break_count < 3:
        blockers.append("fewer_than_three_preregistered_break_episodes")
    if control_count < 2:
        blockers.append("fewer_than_two_preregistered_negative_controls")
    eligible = int(observations.get("eligible_for_model", pd.Series(dtype=bool)).eq(True).sum())
    if eligible == 0:
        blockers.append("no_point_in_time_direct_observations")
    overall_coverage = (
        float(origin_coverage["direct_family_coverage"].mean())
        if not origin_coverage.empty
        else 0.0
    )
    if overall_coverage < minimum_origin_coverage:
        blockers.append("overall_direct_family_coverage_below_threshold")
    weak_episodes = (
        episode_coverage[
            episode_coverage["mean_direct_family_coverage"].lt(minimum_episode_coverage)
        ]["episode"].astype(str).tolist()
        if not episode_coverage.empty
        else episodes["episode"].astype(str).tolist()
    )
    if weak_episodes:
        blockers.append("episode_coverage_below_threshold")
    weak_families = (
        family_coverage[family_coverage["episodes_with_any_coverage"].lt(3)]["family"]
        .astype(str)
        .tolist()
        if not family_coverage.empty
        else list(REQUIRED_DIRECT_FAMILIES)
    )
    if weak_families:
        blockers.append("direct_family_not_reconstructed_across_three_episodes")
    invalid_availability = int(
        (~observations.get("availability_valid", pd.Series(dtype=bool))).sum()
    )
    if invalid_availability:
        blockers.append("invalid_or_missing_available_dates")
    retrospective_only = int(
        observations.get("vintage_status", pd.Series(dtype=str))
        .astype(str)
        .eq("latest_vintage")
        .sum()
    )
    return pd.DataFrame(
        [
            {
                "episodes": len(episodes),
                "break_episodes": break_count,
                "negative_controls": control_count,
                "observation_rows": len(observations),
                "eligible_point_in_time_rows": eligible,
                "latest_vintage_rows_excluded": retrospective_only,
                "mean_direct_family_coverage": overall_coverage,
                "minimum_required_origin_coverage": minimum_origin_coverage,
                "weak_episodes": ";".join(weak_episodes),
                "weak_families": ";".join(weak_families),
                "feasibility_passed": not blockers,
                "economic_forecasting_unlocked": not blockers,
                "equity_timing_unlocked": False,
                "strategy_utility_unlocked": False,
                "blockers": ";".join(blockers),
                "allocation_authority": 0.0,
            }
        ]
    )


def _observation_exclusion_reason(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("episode_registered", False)):
        reasons.append("episode_not_registered")
    if not bool(row.get("family_registered", False)):
        reasons.append("family_not_registered")
    if not bool(row.get("availability_valid", False)):
        reasons.append("invalid_available_date")
    if not bool(row.get("source_traceable", False)):
        reasons.append("source_not_traceable")
    if not bool(row.get("vintage_eligible", False)):
        reasons.append("revision_or_vintage_not_safe")
    if pd.isna(row.get("value")):
        reasons.append("value_missing")
    return ";".join(reasons)
