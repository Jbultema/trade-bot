from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.research.capital_cycle_ai_pipeline import audit_ai_family_coverage
from trade_bot.research.capital_cycle_feasibility import REQUIRED_DIRECT_FAMILIES

DEFAULT_AI_VINTAGE_DIR = Path("research/capital_cycle/prospective_ai_vintages")


@dataclass(frozen=True)
class ProspectiveAiVintage:
    observations: pd.DataFrame
    missingness: pd.DataFrame
    observation_path: Path
    missingness_path: Path
    manifest_path: Path


def save_prospective_ai_vintage(
    *,
    observations: pd.DataFrame,
    as_of: str | pd.Timestamp,
    output_dir: str | Path = DEFAULT_AI_VINTAGE_DIR,
) -> ProspectiveAiVintage:
    """Save an immutable monthly AI evidence vintage, including explicit missingness."""

    as_of_date = pd.Timestamp(as_of).normalize()
    month = as_of_date.strftime("%Y-%m")
    destination = Path(output_dir) / month
    if destination.exists():
        raise FileExistsError(
            f"Prospective AI vintage {month} already exists and is immutable: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    clean = observations.copy()
    if clean.empty:
        clean = pd.DataFrame(
            columns=[
                "family",
                "metric",
                "entity",
                "period_end",
                "available_date",
                "value",
                "source_id",
                "source_url",
                "vintage_status",
            ]
        )
    for column in ("period_end", "available_date"):
        if column in clean:
            clean[column] = pd.to_datetime(clean[column], errors="coerce")
    if "available_date" in clean:
        clean = clean[clean["available_date"].le(as_of_date)].copy()
    if "family" in clean:
        clean = clean[clean["family"].astype(str).isin(REQUIRED_DIRECT_FAMILIES)].copy()
    clean["vintage_as_of"] = as_of_date
    clean["allocation_authority"] = 0.0
    missingness = audit_ai_family_coverage(clean, as_of=as_of_date)
    observation_path = destination / "observations.csv"
    missingness_path = destination / "missingness.csv"
    clean.to_csv(observation_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    manifest = {
        "vintage_month": month,
        "as_of": as_of_date.date().isoformat(),
        "observation_rows": len(clean),
        "observed_families": int(missingness["observed"].sum()),
        "available_families": int(missingness["available"].sum()),
        "required_families": len(REQUIRED_DIRECT_FAMILIES),
        "observations_sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
        "missingness_sha256": hashlib.sha256(missingness_path.read_bytes()).hexdigest(),
        "immutable": True,
        "allocation_authority": 0.0,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return ProspectiveAiVintage(
        observations=clean,
        missingness=missingness,
        observation_path=observation_path,
        missingness_path=missingness_path,
        manifest_path=manifest_path,
    )
