from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.DEFAULT import DEFAULT_EXPERIMENTS_DIR


def load_experiment_scorecards(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "scorecard.csv"):
        frame.insert(0, "iteration", iteration)
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        for column, default in {
            "phase": "legacy",
            "family": "unknown",
            "parent": "",
            "role": "unknown",
            "scenario_sizing": "",
        }.items():
            if column not in frame.columns:
                frame[column] = default
            else:
                frame[column] = frame[column].fillna(default)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["iteration_rank"] = combined.groupby("iteration")["promotion_score"].rank(
        ascending=False,
        method="first",
    )
    return combined.sort_values(["iteration", "iteration_rank"])


def load_experiment_regime_metrics(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "regime_metrics.csv"):
        frame.insert(0, "iteration", iteration)
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_experiment_walk_forward(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "walk_forward_summary.csv"):
        frame.insert(0, "iteration", iteration)
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        elif "strategy" not in frame.columns:
            frame = frame.rename(columns={frame.columns[1]: "strategy"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_experiment_candidates(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    frames = []
    for iteration, frame in _load_iteration_csvs(root, "candidates.csv"):
        frame.insert(0, "iteration", iteration)
        for column, default in {
            "phase": "legacy",
            "family": "unknown",
            "parent": "",
            "role": "unknown",
            "scenario_sizing": "",
            "scenario_sizing_json": "",
        }.items():
            if column not in frame.columns:
                frame[column] = default
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_experiment_history(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty:
        return pd.DataFrame()

    return (
        scorecards.groupby(["iteration", "promotion_decision"], as_index=False)
        .agg(
            candidates=("strategy", "count"),
            best_score=("promotion_score", "max"),
            median_score=("promotion_score", "median"),
            best_calmar=("calmar", "max"),
            best_cagr=("cagr", "max"),
            best_max_drawdown=("max_drawdown", "max"),
        )
        .sort_values(["iteration", "best_score"], ascending=[True, False])
    )


def summarize_experiment_families(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty or "family" not in scorecards.columns:
        return pd.DataFrame()

    return (
        scorecards.groupby(["phase", "family"], as_index=False, dropna=False)
        .agg(
            candidates=("strategy", "count"),
            promoted=(
                "promotion_decision",
                lambda values: int((values == "promote_candidate").sum()),
            ),
            evolved=(
                "promotion_decision",
                lambda values: int((values == "evolve_next_iteration").sum()),
            ),
            best_score=("promotion_score", "max"),
            best_calmar=("calmar", "max"),
            best_cagr=("cagr", "max"),
            best_max_drawdown=("max_drawdown", "max"),
        )
        .sort_values(["best_score", "best_calmar"], ascending=False)
    )


def summarize_experiment_operating_systems(scorecards: pd.DataFrame) -> pd.DataFrame:
    if scorecards.empty:
        return pd.DataFrame()

    candidate_frame = scorecards.copy()
    if "robustness_score" not in candidate_frame:
        candidate_frame["robustness_score"] = float("nan")
    if "walk_forward_positive_rate" not in candidate_frame:
        candidate_frame["walk_forward_positive_rate"] = float("nan")
    if "left_tail_regime_cagr" not in candidate_frame:
        candidate_frame["left_tail_regime_cagr"] = float("nan")
    if "left_tail_regime_return" not in candidate_frame:
        candidate_frame["left_tail_regime_return"] = float("nan")
    robust_rows = candidate_frame[candidate_frame["robustness_score"].notna()]
    if not robust_rows.empty:
        candidate_frame = robust_rows

    sort_columns = ["promotion_score", "robustness_score", "calmar"]
    for column in sort_columns:
        if column not in candidate_frame:
            candidate_frame[column] = float("nan")
    leaders = (
        candidate_frame.sort_values(sort_columns, ascending=False)
        .drop_duplicates("family", keep="first")
        .sort_values(sort_columns, ascending=False)
    )
    columns = [
        "iteration",
        "strategy",
        "phase",
        "family",
        "role",
        "scenario_sizing",
        "promotion_decision",
        "promotion_score",
        "robustness_score",
        "cagr",
        "max_drawdown",
        "calmar",
        "walk_forward_positive_rate",
        "left_tail_regime_return",
        "left_tail_regime_cagr",
        "hypothesis",
    ]
    return leaders[[column for column in columns if column in leaders.columns]]


def latest_experiment_iteration(scorecards: pd.DataFrame) -> int | None:
    if scorecards.empty or "iteration" not in scorecards:
        return None
    return int(scorecards["iteration"].max())


def _load_iteration_csvs(root: str | Path, filename: str) -> list[tuple[int, pd.DataFrame]]:
    experiment_root = Path(root)
    if not experiment_root.exists():
        return []

    frames = []
    for path in sorted(experiment_root.glob(f"iteration_*/{filename}")):
        frames.append((_iteration_from_path(path), pd.read_csv(path)))
    return frames


def _iteration_from_path(path: Path) -> int:
    name = path.parent.name
    try:
        return int(name.split("_")[-1])
    except (IndexError, ValueError):
        return -1
