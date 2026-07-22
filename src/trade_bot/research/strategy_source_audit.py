from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_RESET_EXPERIMENTS_DIR,
    DEFAULT_RUN_STORE_DB_PATH,
    DEFAULT_STRATEGY_SOURCE_AUDIT_DIR,
)
from trade_bot.storage.warehouse import TradingWarehouse

DEFAULT_HIGH_CAGR_MIN = 0.20
DEFAULT_HIGH_CAGR_MAX = 0.24
_CAGR_COLUMN_PATTERN = re.compile(r"(^|_)cagr($|_)|cagr", re.IGNORECASE)
_TEXT_REFERENCE_PATTERN = re.compile(
    r"(20\s*-\s*22|22\s*percent|22%|0\.22|twenty[- ]two)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StrategySourceAudit:
    full_history_top: pd.DataFrame
    high_cagr_metric_hits: pd.DataFrame
    ambiguous_references: pd.DataFrame
    summary: str


def build_strategy_source_audit(
    *,
    warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    experiment_roots: tuple[str | Path, ...] = (
        DEFAULT_EXPERIMENTS_DIR,
        DEFAULT_RESET_EXPERIMENTS_DIR,
    ),
    text_roots: tuple[str | Path, ...] = (Path("README.md"), Path("docs")),
    scan_roots: tuple[str | Path, ...] = (Path("reports"), Path("data")),
    high_cagr_min: float = DEFAULT_HIGH_CAGR_MIN,
    high_cagr_max: float = DEFAULT_HIGH_CAGR_MAX,
    top_n: int = 50,
) -> StrategySourceAudit:
    """Reconcile where the apparent top-CAGR strategies are actually stored."""

    full_history = pd.concat(
        [
            _snapshot_strategy_metrics(warehouse_path),
            *(_experiment_scorecards(root) for root in experiment_roots),
        ],
        ignore_index=True,
    )
    if not full_history.empty:
        full_history = _sort_metric_frame(full_history).head(top_n).reset_index(drop=True)

    high_cagr_hits = _high_cagr_metric_hits(
        scan_roots=scan_roots,
        warehouse_path=warehouse_path,
        high_cagr_min=high_cagr_min,
        high_cagr_max=high_cagr_max,
    )
    ambiguous = _ambiguous_text_references(text_roots)
    summary = _summary_markdown(
        full_history,
        high_cagr_hits,
        ambiguous,
        high_cagr_min=high_cagr_min,
        high_cagr_max=high_cagr_max,
    )
    return StrategySourceAudit(
        full_history_top=full_history,
        high_cagr_metric_hits=high_cagr_hits,
        ambiguous_references=ambiguous,
        summary=summary,
    )


def write_strategy_source_audit(
    *,
    output_dir: str | Path = DEFAULT_STRATEGY_SOURCE_AUDIT_DIR,
    warehouse_path: str | Path = DEFAULT_RUN_STORE_DB_PATH,
    experiment_roots: tuple[str | Path, ...] = (
        DEFAULT_EXPERIMENTS_DIR,
        DEFAULT_RESET_EXPERIMENTS_DIR,
    ),
    text_roots: tuple[str | Path, ...] = (Path("README.md"), Path("docs")),
    scan_roots: tuple[str | Path, ...] = (Path("reports"), Path("data")),
    high_cagr_min: float = DEFAULT_HIGH_CAGR_MIN,
    high_cagr_max: float = DEFAULT_HIGH_CAGR_MAX,
    top_n: int = 50,
) -> StrategySourceAudit:
    audit = build_strategy_source_audit(
        warehouse_path=warehouse_path,
        experiment_roots=experiment_roots,
        text_roots=text_roots,
        scan_roots=scan_roots,
        high_cagr_min=high_cagr_min,
        high_cagr_max=high_cagr_max,
        top_n=top_n,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audit.full_history_top.to_csv(output_path / "full_history_top_strategies.csv", index=False)
    audit.high_cagr_metric_hits.to_csv(output_path / "high_cagr_metric_hits.csv", index=False)
    audit.ambiguous_references.to_csv(output_path / "ambiguous_22pct_references.csv", index=False)
    (output_path / "summary.md").write_text(audit.summary, encoding="utf-8")
    return audit


def _snapshot_strategy_metrics(warehouse_path: str | Path) -> pd.DataFrame:
    warehouse = TradingWarehouse(warehouse_path)
    frame = warehouse.read_table("snapshot_strategy_metrics")
    required = {"strategy", "cagr", "max_drawdown"}
    if frame.empty or not required.issubset(frame.columns):
        return _empty_full_history_frame()
    frame = frame.copy()
    if "updated_at_utc" in frame:
        latest_time = str(frame["updated_at_utc"].dropna().astype(str).max())
        frame = frame[frame["updated_at_utc"].astype(str) == latest_time]
    columns = {
        "strategy": "strategy",
        "cagr": "cagr",
        "max_drawdown": "max_drawdown",
        "calmar": "calmar",
        "sharpe": "sharpe",
        "final_equity": "final_equity",
        "run_id": "run_id",
        "market_date": "market_date",
    }
    output = _select_columns(frame, columns)
    output["source_scope"] = "runtime_snapshot_full_history"
    output["source_path"] = str(warehouse_path)
    output["iteration"] = pd.NA
    return _normalize_metric_columns(output)


def _experiment_scorecards(root: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(root).glob("iteration_*/scorecard.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        if "strategy" not in frame or "cagr" not in frame:
            continue
        columns = {
            "strategy": "strategy",
            "cagr": "cagr",
            "max_drawdown": "max_drawdown",
            "calmar": "calmar",
            "sharpe": "sharpe",
            "promotion_score": "promotion_score",
            "growth_constrained_utility_score": "growth_constrained_utility_score",
        }
        selected = _select_columns(frame, columns)
        selected["iteration"] = _iteration_from_path(path)
        selected["source_path"] = str(path)
        selected["source_scope"] = "experiment_scorecard_full_history"
        rows.append(selected)
    if not rows:
        return _empty_full_history_frame()
    compact_rows = [row.dropna(axis=1, how="all") for row in rows]
    return _normalize_metric_columns(pd.concat(compact_rows, ignore_index=True))


def _high_cagr_metric_hits(
    *,
    scan_roots: tuple[str | Path, ...],
    warehouse_path: str | Path,
    high_cagr_min: float,
    high_cagr_max: float,
) -> pd.DataFrame:
    rows = []
    snapshot = _snapshot_strategy_metrics(warehouse_path)
    for _, row in snapshot.iterrows():
        value = _optional_float(row.get("cagr"))
        if value is not None and high_cagr_min <= value <= high_cagr_max:
            rows.append(_metric_hit(row, "snapshot_strategy_metrics", "cagr", value))

    for root in scan_roots:
        root_path = Path(root)
        paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.csv"))
        for path in paths:
            if _skip_scan_path(path):
                continue
            try:
                frame = pd.read_csv(path)
            except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if frame.empty:
                continue
            strategy_column = _strategy_column(frame)
            for column in frame.columns:
                if not _is_cagr_like_column(column):
                    continue
                values = pd.to_numeric(frame[column], errors="coerce")
                matches = frame[values.between(high_cagr_min, high_cagr_max, inclusive="both")]
                for _, match in matches.head(25).iterrows():
                    rows.append(
                        {
                            "source_scope": _metric_scope_for_path(path, column),
                            "source_path": str(path),
                            "strategy": (
                                str(match.get(strategy_column, "")) if strategy_column else ""
                            ),
                            "metric_column": column,
                            "metric_value": float(match[column]),
                            "max_drawdown": _optional_float(match.get("max_drawdown")),
                            "calmar": _optional_float(match.get("calmar")),
                            "iteration": _iteration_from_path(path),
                            "run_id": "",
                            "market_date": "",
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source_scope",
                "source_path",
                "strategy",
                "metric_column",
                "metric_value",
                "max_drawdown",
                "calmar",
                "iteration",
                "run_id",
                "market_date",
            ]
        )
    frame = pd.DataFrame(rows).drop_duplicates()
    return frame.sort_values(["source_scope", "metric_value"], ascending=[True, False])


def _ambiguous_text_references(text_roots: tuple[str | Path, ...]) -> pd.DataFrame:
    rows = []
    for root in text_roots:
        root_path = Path(root)
        paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.md"))
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if _TEXT_REFERENCE_PATTERN.search(line):
                    context = " ".join(
                        lines[max(0, line_number - 2) : min(len(lines), line_number + 1)]
                    )
                    rows.append(
                        {
                            "source_path": str(path),
                            "line_number": line_number,
                            "reference_scope": _reference_scope(context),
                            "text": line.strip(),
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=["source_path", "line_number", "reference_scope", "text"],
    )


def _metric_hit(
    row: pd.Series, source_path: str, metric_column: str, value: float
) -> dict[str, object]:
    return {
        "source_scope": str(row.get("source_scope", "")),
        "source_path": source_path,
        "strategy": str(row.get("strategy", "")),
        "metric_column": metric_column,
        "metric_value": value,
        "max_drawdown": _optional_float(row.get("max_drawdown")),
        "calmar": _optional_float(row.get("calmar")),
        "iteration": row.get("iteration", ""),
        "run_id": str(row.get("run_id", "")),
        "market_date": str(row.get("market_date", "")),
    }


def _summary_markdown(
    full_history: pd.DataFrame,
    high_cagr_hits: pd.DataFrame,
    ambiguous: pd.DataFrame,
    *,
    high_cagr_min: float,
    high_cagr_max: float,
) -> str:
    lines = [
        "# Strategy Source Audit",
        "",
        "This reconciles the sources that can make a strategy look like a top-CAGR candidate.",
        "",
    ]
    if full_history.empty:
        lines.append("No full-history strategy metrics were found.")
    else:
        best = full_history.iloc[0]
        lines.extend(
            [
                "## Best Full-History Rows",
                "",
                (
                    f"- Best row: `{best['strategy']}` from `{best['source_scope']}` "
                    f"with CAGR {_format_percent(best.get('cagr'))}, max drawdown "
                    f"{_format_percent(best.get('max_drawdown'))}, and Calmar "
                    f"{_format_float(best.get('calmar'))}."
                ),
            ]
        )
        scope_summary = (
            full_history.groupby("source_scope", as_index=False)
            .agg(
                rows=("strategy", "count"),
                best_cagr=("cagr", "max"),
                best_max_drawdown=("max_drawdown", "max"),
            )
            .sort_values("best_cagr", ascending=False)
        )
        for _, row in scope_summary.iterrows():
            lines.append(
                f"- `{row['source_scope']}`: {int(row['rows'])} rows, best CAGR "
                f"{_format_percent(row['best_cagr'])}, best max-drawdown "
                f"{_format_percent(row['best_max_drawdown'])}."
            )
    lines.extend(
        [
            "",
            "## 20-24 Percent CAGR Hits",
            "",
            (
                f"The high-CAGR screen is {high_cagr_min:.0%}-{high_cagr_max:.0%}. "
                "Rows in this band must still be judged by `source_scope` before they are "
                "called strategy champions."
            ),
            f"- Metric hits found: {len(high_cagr_hits):,}.",
            f"- Text/doc references found: {len(ambiguous):,}.",
            "",
            "## Interpretation",
            "",
            "- `runtime_snapshot_full_history` means the strategy is operable in the latest snapshot pipeline.",
            "- `experiment_scorecard_full_history` means an archived research iteration scorecard.",
            "- Rolling, yearly, window, regime, and walk-forward metrics are diagnostic slices, not whole-strategy CAGR.",
        ]
    )
    return "\n".join(lines) + "\n"


def _select_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for source, target in mapping.items():
        output[target] = frame.get(source, pd.NA)
    return output


def _normalize_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ["cagr", "max_drawdown", "calmar", "sharpe", "promotion_score", "final_equity"]:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    if "strategy" in output:
        output["strategy"] = output["strategy"].astype(str)
    return output


def _sort_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = _normalize_metric_columns(frame)
    for column in ["cagr", "max_drawdown", "calmar"]:
        if column not in output:
            output[column] = pd.NA
    return output.sort_values(["cagr", "max_drawdown", "calmar"], ascending=False)


def _empty_full_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_scope",
            "strategy",
            "cagr",
            "max_drawdown",
            "calmar",
            "sharpe",
            "final_equity",
            "promotion_score",
            "growth_constrained_utility_score",
            "iteration",
            "run_id",
            "market_date",
            "source_path",
        ]
    )


def _strategy_column(frame: pd.DataFrame) -> str | None:
    for column in ["strategy", "strategy_name", "name"]:
        if column in frame:
            return column
    return None


def _is_cagr_like_column(column: str) -> bool:
    name = column.lower()
    if "cagr" not in name:
        return False
    if any(fragment in name for fragment in ["win_rate", "rank", "score", "delta"]):
        return False
    return bool(_CAGR_COLUMN_PATTERN.search(column))


def _skip_scan_path(path: Path) -> bool:
    parts = set(path.parts)
    return ".venv" in parts or "__pycache__" in parts


def _metric_scope_for_path(path: Path, column: str) -> str:
    text = f"{path.as_posix()} {column}".lower()
    if "snapshot_strategy_metrics" in text:
        return "runtime_snapshot_full_history"
    if path.name == "scorecard.csv":
        return "experiment_scorecard_full_history"
    if "rolling" in text:
        return "rolling_window_diagnostic"
    if "yearly" in text or "calendar" in text:
        return "calendar_year_diagnostic"
    if "window" in text:
        return "window_diagnostic"
    if "walk_forward" in text:
        return "walk_forward_diagnostic"
    if "regime" in text:
        return "regime_diagnostic"
    return "noncanonical_metric_file"


def _reference_scope(line: str) -> str:
    lowered = line.lower()
    if "runtime snapshot" in lowered:
        return "runtime_snapshot_reference"
    if "drawdown system" in lowered or "comparison" in lowered:
        return "conceptual_reference"
    if "scorecard" in lowered:
        return "scorecard_reference"
    return "ambiguous_text_reference"


def _iteration_from_path(path: Path) -> int | pd.NA:
    for part in reversed(path.parts):
        if part.startswith("iteration_"):
            try:
                return int(part.split("_")[-1])
            except ValueError:
                return pd.NA
    return pd.NA


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _format_percent(value: object) -> str:
    numeric = _optional_float(value)
    return "n/a" if numeric is None else f"{numeric:.2%}"


def _format_float(value: object) -> str:
    numeric = _optional_float(value)
    return "n/a" if numeric is None else f"{numeric:.2f}"
