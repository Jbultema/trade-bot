from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import BotConfig, configured_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.DEFAULTS import DEFAULT_EXPERIMENTS_DIR, DEFAULT_RESET_EXPERIMENTS_DIR
from trade_bot.features.indicators import daily_returns
from trade_bot.research.curation import rank_strategy_candidates
from trade_bot.research.experiments import (
    ExperimentCandidate,
    _candidate_tickers,
    _load_previous_candidates,
    _load_previous_scorecards,
    _strategy_prices,
    apply_decision_sanity_overlay,
    apply_operability_hysteresis,
    apply_scenario_position_sizing,
    generate_iteration_candidates,
)
from trade_bot.research.future_state_ml import (
    apply_future_state_position_sizing,
    apply_strategy_drawdown_position_sizing,
)
from trade_bot.research.prebreak_hindsight import (
    _hard_defense_source,
    _policy_confirm_gate_budget,
    _policy_portfolio_confirm_budget,
    _policy_portfolio_watch_floor_budget,
    _policy_stage_floor_budget,
    _policy_watch_warning_floor_budget,
    _safe_float,
)
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_POLICY_BACKTEST_OUTPUT_DIR = Path("reports/prebreak_risk_policy_backtest")
DEFAULT_POLICY_SNAPSHOT_PATH = Path("reports/prebreak_hindsight/snapshot_signal_panel.csv")
DEFAULT_POLICY_MAX_FORWARD_FILL_DAYS = 10


@dataclass(frozen=True)
class RiskPolicyBacktestResult:
    strategy_policy_metrics: pd.DataFrame
    policy_summary: pd.DataFrame
    coverage: pd.DataFrame
    summary: str


BudgetPolicy = Callable[[pd.Series], float]


def run_prebreak_risk_policy_backtest(
    config: BotConfig,
    *,
    iteration: int = 164,
    snapshot_path: str | Path = DEFAULT_POLICY_SNAPSHOT_PATH,
    experiment_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_POLICY_BACKTEST_OUTPUT_DIR,
    top_n: int = 8,
    max_forward_fill_days: int = DEFAULT_POLICY_MAX_FORWARD_FILL_DAYS,
    refresh_data: bool = False,
) -> RiskPolicyBacktestResult:
    experiment_root = Path(experiment_root) if experiment_root else _active_experiment_root()
    snapshot_signals = pd.read_csv(snapshot_path)
    scorecards = _load_previous_scorecards(experiment_root, iteration + 1)
    candidates_manifest = _load_previous_candidates(experiment_root, iteration + 1)
    candidates = _selected_candidates(
        iteration,
        scorecards=scorecards,
        candidates_manifest=candidates_manifest,
        top_n=top_n,
        experiment_root=experiment_root,
    )
    tickers = sorted(set(configured_tickers(config)) | _candidate_tickers(candidates))
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    )

    policy_budgets = build_policy_budget_frame(
        snapshot_signals,
        prices.index,
        max_forward_fill_days=max_forward_fill_days,
    )
    coverage = _coverage_frame(policy_budgets)
    policy_metrics: list[PerformanceMetrics] = []
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_prices = _strategy_prices(
            prices,
            candidate.strategy.tickers,
            candidate.strategy.defensive_ticker,
        )
        base_result = _run_candidate_backtest(config, candidate, prices, candidate_prices)
        policy_metrics.append(
            calculate_metrics(
                name=f"{candidate.name}__base",
                returns=base_result.returns,
                equity=base_result.equity,
                turnover=base_result.turnover,
                transaction_costs=base_result.transaction_costs,
            )
        )
        rows.append(
            _metadata_row(
                candidate,
                policy_name="base",
                base_result=base_result,
                policy_result=base_result,
                policy_budgets=policy_budgets.get("base"),
            )
        )
        for policy_name in policy_budgets.columns:
            if policy_name == "base":
                continue
            policy_budget = policy_budgets[policy_name].reindex(base_result.weights.index)
            policy_result = apply_budget_overlay_to_result(
                base_result,
                candidate_prices,
                policy_budget,
                defensive_ticker=candidate.strategy.defensive_ticker or "BIL",
                transaction_cost_bps=config.execution.transaction_cost_bps,
                name=f"{candidate.name}__{policy_name}",
            )
            policy_metrics.append(
                calculate_metrics(
                    name=policy_result.name,
                    returns=policy_result.returns,
                    equity=policy_result.equity,
                    turnover=policy_result.turnover,
                    transaction_costs=policy_result.transaction_costs,
                )
            )
            rows.append(
                _metadata_row(
                    candidate,
                    policy_name=policy_name,
                    base_result=base_result,
                    policy_result=policy_result,
                    policy_budgets=policy_budget,
                )
            )

    metrics = pd.DataFrame([metric.__dict__ for metric in policy_metrics]).set_index("name")
    metadata = pd.DataFrame(rows)
    strategy_policy_metrics = metadata.join(metrics, on="result_name")
    policy_summary = summarize_policy_backtest(strategy_policy_metrics)
    summary = build_policy_backtest_summary(policy_summary, coverage)
    write_policy_backtest_outputs(
        RiskPolicyBacktestResult(
            strategy_policy_metrics=strategy_policy_metrics,
            policy_summary=policy_summary,
            coverage=coverage,
            summary=summary,
        ),
        output_dir=output_dir,
    )
    return RiskPolicyBacktestResult(
        strategy_policy_metrics=strategy_policy_metrics,
        policy_summary=policy_summary,
        coverage=coverage,
        summary=summary,
    )


def build_policy_budget_frame(
    snapshot_signals: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    max_forward_fill_days: int = DEFAULT_POLICY_MAX_FORWARD_FILL_DAYS,
) -> pd.DataFrame:
    policies: dict[str, BudgetPolicy] = {
        "actual_snapshot_budget": lambda row: _safe_float(
            row.get("risk_budget_multiplier", 1.0),
            1.0,
        ),
        "hindsight_stage_floor": _policy_stage_floor_budget,
        "hindsight_watch_warning_floor": _policy_watch_warning_floor_budget,
        "hindsight_portfolio_watch_floor": _policy_portfolio_watch_floor_budget,
        "hindsight_portfolio_confirm_30d": lambda row: _policy_portfolio_confirm_budget(row, 30),
        "hindsight_stage_confirm_30d": lambda row: _policy_confirm_gate_budget(row, 30),
        "signal_confirm_floor_75": lambda row: _signal_confirm_floor_budget(row, 0.75),
        "signal_confirm_floor_65": lambda row: _signal_confirm_floor_budget(row, 0.65),
        "signal_confirm_floor_85": lambda row: _signal_confirm_floor_budget(row, 0.85),
    }
    indexed = _snapshot_rows_by_date(snapshot_signals)
    budgets = pd.DataFrame(index=pd.DatetimeIndex(trading_index).sort_values())
    budgets["base"] = 1.0
    for policy_name, policy in policies.items():
        dated = indexed.apply(policy, axis=1).astype(float).clip(0.0, 1.0)
        budgets[policy_name] = _align_sparse_budget(
            dated,
            budgets.index,
            max_forward_fill_days=max_forward_fill_days,
        )
    return budgets


def apply_budget_overlay_to_result(
    result: BacktestResult,
    prices: pd.DataFrame,
    policy_budget: pd.Series,
    *,
    defensive_ticker: str,
    transaction_cost_bps: float,
    name: str,
) -> BacktestResult:
    weights = result.weights.copy().astype(float)
    if defensive_ticker not in weights:
        weights[defensive_ticker] = 0.0
    budget = policy_budget.reindex(weights.index).fillna(1.0).clip(0.0, 1.0)
    risk_columns = [column for column in weights.columns if column != defensive_ticker]
    adjusted = weights.copy()
    risk_weight = adjusted[risk_columns].sum(axis=1) if risk_columns else pd.Series(0.0, index=weights.index)
    adjusted.loc[:, risk_columns] = adjusted[risk_columns].mul(budget, axis=0)
    adjusted[defensive_ticker] = adjusted[defensive_ticker] + risk_weight * (1.0 - budget)
    adjusted = adjusted.reindex(columns=prices.columns, fill_value=0.0)

    asset_returns = daily_returns(prices).reindex(adjusted.index).fillna(0.0)
    turnover = adjusted.diff().abs().sum(axis=1).fillna(adjusted.abs().sum(axis=1))
    transaction_costs = turnover * transaction_cost_bps / 10000.0
    gross_returns = (adjusted * asset_returns).sum(axis=1)
    net_returns = gross_returns - transaction_costs
    equity = result.equity.iloc[0] / (1.0 + result.returns.iloc[0]) * (1.0 + net_returns).cumprod()
    return BacktestResult(
        name=name,
        equity=equity.rename(name),
        returns=net_returns.rename(name),
        gross_returns=gross_returns.rename(name),
        weights=adjusted,
        target_weights=result.target_weights,
        turnover=turnover.rename(name),
        transaction_costs=transaction_costs.rename(name),
    )


def summarize_policy_backtest(strategy_policy_metrics: pd.DataFrame) -> pd.DataFrame:
    if strategy_policy_metrics.empty:
        return pd.DataFrame()
    metric_columns = [
        "cagr",
        "max_drawdown",
        "calmar",
        "sharpe",
        "average_turnover",
        "budget_active_day_rate",
        "median_policy_budget",
    ]
    rows: list[dict[str, object]] = []
    for policy_name, group in strategy_policy_metrics.groupby("policy_name", sort=False):
        row: dict[str, object] = {
            "policy_name": policy_name,
            "strategies": len(group),
        }
        for column in metric_columns:
            row[f"median_{column}"] = pd.to_numeric(group[column], errors="coerce").median()
        row["median_delta_cagr_vs_base"] = pd.to_numeric(
            group["delta_cagr_vs_base"],
            errors="coerce",
        ).median()
        row["median_delta_max_drawdown_vs_base"] = pd.to_numeric(
            group["delta_max_drawdown_vs_base"],
            errors="coerce",
        ).median()
        row["win_rate_cagr_vs_base"] = float((group["delta_cagr_vs_base"] > 0).mean())
        row["win_rate_drawdown_vs_base"] = float((group["delta_max_drawdown_vs_base"] > 0).mean())
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["median_delta_cagr_vs_base", "median_delta_max_drawdown_vs_base"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_policy_backtest_summary(policy_summary: pd.DataFrame, coverage: pd.DataFrame) -> str:
    lines = [
        "# Pre-Break Risk Policy Backtest",
        "",
        "This report applies sparse saved snapshot risk budgets on top of the selected",
        "top experiment strategies. Hindsight-stage policies use known historical break",
        "windows and are crisis-playback research, not deployable trading rules.",
        "",
        "## Snapshot Coverage",
        "",
    ]
    if coverage.empty:
        lines.append("- no policy coverage rows were available")
    else:
        for _, row in coverage.iterrows():
            lines.append(
                "- "
                f"{row['policy_name']}: active on {_safe_float(row['active_day_rate']):.1%} "
                f"of trading days; median budget {_safe_float(row['median_budget']):.1%}"
            )
    lines.extend(["", "## Policy Ranking", ""])
    if policy_summary.empty:
        lines.append("No policy backtest rows were available.")
    else:
        for _, row in policy_summary.head(10).iterrows():
            lines.append(
                "- "
                f"{row['policy_name']}: median CAGR {_safe_float(row['median_cagr']):.2%}, "
                f"median max DD {_safe_float(row['median_max_drawdown']):.2%}, "
                f"median delta CAGR {_safe_float(row['median_delta_cagr_vs_base']):+.2%}, "
                f"median DD improvement {_safe_float(row['median_delta_max_drawdown_vs_base']):+.2%}"
            )
    return "\n".join(lines)


def write_policy_backtest_outputs(
    result: RiskPolicyBacktestResult,
    *,
    output_dir: str | Path = DEFAULT_POLICY_BACKTEST_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.strategy_policy_metrics.to_csv(output_path / "strategy_policy_metrics.csv", index=False)
    result.policy_summary.to_csv(output_path / "policy_summary.csv", index=False)
    result.coverage.to_csv(output_path / "coverage.csv", index=False)
    (output_path / "summary.md").write_text(result.summary, encoding="utf-8")


def _run_candidate_backtest(
    config: BotConfig,
    candidate: ExperimentCandidate,
    full_prices: pd.DataFrame,
    candidate_prices: pd.DataFrame,
) -> BacktestResult:
    base_target_weights = build_strategy_weights(candidate_prices, candidate.strategy)
    target_weights = base_target_weights
    if candidate.future_state_model is not None:
        target_weights = apply_future_state_position_sizing(
            target_weights,
            full_prices,
            candidate.future_state_model,
            defensive_ticker=candidate.strategy.defensive_ticker,
        )
    if candidate.scenario_sizing is not None:
        target_weights = apply_scenario_position_sizing(
            target_weights,
            candidate_prices,
            candidate.scenario_sizing,
            defensive_ticker=candidate.strategy.defensive_ticker,
        )
    if candidate.strategy_drawdown_model is not None:
        target_weights = apply_strategy_drawdown_position_sizing(
            target_weights,
            full_prices,
            candidate.strategy_drawdown_model,
            defensive_ticker=candidate.strategy.defensive_ticker,
        )
    if candidate.decision_sanity is not None:
        target_weights = apply_decision_sanity_overlay(
            base_target_weights,
            target_weights,
            candidate_prices,
            candidate.decision_sanity,
            defensive_ticker=candidate.strategy.defensive_ticker,
        )
    target_weights = apply_operability_hysteresis(target_weights, candidate.strategy)
    return run_backtest(
        candidate.name,
        candidate_prices,
        target_weights,
        config.execution,
        volatility_target=candidate.strategy.volatility_target,
        drawdown_control=candidate.strategy.drawdown_control,
    )


def _selected_candidates(
    iteration: int,
    *,
    scorecards: pd.DataFrame,
    candidates_manifest: pd.DataFrame,
    top_n: int,
    experiment_root: Path,
) -> tuple[ExperimentCandidate, ...]:
    candidates = generate_iteration_candidates(
        iteration,
        previous_scorecards=_load_previous_scorecards(experiment_root, iteration),
        previous_candidates=_load_previous_candidates(experiment_root, iteration),
    )
    if scorecards.empty:
        return candidates[:top_n]
    ranked = rank_strategy_candidates(scorecards)
    names = ranked["strategy"].astype(str).tolist() if "strategy" in ranked else []
    manifest_names = set(candidates_manifest["strategy"].astype(str)) if not candidates_manifest.empty else set()
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    selected = [
        candidate_by_name[name]
        for name in names
        if name in candidate_by_name and (not manifest_names or name in manifest_names)
    ]
    return tuple(selected[:top_n] or candidates[:top_n])


def _active_experiment_root() -> Path:
    if DEFAULT_RESET_EXPERIMENTS_DIR.exists():
        return DEFAULT_RESET_EXPERIMENTS_DIR
    return DEFAULT_EXPERIMENTS_DIR


def _snapshot_rows_by_date(snapshot_signals: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot_signals.copy()
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    frame = frame.dropna(subset=["market_date"])
    frame["market_date"] = frame["market_date"].dt.normalize()
    frame["risk_budget_multiplier"] = pd.to_numeric(
        frame.get("risk_budget_multiplier", 1.0),
        errors="coerce",
    ).fillna(1.0)
    if "event_name" in frame:
        event_rows = frame[frame["event_name"].fillna("").astype(str).ne("")]
        if not event_rows.empty:
            frame = event_rows
    policy_rows = []
    for market_date, group in frame.groupby("market_date", sort=True):
        sorted_group = group.sort_values("risk_budget_multiplier", ascending=True)
        policy_rows.append(sorted_group.iloc[0].copy())
        policy_rows[-1]["market_date"] = market_date
    return pd.DataFrame(policy_rows).set_index("market_date").sort_index()


def _align_sparse_budget(
    dated_budget: pd.Series,
    trading_index: pd.DatetimeIndex,
    *,
    max_forward_fill_days: int,
) -> pd.Series:
    budget = pd.Series(1.0, index=trading_index)
    dated_budget = dated_budget.dropna().sort_index()
    for date, value in dated_budget.items():
        start = pd.Timestamp(date).normalize()
        end = start + pd.Timedelta(days=max_forward_fill_days)
        mask = (budget.index >= start) & (budget.index <= end)
        budget.loc[mask] = budget.loc[mask].clip(upper=float(value))
    return budget


def _signal_confirm_floor_budget(row: pd.Series, floor: float) -> float:
    actual = _safe_float(row.get("risk_budget_multiplier", 1.0), 1.0)
    if not _hard_defense_source(row).startswith("portfolio_"):
        return actual
    break_count = _safe_float(row.get("decision_sanity_break_count", 0.0), 0.0)
    if break_count >= 2:
        return actual
    return max(actual, floor)


def _coverage_frame(policy_budgets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy_name, budget in policy_budgets.items():
        rows.append(
            {
                "policy_name": policy_name,
                "trading_days": len(budget),
                "active_days": int((budget < 0.999).sum()),
                "active_day_rate": float((budget < 0.999).mean()),
                "median_budget": float(budget.median()),
                "min_budget": float(budget.min()),
            }
        )
    return pd.DataFrame(rows)


def _metadata_row(
    candidate: ExperimentCandidate,
    *,
    policy_name: str,
    base_result: BacktestResult,
    policy_result: BacktestResult,
    policy_budgets: pd.Series | None,
) -> dict[str, object]:
    base_metrics = calculate_metrics(
        name=base_result.name,
        returns=base_result.returns,
        equity=base_result.equity,
        turnover=base_result.turnover,
        transaction_costs=base_result.transaction_costs,
    )
    policy_metrics = calculate_metrics(
        name=policy_result.name,
        returns=policy_result.returns,
        equity=policy_result.equity,
        turnover=policy_result.turnover,
        transaction_costs=policy_result.transaction_costs,
    )
    base_max_drawdown_date = _max_drawdown_date(base_result.equity)
    policy_max_drawdown_date = _max_drawdown_date(policy_result.equity)
    budget = (
        policy_budgets.reindex(policy_result.returns.index).fillna(1.0)
        if policy_budgets is not None
        else pd.Series(1.0, index=policy_result.returns.index)
    )
    return {
        "strategy": candidate.name,
        "policy_name": policy_name,
        "result_name": policy_result.name if policy_name != "base" else f"{candidate.name}__base",
        "family": candidate.family,
        "phase": candidate.phase,
        "role": candidate.role,
        "parent": candidate.parent or "",
        "base_cagr": base_metrics.cagr,
        "base_max_drawdown": base_metrics.max_drawdown,
        "base_max_drawdown_date": str(base_max_drawdown_date.date()),
        "base_calmar": base_metrics.calmar,
        "policy_cagr": policy_metrics.cagr,
        "policy_max_drawdown": policy_metrics.max_drawdown,
        "policy_max_drawdown_date": str(policy_max_drawdown_date.date()),
        "policy_calmar": policy_metrics.calmar,
        "delta_cagr_vs_base": policy_metrics.cagr - base_metrics.cagr,
        "delta_max_drawdown_vs_base": policy_metrics.max_drawdown - base_metrics.max_drawdown,
        "delta_calmar_vs_base": policy_metrics.calmar - base_metrics.calmar,
        "budget_active_day_rate": float((budget < 0.999).mean()),
        "median_policy_budget": float(budget.median()),
        "min_policy_budget": float(budget.min()),
        "budget_on_base_max_drawdown_date": _safe_float(
            budget.reindex([base_max_drawdown_date]).iloc[0],
            1.0,
        ),
    }


def _max_drawdown_date(equity: pd.Series) -> pd.Timestamp:
    clean = equity.dropna()
    if clean.empty:
        return pd.Timestamp("1970-01-01")
    drawdown = clean / clean.cummax() - 1.0
    return pd.Timestamp(drawdown.idxmin())
