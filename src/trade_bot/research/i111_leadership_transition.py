from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.config import (
    BotConfig,
    ExecutionConfig,
    StrategyConfig,
    configured_tickers,
    load_config,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import unusable_required_price_columns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.risk_landscape_survey import AI_GROWTH_TICKERS
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_OUTPUT_DIR = Path("reports/i111_leadership_transition")

PERIODS: dict[str, tuple[str, str | None]] = {
    "development": ("2005-01-01", "2012-12-31"),
    "validation": ("2013-01-01", "2018-12-31"),
    "holdout": ("2019-01-01", "2022-12-31"),
    "post_selection": ("2023-01-01", None),
    "full": ("2005-01-01", None),
    "current_break": ("2026-07-25", None),
}

DIVERSIFIER_SETS: dict[str, tuple[str, ...]] = {
    "defensive_sectors": ("RSP", "XLV", "XLP", "XLU", "VTV", "USMV", "GLD", "IEF"),
    "cyclical_broadening": ("RSP", "XLF", "XLE", "XLI", "XLB", "VTV", "GLD", "IEF"),
    "all_sector_leaders": (
        "RSP",
        "XLF",
        "XLY",
        "XLP",
        "XLE",
        "XLV",
        "XLI",
        "XLU",
        "XLB",
        "GLD",
        "IEF",
    ),
    "global_quality": ("RSP", "VTV", "QUAL", "USMV", "EFA", "EWJ", "GLD", "IEF"),
}


@dataclass(frozen=True)
class TransitionPolicy:
    name: str
    soft_cap: float | None
    hard_cap: float
    soft_threshold: float
    hard_threshold: float


POLICIES = (
    TransitionPolicy("clustered_70_35", 0.70, 0.35, 0.75, 0.875),
    TransitionPolicy("confirmed_80_25", 0.80, 0.25, 0.875, 0.99),
    TransitionPolicy("extreme_only_25", None, 0.25, 0.90, 0.99),
    TransitionPolicy("graduated_65_20", 0.65, 0.20, 0.75, 0.875),
)


@dataclass(frozen=True)
class TransitionCandidate:
    name: str
    diversifier_family: str
    policy: str
    lookback_days: int
    strategy: StrategyConfig


@dataclass(frozen=True)
class LeadershipTransitionResult:
    output_dir: Path
    candidate_metrics: pd.DataFrame
    training_selection: pd.DataFrame
    execution_stress: pd.DataFrame
    current_allocations: pd.DataFrame
    report_path: Path


def build_leadership_transition_candidates(
    config: BotConfig,
) -> tuple[TransitionCandidate, ...]:
    primary = config.strategies[config.primary_strategy]
    candidates: list[TransitionCandidate] = []
    for diversifier_family, diversifiers in DIVERSIFIER_SETS.items():
        for policy in POLICIES:
            for lookback_days in (42, 63, 84):
                name = f"leadership_{diversifier_family}_{policy.name}_lb{lookback_days}"
                strategy = primary.model_copy(
                    update={
                        "type": "dual_momentum_risk_repair",
                        "risk_repair_signal": "balanced",
                        "risk_repair_constructive_floor": 0.0,
                        "risk_repair_defensive_cap": None,
                        "risk_repair_ai_soft_cap": policy.soft_cap,
                        "risk_repair_ai_hard_cap": policy.hard_cap,
                        "risk_repair_ai_soft_threshold": policy.soft_threshold,
                        "risk_repair_ai_hard_threshold": policy.hard_threshold,
                        "risk_repair_ai_cap_basis": "risk_sleeve",
                        "risk_repair_ai_excess_destination": "diversifier_mix",
                        "risk_repair_ai_diversifier_tickers": list(diversifiers),
                        "risk_repair_lookback_days": lookback_days,
                    }
                )
                candidates.append(
                    TransitionCandidate(
                        name=name,
                        diversifier_family=diversifier_family,
                        policy=policy.name,
                        lookback_days=lookback_days,
                        strategy=strategy,
                    )
                )
    return tuple(candidates)


def run_i111_leadership_transition(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    refresh_data: bool = False,
) -> LeadershipTransitionResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = build_leadership_transition_candidates(config)
    primary = config.strategies[config.primary_strategy]
    strategies: dict[str, StrategyConfig] = {"primary_reference": primary}
    strategies.update({candidate.name: candidate.strategy for candidate in candidates})
    tickers = set(configured_tickers(config))
    for strategy in strategies.values():
        tickers.update(required_strategy_tickers(strategy))
    prices = load_or_fetch_yahoo_prices(
        sorted(tickers),
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    ).sort_index()

    results = {
        name: _run_strategy(name, strategy, prices, config.execution)
        for name, strategy in strategies.items()
    }
    candidate_lookup = {candidate.name: candidate for candidate in candidates}
    candidate_metrics = _candidate_metrics(results, candidate_lookup)
    training_selection = select_training_winners(candidate_metrics)
    selected_names = list(training_selection["candidate"])
    execution_stress = _execution_stress(
        config,
        prices,
        {name: strategies[name] for name in selected_names},
    )
    current_allocations = _current_allocations(
        {name: results[name] for name in ["primary_reference", *selected_names]},
        candidate_lookup,
    )

    candidate_metrics.to_csv(output / "candidate_metrics.csv", index=False)
    training_selection.to_csv(output / "training_selection.csv", index=False)
    execution_stress.to_csv(output / "execution_stress.csv", index=False)
    current_allocations.to_csv(output / "current_allocations.csv", index=False)
    roster = pd.DataFrame(
        [
            {
                "candidate": candidate.name,
                "diversifier_family": candidate.diversifier_family,
                "policy": candidate.policy,
                "lookback_days": candidate.lookback_days,
                "diversifiers": ",".join(candidate.strategy.risk_repair_ai_diversifier_tickers),
            }
            for candidate in candidates
        ]
    )
    roster.to_csv(output / "candidate_roster.csv", index=False)
    report_path = output / "report.md"
    report_path.write_text(
        _render_report(
            candidate_metrics,
            training_selection,
            execution_stress,
            current_allocations,
            data_end=prices.index.max(),
        ),
        encoding="utf-8",
    )
    write_research_manifest(
        output,
        study="i111_leadership_transition",
        config=config,
        prices=prices,
        parameters={
            "variant_count": len(candidates),
            "selection_periods": ["development", "validation"],
            "holdout_periods_hidden_from_selection": [
                "holdout",
                "post_selection",
                "current_break",
            ],
            "diversifier_families": DIVERSIFIER_SETS,
            "policies": [policy.__dict__ for policy in POLICIES],
        },
        artifacts=[
            "candidate_metrics.csv",
            "training_selection.csv",
            "execution_stress.csv",
            "current_allocations.csv",
            "candidate_roster.csv",
            "report.md",
        ],
    )
    return LeadershipTransitionResult(
        output_dir=output,
        candidate_metrics=candidate_metrics,
        training_selection=training_selection,
        execution_stress=execution_stress,
        current_allocations=current_allocations,
        report_path=report_path,
    )


def select_training_winners(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select one candidate per replacement family without reading later periods."""

    training = metrics[metrics["period"].isin({"development", "validation"})].copy()
    if training.empty:
        return pd.DataFrame()
    scored: list[pd.DataFrame] = []
    for _, frame in training.groupby("period", sort=False):
        ranked = frame.copy()
        ranked["period_score"] = (
            0.35 * ranked["cagr"].rank(pct=True)
            + 0.30 * ranked["sharpe"].rank(pct=True)
            + 0.25 * ranked["calmar"].rank(pct=True)
            + 0.10 * (-ranked["average_turnover"]).rank(pct=True)
        )
        scored.append(ranked[["candidate", "diversifier_family", "period", "period_score"]])
    score_frame = pd.concat(scored, ignore_index=True)
    pivot = score_frame.pivot(index="candidate", columns="period", values="period_score")
    pivot["training_score"] = pivot[["development", "validation"]].min(axis=1)
    family = training[["candidate", "diversifier_family"]].drop_duplicates().set_index("candidate")
    ranked = pivot.join(family).reset_index()
    winners = (
        ranked.sort_values(
            ["diversifier_family", "training_score", "candidate"],
            ascending=[True, False, True],
        )
        .groupby("diversifier_family", as_index=False)
        .head(1)
    )
    return winners.sort_values("training_score", ascending=False).reset_index(drop=True)


def _run_strategy(
    name: str,
    strategy: StrategyConfig,
    prices: pd.DataFrame,
    execution: ExecutionConfig,
) -> BacktestResult:
    columns = required_strategy_tickers(strategy)
    missing = unusable_required_price_columns(prices, columns)
    if missing:
        raise KeyError(f"Missing, empty, or stale price columns for strategy: {missing}")
    strategy_prices = prices[columns].sort_index().dropna(how="all")
    weights = build_strategy_weights(strategy_prices, strategy)
    return run_backtest(
        name,
        strategy_prices,
        weights,
        execution,
        volatility_target=strategy.volatility_target,
        drawdown_control=strategy.drawdown_control,
    )


def _candidate_metrics(
    results: dict[str, BacktestResult],
    candidate_lookup: dict[str, TransitionCandidate],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        candidate = candidate_lookup.get(name)
        family = candidate.diversifier_family if candidate else "reference"
        policy = candidate.policy if candidate else "reference"
        for period, (start, end) in PERIODS.items():
            metrics = _period_metrics(result, start, end)
            period_weights = result.weights.loc[metrics["start"] : metrics["end"]]
            ai_columns = [column for column in period_weights if column in AI_GROWTH_TICKERS]
            active = period_weights.drop(columns=["BIL"], errors="ignore").sum(axis=1)
            ai = period_weights[ai_columns].sum(axis=1)
            rows.append(
                {
                    "candidate": name,
                    "diversifier_family": family,
                    "policy": policy,
                    "period": period,
                    **metrics,
                    "average_ai_weight": float(ai.mean()),
                    "average_ai_share_of_risk": float(
                        ai.div(active.where(active > 0.0)).fillna(0.0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _period_metrics(
    result: BacktestResult,
    start: str,
    end: str | None,
) -> dict[str, Any]:
    returns = result.returns.loc[start:end] if end else result.returns.loc[start:]
    returns = returns.dropna()
    if returns.empty:
        raise ValueError(f"No returns for period {start} through {end}.")
    equity = 100_000.0 * (1.0 + returns).cumprod()
    metrics = calculate_metrics(
        result.name,
        returns,
        equity,
        result.turnover.reindex(returns.index),
        result.transaction_costs.reindex(returns.index),
    )
    return {
        "start": returns.index.min(),
        "end": returns.index.max(),
        "cagr": metrics.cagr,
        "sharpe": metrics.sharpe,
        "calmar": metrics.calmar,
        "max_drawdown": metrics.max_drawdown,
        "average_turnover": metrics.average_turnover,
    }


def _execution_stress(
    config: BotConfig,
    prices: pd.DataFrame,
    strategies: dict[str, StrategyConfig],
) -> pd.DataFrame:
    profiles = {
        "wednesday_lag1_5bps": config.execution.model_copy(
            update={"rebalance": "W-WED", "signal_lag_days": 1, "transaction_cost_bps": 5.0}
        ),
        "wednesday_lag2_10bps": config.execution.model_copy(
            update={"rebalance": "W-WED", "signal_lag_days": 2, "transaction_cost_bps": 10.0}
        ),
        "monday_lag1_10bps": config.execution.model_copy(
            update={"rebalance": "W-MON", "signal_lag_days": 1, "transaction_cost_bps": 10.0}
        ),
        "friday_lag2_10bps": config.execution.model_copy(
            update={"rebalance": "W-FRI", "signal_lag_days": 2, "transaction_cost_bps": 10.0}
        ),
        "daily_lag2_10bps": config.execution.model_copy(
            update={"rebalance": "D", "signal_lag_days": 2, "transaction_cost_bps": 10.0}
        ),
        "wednesday_lag2_25bps": config.execution.model_copy(
            update={"rebalance": "W-WED", "signal_lag_days": 2, "transaction_cost_bps": 25.0}
        ),
    }
    rows: list[dict[str, object]] = []
    for name, strategy in strategies.items():
        for profile, execution in profiles.items():
            result = _run_strategy(name, strategy, prices, execution)
            metrics = _period_metrics(result, "2005-01-01", None)
            rows.append({"candidate": name, "execution_profile": profile, **metrics})
    return pd.DataFrame(rows)


def _current_allocations(
    results: dict[str, BacktestResult],
    candidate_lookup: dict[str, TransitionCandidate],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        candidate = candidate_lookup.get(name)
        diversifiers = candidate.strategy.risk_repair_ai_diversifier_tickers if candidate else []
        executed = _allocation_summary(result.weights.iloc[-1], diversifiers)
        target = _allocation_summary(result.target_weights.iloc[-1], diversifiers)
        rows.append(
            {
                "candidate": name,
                "market_date": result.weights.index[-1],
                **{f"executed_{key}": value for key, value in executed.items()},
                **{f"target_{key}": value for key, value in target.items()},
            }
        )
    return pd.DataFrame(rows)


def _allocation_summary(
    weights: pd.Series,
    diversifiers: list[str],
) -> dict[str, float | str]:
    weights = weights.clip(lower=0.0)
    ai_columns = [column for column in weights.index if column in AI_GROWTH_TICKERS]
    active = float(weights.drop(labels=["BIL"], errors="ignore").sum())
    ai = float(weights.reindex(ai_columns).fillna(0.0).sum())
    replacement = float(weights.reindex(diversifiers).fillna(0.0).sum())
    positive = weights[weights > 0.005].sort_values(ascending=False)
    return {
        "ai_weight": ai,
        "active_risk_weight": active,
        "ai_share_of_risk": ai / active if active > 0.0 else 0.0,
        "replacement_weight": replacement,
        "bil_weight": float(weights.get("BIL", 0.0)),
        "unallocated_weight": max(0.0, 1.0 - float(weights.sum())),
        "top_positions": ", ".join(
            f"{ticker} {weight:.0%}" for ticker, weight in positive.items()
        ),
        "static_ai_crash_40_loss": -0.40 * ai - 0.12 * replacement,
    }


def _render_report(
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    execution_stress: pd.DataFrame,
    current_allocations: pd.DataFrame,
    *,
    data_end: pd.Timestamp,
) -> str:
    reference = metrics[metrics["candidate"].eq("primary_reference")].set_index("period")
    lines = [
        "# I111 Native Leadership Transition",
        "",
        f"Evidence through {data_end.date()}; 48 fixed-grid native variants.",
        "",
        "Selection uses only 2005-2012 development and 2013-2018 validation ranks. The 2019-2022 holdout, 2023+ period, and current break are diagnostic only.",
        "",
        "## Reference",
        "",
        f"Primary full CAGR {reference.loc['full', 'cagr']:.1%}, max drawdown {reference.loc['full', 'max_drawdown']:.1%}; 2023+ CAGR {reference.loc['post_selection', 'cagr']:.1%}.",
        "",
        "## Training-Selected Family Winners",
        "",
        "| family | candidate | holdout CAGR | 2023+ CAGR | full CAGR | max DD | executed AI/risk | target AI/risk | target replacement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, winner in selection.iterrows():
        name = str(winner["candidate"])
        candidate = metrics[metrics["candidate"].eq(name)].set_index("period")
        current = current_allocations[current_allocations["candidate"].eq(name)].iloc[0]
        lines.append(
            f"| {winner['diversifier_family']} | `{name}` | "
            f"{candidate.loc['holdout', 'cagr']:.1%} | "
            f"{candidate.loc['post_selection', 'cagr']:.1%} | "
            f"{candidate.loc['full', 'cagr']:.1%} | "
            f"{candidate.loc['full', 'max_drawdown']:.1%} | "
            f"{current['executed_ai_share_of_risk']:.1%} | "
            f"{current['target_ai_share_of_risk']:.1%} | "
            f"{current['target_replacement_weight']:.1%} |"
        )
    lines.extend(["", "## Current Allocation", ""])
    for _, row in current_allocations.iterrows():
        lines.append(
            f"- `{row['candidate']}`: executed {row['executed_top_positions']} "
            f"(AI/risk {row['executed_ai_share_of_risk']:.1%}); latest target "
            f"{row['target_top_positions']} "
            f"(AI/risk {row['target_ai_share_of_risk']:.1%}, "
            f"replacement {row['target_replacement_weight']:.1%}, "
            f"static AI-crash loss {row['target_static_ai_crash_40_loss']:.1%})."
        )
    lines.extend(["", "## Execution Stress", ""])
    for name, group in execution_stress.groupby("candidate", sort=False):
        lines.append(
            f"- `{name}`: median CAGR {group['cagr'].median():.1%}; "
            f"worst max drawdown {group['max_drawdown'].min():.1%}; "
            f"25 bps CAGR {group.loc[group['execution_profile'].eq('wednesday_lag2_25bps'), 'cagr'].iloc[0]:.1%}."
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This changes the native target construction, not a post-strategy overlay.",
            "- AI reductions can be redirected only to positive-momentum replacement assets; otherwise the excess remains defensive.",
            "- Per-asset caps apply to both AI and replacement assets.",
            "- The study cannot replay 2000-2002 because the complete executable universe does not exist point-in-time that far back.",
            "- No candidate should replace the operating primary solely from retrospective evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test native i111 leadership transitions.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_i111_leadership_transition(
        load_config(args.config),
        output_dir=args.output,
        refresh_data=args.refresh_data,
    )
    print(result.report_path)


if __name__ == "__main__":
    main()
