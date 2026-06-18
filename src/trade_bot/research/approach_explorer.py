from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trade_bot.config import BotConfig, StrategyConfig
from trade_bot.DEFAULT import DEFAULT_EXPERIMENTS_DIR
from trade_bot.portfolio.risk import current_positions
from trade_bot.strategies.momentum import build_strategy_weights


def build_approach_catalog(
    config: BotConfig,
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENTS_DIR,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, strategy in config.strategies.items():
        rows.append(
            {
                "approach_id": f"baseline::{name}",
                "label": f"baseline | {name}",
                "source": "baseline",
                "iteration": pd.NA,
                "strategy": name,
                "phase": "configured",
                "family": "baseline",
                "role": "configured_strategy",
                "parent": "",
                "promotion_decision": "configured",
                "promotion_score": pd.NA,
                "hypothesis": "Configured baseline strategy currently included in the main run.",
                "strategy_json": json.dumps(strategy.model_dump(mode="json"), sort_keys=True),
            }
        )

    experiment_candidates = load_experiment_candidates(experiment_root)
    if not experiment_candidates.empty:
        rows.extend(experiment_candidates.to_dict(orient="records"))

    return pd.DataFrame(rows)


def load_experiment_candidates(root: str | Path = DEFAULT_EXPERIMENTS_DIR) -> pd.DataFrame:
    experiment_root = Path(root)
    if not experiment_root.exists():
        return pd.DataFrame()

    candidate_frames = []
    for manifest_path in sorted(experiment_root.glob("iteration_*/candidates.csv")):
        frame = pd.read_csv(manifest_path)
        iteration = _iteration_from_path(manifest_path)
        frame.insert(0, "iteration", iteration)
        candidate_frames.append(frame)
    if not candidate_frames:
        return pd.DataFrame()

    candidates = pd.concat(candidate_frames, ignore_index=True)
    scorecards = _load_experiment_scorecards(experiment_root)
    if not scorecards.empty:
        scorecard_columns = [
            "iteration",
            "strategy",
            "promotion_decision",
            "promotion_score",
            "cagr",
            "sharpe",
            "max_drawdown",
            "calmar",
            "excess_cagr_vs_spy",
            "excess_cagr_vs_qqq",
            "drawdown_improvement_vs_spy",
            "drawdown_improvement_vs_qqq",
            "worst_1y_cagr",
            "worst_3y_cagr",
            "positive_1y_window_rate",
        ]
        available_columns = [column for column in scorecard_columns if column in scorecards]
        candidates = candidates.merge(
            scorecards[available_columns],
            on=["iteration", "strategy"],
            how="left",
            suffixes=("", "_scorecard"),
        )

    candidates["source"] = "experiment"
    candidates["approach_id"] = candidates.apply(
        lambda row: f"experiment::{int(row['iteration']):02d}::{row['strategy']}",
        axis=1,
    )
    candidates["label"] = candidates.apply(
        lambda row: (
            f"experiment {int(row['iteration']):02d} | {row['strategy']} "
            f"| {row.get('promotion_decision', 'unscored')}"
        ),
        axis=1,
    )
    for column, default in {
        "phase": "unknown",
        "family": "unknown",
        "role": "unknown",
        "parent": "",
        "hypothesis": "",
        "promotion_decision": "unscored",
    }.items():
        if column not in candidates:
            candidates[column] = default
    return candidates


def strategy_from_catalog_row(row: pd.Series) -> StrategyConfig:
    raw = row.get("strategy_json")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Approach row does not contain a strategy_json payload.")
    return StrategyConfig.model_validate(json.loads(raw))


def build_approach_mechanics(strategy: StrategyConfig, config: BotConfig) -> pd.DataFrame:
    rows = [
        _mechanic("Strategy type", strategy.type, _strategy_type_explanation(strategy)),
        _mechanic(
            "Tradable universe",
            f"{len(strategy.tickers)} assets",
            ", ".join(strategy.tickers),
        ),
        _mechanic(
            "Decision cadence",
            config.execution.rebalance,
            "Signals are converted to target weights on this rebalance cadence.",
        ),
        _mechanic(
            "Execution lag",
            f"{config.execution.signal_lag_days} session(s)",
            "Backtests assume trades happen after signals are known, not at the same close.",
        ),
        _mechanic(
            "Transaction cost",
            f"{config.execution.transaction_cost_bps:.1f} bps turnover cost",
            "Every weight change pays this cost in the backtest.",
        ),
    ]
    if strategy.type == "absolute_momentum":
        rows.append(
            _mechanic(
                "Trend filter",
                f"{strategy.moving_average_days}-day moving average",
                "Risk assets are held only when price is above its moving average.",
            )
        )
    if strategy.type in {"relative_momentum", "dual_momentum"}:
        rows.extend(
            [
                _mechanic(
                    "Momentum lookback",
                    f"{strategy.lookback_days} trading days",
                    "Historical return window used to compare candidate assets.",
                ),
                _mechanic(
                    "Skip window",
                    f"{strategy.skip_days} trading days",
                    "Recent days excluded from the momentum calculation to reduce short-term reversal noise.",
                ),
                _mechanic(
                    "Number selected",
                    f"Top {strategy.top_n}",
                    "Only the highest-ranked assets are eligible for risk exposure.",
                ),
                _mechanic(
                    "Ranking metric",
                    strategy.ranking_metric,
                    _ranking_explanation(strategy.ranking_metric),
                ),
                _mechanic(
                    "Weighting method",
                    strategy.weighting,
                    _weighting_explanation(strategy.weighting),
                ),
                _mechanic(
                    "Volatility lookback",
                    f"{strategy.volatility_lookback_days} trading days",
                    "Used when ranking or sizing depends on realized volatility.",
                ),
            ]
        )
    if strategy.type == "dual_momentum":
        rows.append(
            _mechanic(
                "Absolute return hurdle",
                f"{strategy.min_return:.2%}",
                "Selected assets must also clear this return threshold or capital can move to the defensive asset.",
            )
        )
    if strategy.trend_filter_days:
        rows.append(
            _mechanic(
                "Selection trend confirmation",
                f"{strategy.trend_filter_days}-day moving average",
                "Selected risk assets must remain above this trend filter.",
            )
        )
    if strategy.max_asset_weight:
        rows.append(
            _mechanic(
                "Single-asset cap",
                f"{strategy.max_asset_weight:.0%}",
                "Any excess weight is moved to the defensive asset when one is configured.",
            )
        )
    rows.append(
        _mechanic(
            "Defensive asset",
            strategy.defensive_ticker or "none",
            "Destination for capital when the strategy has no eligible risk signal or capped residual weight.",
        )
    )
    if strategy.volatility_target:
        rows.append(
            _mechanic(
                "Volatility target",
                f"{strategy.volatility_target.annualized_volatility:.0%} annualized",
                (
                    f"Exposure is scaled from lagged {strategy.volatility_target.lookback_days}-day "
                    f"realized volatility and capped at {strategy.volatility_target.max_leverage:.1f}x."
                ),
            )
        )
    if strategy.drawdown_control:
        rows.append(
            _mechanic(
                "Drawdown control",
                f"{strategy.drawdown_control.max_drawdown:.0%} trigger",
                (
                    f"Lagged strategy drawdown over {strategy.drawdown_control.equity_lookback_days} "
                    f"days scales risk to {strategy.drawdown_control.risk_multiplier:.0%}."
                ),
            )
        )
    return pd.DataFrame(rows)


def build_approach_steps(strategy: StrategyConfig) -> pd.DataFrame:
    if strategy.type == "buy_hold":
        steps = [
            "Assign equal target weights to the configured ticker list.",
            "Rebalance on the execution cadence.",
            "Do not use trend, macro, scenario, or defensive off-ramp logic inside this strategy.",
        ]
    elif strategy.type == "absolute_momentum":
        steps = [
            f"Compute each asset's {strategy.moving_average_days}-day moving average.",
            "Hold assets whose price is above trend and split capital equally across active assets.",
            f"If no asset is above trend, hold {strategy.defensive_ticker or 'cash/no position'}.",
        ]
    else:
        steps = [
            f"Compute {strategy.lookback_days}-day returns after skipping the most recent {strategy.skip_days} trading days.",
            f"Rank assets by {strategy.ranking_metric}.",
            f"Keep only the top {strategy.top_n} ranked assets.",
        ]
        if strategy.type == "dual_momentum":
            steps.append(
                f"Drop selected assets that fail the {strategy.min_return:.2%} absolute return hurdle."
            )
        if strategy.trend_filter_days:
            steps.append(
                f"Drop selected assets below their {strategy.trend_filter_days}-day moving average."
            )
        steps.extend(
            [
                f"Size surviving assets with {strategy.weighting} weights.",
                "Clip long-only weights so total risk exposure never exceeds 100%.",
            ]
        )
        if strategy.defensive_ticker:
            steps.append(f"Move unallocated or no-signal capital into {strategy.defensive_ticker}.")
    if strategy.volatility_target:
        steps.append("Apply lagged volatility-target scaling after target weights are formed.")
    if strategy.drawdown_control:
        steps.append("Apply lagged drawdown-control scaling after target weights are formed.")
    return pd.DataFrame(
        [{"step": index + 1, "detail": detail} for index, detail in enumerate(steps)]
    )


def build_approach_risk_notes(strategy: StrategyConfig, row: pd.Series) -> pd.DataFrame:
    notes = [
        {
            "topic": "Execution realism",
            "why_it_matters": "The approach is only tradable if signals can be reviewed and executed after they are known.",
            "watch_item": "Signal lag, weekly rebalance cadence, turnover, and stale-price checks.",
        },
        {
            "topic": "Regime transition risk",
            "why_it_matters": "Momentum and trend systems can be late when leadership changes quickly.",
            "watch_item": "Worst 1-year and 3-year rolling windows, scenario risk, breadth, credit, and news phase.",
        },
    ]
    if strategy.type in {"relative_momentum", "dual_momentum"}:
        notes.append(
            {
                "topic": "Whipsaw",
                "why_it_matters": "Repeated rotations can lose money when markets chop without persistent leadership.",
                "watch_item": "Turnover, false breakouts, and whether top holdings change every rebalance.",
            }
        )
    if "QQQ" in strategy.tickers or any(
        ticker in strategy.tickers for ticker in ["SMH", "SOXX", "NVDA"]
    ):
        notes.append(
            {
                "topic": "AI/concentration dependence",
                "why_it_matters": "Recent outperformance can reflect a narrow historical period that may not persist.",
                "watch_item": "AI-unit-economics events, semis versus broad market, and QQQ/RSP breadth.",
            }
        )
    if strategy.defensive_ticker:
        notes.append(
            {
                "topic": "Defensive destination",
                "why_it_matters": "The off-ramp only helps if the defensive asset behaves as expected when risk sells off.",
                "watch_item": f"{strategy.defensive_ticker} liquidity, yield behavior, and whether duration/credit exposure is intended.",
            }
        )
    if row.get("promotion_decision") in {"reject_left_tail", "reject_regime_fragility"}:
        notes.append(
            {
                "topic": "Experiment warning",
                "why_it_matters": "This approach was rejected by current research triage.",
                "watch_item": str(row.get("promotion_decision")),
            }
        )
    return pd.DataFrame(notes)


def build_latest_approach_weights(prices: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [*strategy.tickers, *([strategy.defensive_ticker] if strategy.defensive_ticker else [])]
        )
    )
    available_columns = [column for column in columns if column in prices.columns]
    missing_columns = sorted(set(columns) - set(available_columns))
    available_risk_tickers = [ticker for ticker in strategy.tickers if ticker in available_columns]
    if not available_risk_tickers:
        return pd.DataFrame(
            [
                {
                    "ticker": "n/a",
                    "weight": 0.0,
                    "note": "No strategy tickers are available in loaded prices.",
                }
            ]
        )
    strategy_prices = prices[available_columns].dropna(how="all")
    strategy_for_prices = StrategyConfig.model_validate(
        {
            **strategy.model_dump(mode="json"),
            "tickers": available_risk_tickers,
            "defensive_ticker": (
                strategy.defensive_ticker
                if strategy.defensive_ticker in available_columns
                else None
            ),
        }
    )
    weights = build_strategy_weights(strategy_prices, strategy_for_prices)
    positions = current_positions(weights, top_n=20)
    frame = pd.DataFrame(
        [
            {"ticker": ticker, "weight": float(weight), "note": ""}
            for ticker, weight in positions.items()
        ]
    )
    if frame.empty:
        frame = pd.DataFrame(
            [{"ticker": "none", "weight": 0.0, "note": "No current active position."}]
        )
    if missing_columns:
        frame.loc[len(frame)] = {
            "ticker": "missing",
            "weight": 0.0,
            "note": ", ".join(missing_columns),
        }
    return frame


def approach_scorecard_row(row: pd.Series) -> pd.DataFrame:
    metric_columns = [
        "promotion_decision",
        "promotion_score",
        "cagr",
        "sharpe",
        "max_drawdown",
        "calmar",
        "excess_cagr_vs_spy",
        "excess_cagr_vs_qqq",
        "drawdown_improvement_vs_spy",
        "drawdown_improvement_vs_qqq",
        "worst_1y_cagr",
        "worst_3y_cagr",
        "positive_1y_window_rate",
    ]
    return pd.DataFrame(
        [
            {
                column: row[column]
                for column in metric_columns
                if column in row and pd.notna(row[column])
            }
        ]
    )


def _load_experiment_scorecards(root: Path) -> pd.DataFrame:
    frames = []
    for scorecard_path in sorted(root.glob("iteration_*/scorecard.csv")):
        frame = pd.read_csv(scorecard_path)
        frame.insert(0, "iteration", _iteration_from_path(scorecard_path))
        if "name" in frame.columns and "strategy" not in frame.columns:
            frame = frame.rename(columns={"name": "strategy"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _iteration_from_path(path: Path) -> int:
    try:
        return int(path.parent.name.split("_")[-1])
    except (IndexError, ValueError):
        return -1


def _mechanic(component: str, setting: str, interpretation: str) -> dict[str, str]:
    return {
        "component": component,
        "setting": setting,
        "interpretation": interpretation,
    }


def _strategy_type_explanation(strategy: StrategyConfig) -> str:
    explanations = {
        "buy_hold": "Static long-only exposure to the configured assets.",
        "absolute_momentum": "Trend-following system that exits risk assets when their own trend breaks.",
        "relative_momentum": "Cross-sectional rotation into the strongest assets, without an absolute return hurdle.",
        "dual_momentum": "Cross-sectional rotation plus an absolute momentum hurdle before taking risk.",
    }
    return explanations[strategy.type]


def _ranking_explanation(metric: str) -> str:
    explanations = {
        "return": "Ranks by raw lookback return.",
        "risk_adjusted_return": "Ranks by lookback return divided by realized volatility.",
        "return_trend_quality": "Ranks by return with a trend-quality boost from price versus moving average.",
    }
    return explanations.get(metric, metric)


def _weighting_explanation(weighting: str) -> str:
    explanations = {
        "equal": "Splits capital equally across selected assets.",
        "inverse_volatility": "Allocates more to lower-volatility selected assets.",
        "momentum_score": "Allocates more to stronger positive momentum scores.",
        "risk_adjusted_score": "Allocates more to stronger risk-adjusted scores.",
    }
    return explanations.get(weighting, weighting)
