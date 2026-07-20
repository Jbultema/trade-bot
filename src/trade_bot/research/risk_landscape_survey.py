from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.backtest.metrics import PerformanceMetrics, calculate_metrics
from trade_bot.config import BotConfig, configured_tickers
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import daily_returns, drawdown, moving_average, rolling_drawdown
from trade_bot.research.experiments import (
    _candidate_tickers,
    _load_previous_candidates,
    _load_previous_scorecards,
    _strategy_prices,
)
from trade_bot.research.prebreak_hindsight import _safe_float
from trade_bot.research.risk_policy_backtest import (
    _active_experiment_root,
    _run_candidate_backtest,
    _selected_candidates,
)

DEFAULT_LANDSCAPE_OUTPUT_DIR = Path("reports/risk_landscape_survey")
DEFAULT_SIGNAL_RANKING_PATH = Path("reports/prebreak_hindsight/signal_predictiveness_rank.csv")
DEFAULT_DRAWDOWN_THRESHOLD = -0.05
AI_GROWTH_TICKERS = {
    "QQQ",
    "QQQM",
    "SMH",
    "SOXX",
    "IGV",
    "XLK",
    "XLC",
    "VUG",
    "IWF",
    "NVDA",
    "AVGO",
    "MSFT",
    "META",
    "AMZN",
    "PLTR",
}
DEFENSIVE_BASKET = {"BIL": 0.60, "GLD": 0.25, "TLT": 0.15}


@dataclass(frozen=True)
class LandscapeSurveyResult:
    strategy_metrics: pd.DataFrame
    drawdown_episodes: pd.DataFrame
    signal_family_rankings: pd.DataFrame
    architecture_metrics: pd.DataFrame
    architecture_summary: pd.DataFrame
    summary: str


def run_risk_landscape_survey(
    config: BotConfig,
    *,
    iteration: int = 164,
    top_n: int = 8,
    experiment_root: str | Path | None = None,
    signal_ranking_path: str | Path = DEFAULT_SIGNAL_RANKING_PATH,
    output_dir: str | Path = DEFAULT_LANDSCAPE_OUTPUT_DIR,
    refresh_data: bool = False,
) -> LandscapeSurveyResult:
    experiment_root = Path(experiment_root) if experiment_root else _active_experiment_root()
    candidates = _selected_candidates(
        iteration,
        scorecards=_load_previous_scorecards(experiment_root, iteration + 1),
        candidates_manifest=_load_previous_candidates(experiment_root, iteration + 1),
        top_n=top_n,
        experiment_root=experiment_root,
    )
    tickers = sorted(
        set(configured_tickers(config))
        | _candidate_tickers(candidates)
        | set(DEFENSIVE_BASKET)
        | {"SPY", "QQQ", "SMH", "HYG", "LQD"}
    )
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    )
    base_results: dict[str, BacktestResult] = {}
    strategy_metric_rows: list[dict[str, object]] = []
    drawdown_rows: list[dict[str, object]] = []
    architecture_metric_rows: list[dict[str, object]] = []
    architecture_metrics: list[PerformanceMetrics] = []
    for candidate in candidates:
        candidate_prices = _strategy_prices(
            prices,
            candidate.strategy.tickers,
            candidate.strategy.defensive_ticker,
        )
        base_result = _run_candidate_backtest(config, candidate, prices, candidate_prices)
        base_results[candidate.name] = base_result
        base_metrics = calculate_metrics(
            name=base_result.name,
            returns=base_result.returns,
            equity=base_result.equity,
            turnover=base_result.turnover,
            transaction_costs=base_result.transaction_costs,
        )
        strategy_metric_rows.append(
            _strategy_metric_row(candidate.name, candidate.family, candidate.phase, base_metrics)
        )
        drawdown_rows.extend(
            _drawdown_episode_rows(
                strategy=candidate.name,
                family=candidate.family,
                result=base_result,
                prices=candidate_prices,
            )
        )
        for variant_name, variant_result in _architecture_variant_results(
            base_result,
            prices,
            transaction_cost_bps=config.execution.transaction_cost_bps,
        ).items():
            variant_metrics = calculate_metrics(
                name=variant_result.name,
                returns=variant_result.returns,
                equity=variant_result.equity,
                turnover=variant_result.turnover,
                transaction_costs=variant_result.transaction_costs,
            )
            architecture_metrics.append(variant_metrics)
            architecture_metric_rows.append(
                _architecture_metric_row(
                    strategy=candidate.name,
                    family=candidate.family,
                    variant_name=variant_name,
                    base_result=base_result,
                    base_metrics=base_metrics,
                    variant_metrics=variant_metrics,
                    variant_result=variant_result,
                )
            )
    strategy_metrics = pd.DataFrame(strategy_metric_rows)
    drawdown_episodes = (
        pd.DataFrame(drawdown_rows)
        .sort_values(["drawdown_depth", "strategy"], ascending=[True, True])
        .reset_index(drop=True)
    )
    signal_family_rankings = build_signal_family_rankings(signal_ranking_path)
    architecture_metrics_frame = pd.DataFrame(architecture_metric_rows)
    architecture_summary = summarize_architecture_metrics(architecture_metrics_frame)
    summary = build_landscape_summary(
        strategy_metrics,
        drawdown_episodes,
        signal_family_rankings,
        architecture_summary,
    )
    result = LandscapeSurveyResult(
        strategy_metrics=strategy_metrics,
        drawdown_episodes=drawdown_episodes,
        signal_family_rankings=signal_family_rankings,
        architecture_metrics=architecture_metrics_frame,
        architecture_summary=architecture_summary,
        summary=summary,
    )
    write_landscape_outputs(result, output_dir=output_dir)
    return result


def build_signal_family_rankings(
    signal_ranking_path: str | Path = DEFAULT_SIGNAL_RANKING_PATH,
) -> pd.DataFrame:
    path = Path(signal_ranking_path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "signal" not in frame:
        return pd.DataFrame()
    frame["signal_family"] = frame["signal"].astype(str).map(_signal_family)
    score_columns = [
        column
        for column in [
            "predictive_score",
            "absolute_spearman",
            "event_auc_edge",
            "high_minus_low_break_severity",
        ]
        if column in frame
    ]
    for column in score_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for family, group in frame.groupby("signal_family", sort=False):
        ranked = group.sort_values("predictive_score", ascending=False)
        top = ranked.iloc[0]
        rows.append(
            {
                "signal_family": family,
                "signals": len(group),
                "best_signal": top["signal"],
                "best_predictive_score": _safe_float(top.get("predictive_score")),
                "mean_top5_predictive_score": _safe_float(
                    ranked["predictive_score"].head(5).mean()
                ),
                "best_spearman": _safe_float(top.get("spearman_to_break_severity")),
                "best_auc": _safe_float(top.get("event_auc")),
                "best_risk_direction": str(top.get("risk_direction", "")),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["mean_top5_predictive_score", "best_predictive_score"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarize_architecture_metrics(architecture_metrics: pd.DataFrame) -> pd.DataFrame:
    if architecture_metrics.empty:
        return pd.DataFrame()
    rows = []
    for variant_name, group in architecture_metrics.groupby("variant_name", sort=False):
        rows.append(
            {
                "variant_name": variant_name,
                "strategies": len(group),
                "median_cagr": pd.to_numeric(group["cagr"], errors="coerce").median(),
                "median_max_drawdown": pd.to_numeric(
                    group["max_drawdown"],
                    errors="coerce",
                ).median(),
                "median_calmar": pd.to_numeric(group["calmar"], errors="coerce").median(),
                "median_delta_cagr": pd.to_numeric(
                    group["delta_cagr_vs_base"],
                    errors="coerce",
                ).median(),
                "median_delta_max_drawdown": pd.to_numeric(
                    group["delta_max_drawdown_vs_base"],
                    errors="coerce",
                ).median(),
                "cagr_win_rate": float((group["delta_cagr_vs_base"] > 0).mean()),
                "drawdown_win_rate": float((group["delta_max_drawdown_vs_base"] > 0).mean()),
                "median_active_day_rate": pd.to_numeric(
                    group["active_day_rate"],
                    errors="coerce",
                ).median(),
                "research_read": _architecture_read(group),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["median_delta_cagr", "median_delta_max_drawdown"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_landscape_summary(
    strategy_metrics: pd.DataFrame,
    drawdown_episodes: pd.DataFrame,
    signal_family_rankings: pd.DataFrame,
    architecture_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Risk Landscape Survey",
        "",
        "This is a broad survey pass before threshold tuning. It maps the actual",
        "drawdown wounds in the current top strategies, ranks pre-break signal",
        "families, and tests coarse architecture variants against the top batch.",
        "",
        "## Top Strategy Baseline",
        "",
    ]
    if strategy_metrics.empty:
        lines.append("- no strategy metrics were available")
    else:
        lines.append(
            "- median CAGR "
            f"{pd.to_numeric(strategy_metrics['cagr'], errors='coerce').median():.2%}; "
            "median max DD "
            f"{pd.to_numeric(strategy_metrics['max_drawdown'], errors='coerce').median():.2%}; "
            "median Calmar "
            f"{pd.to_numeric(strategy_metrics['calmar'], errors='coerce').median():.2f}"
        )
    lines.extend(["", "## Drawdown Genome", ""])
    if drawdown_episodes.empty:
        lines.append("- no drawdown episodes crossed the threshold")
    else:
        worst = drawdown_episodes.head(8)
        for _, row in worst.iterrows():
            lines.append(
                "- "
                f"{row['strategy']} trough {row['trough_date']}: "
                f"{_safe_float(row['drawdown_depth']):.2%}, "
                f"dominant loss {row['dominant_loss_ticker']} "
                f"({row['dominant_loss_family']}); "
                f"avg risk weight {_safe_float(row['average_risk_weight']):.1%}"
            )
    lines.extend(["", "## Signal Family Tournament", ""])
    if signal_family_rankings.empty:
        lines.append("- no signal family ranking was available")
    else:
        for _, row in signal_family_rankings.head(8).iterrows():
            lines.append(
                "- "
                f"{row['signal_family']}: top5 score "
                f"{_safe_float(row['mean_top5_predictive_score']):.2f}; "
                f"best {row['best_signal']} "
                f"({_safe_float(row['best_predictive_score']):.2f})"
            )
    lines.extend(["", "## Architecture Survey", ""])
    if architecture_summary.empty:
        lines.append("- no architecture variants were available")
    else:
        for _, row in architecture_summary.iterrows():
            lines.append(
                "- "
                f"{row['variant_name']}: median CAGR "
                f"{_safe_float(row['median_cagr']):.2%}, max DD "
                f"{_safe_float(row['median_max_drawdown']):.2%}, "
                f"delta CAGR {_safe_float(row['median_delta_cagr']):+.2%}, "
                f"DD delta {_safe_float(row['median_delta_max_drawdown']):+.2%}; "
                f"{row['research_read']}"
            )
    lines.extend(
        [
            "",
            "## Next Research Direction",
            "",
            "- Stop tuning sparse pre-break snapshot overlays for now; they did not touch",
            "  the actual max-DD wound in the current top strategies.",
            "- Prioritize strategy-native daily architecture tests: drawdown self-defense,",
            "  AI concentration caps, and defensive-destination design.",
            "- Treat signal families as monitor inputs first, then promote only the ones",
            "  that improve full-history and regime-specific backtests.",
        ]
    )
    return "\n".join(lines)


def write_landscape_outputs(
    result: LandscapeSurveyResult,
    *,
    output_dir: str | Path = DEFAULT_LANDSCAPE_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.strategy_metrics.to_csv(output_path / "strategy_metrics.csv", index=False)
    result.drawdown_episodes.to_csv(output_path / "drawdown_episodes.csv", index=False)
    result.signal_family_rankings.to_csv(output_path / "signal_family_rankings.csv", index=False)
    result.architecture_metrics.to_csv(output_path / "architecture_metrics.csv", index=False)
    result.architecture_summary.to_csv(output_path / "architecture_summary.csv", index=False)
    (output_path / "summary.md").write_text(result.summary, encoding="utf-8")


def _architecture_variant_results(
    base_result: BacktestResult,
    prices: pd.DataFrame,
    *,
    transaction_cost_bps: float,
) -> dict[str, BacktestResult]:
    aligned_prices = prices.reindex(base_result.weights.index).ffill()
    variants = {
        "permanent_15pct_defensive_basket": _apply_basket_scale(
            base_result,
            aligned_prices,
            pd.Series(0.85, index=base_result.weights.index),
            transaction_cost_bps=transaction_cost_bps,
            name=f"{base_result.name}__permanent_15pct_defensive_basket",
        ),
        "qqq_spy_200d_confirmed_throttle": _apply_basket_scale(
            base_result,
            aligned_prices,
            _confirmed_trend_budget(aligned_prices),
            transaction_cost_bps=transaction_cost_bps,
            name=f"{base_result.name}__qqq_spy_200d_confirmed_throttle",
        ),
        "strategy_dd8_confirmed_throttle": _apply_basket_scale(
            base_result,
            aligned_prices,
            _strategy_drawdown_budget(base_result, aligned_prices),
            transaction_cost_bps=transaction_cost_bps,
            name=f"{base_result.name}__strategy_dd8_confirmed_throttle",
        ),
        "ai_concentration_cap_under_stress": _apply_ai_concentration_cap(
            base_result,
            aligned_prices,
            transaction_cost_bps=transaction_cost_bps,
            name=f"{base_result.name}__ai_concentration_cap_under_stress",
        ),
        "hybrid_confirmed_stress_basket": _apply_basket_scale(
            base_result,
            aligned_prices,
            pd.concat(
                [
                    _confirmed_trend_budget(aligned_prices),
                    _strategy_drawdown_budget(base_result, aligned_prices),
                ],
                axis=1,
            ).min(axis=1),
            transaction_cost_bps=transaction_cost_bps,
            name=f"{base_result.name}__hybrid_confirmed_stress_basket",
        ),
    }
    return variants


def _drawdown_episode_rows(
    *,
    strategy: str,
    family: str,
    result: BacktestResult,
    prices: pd.DataFrame,
    threshold: float = DEFAULT_DRAWDOWN_THRESHOLD,
) -> list[dict[str, object]]:
    equity = result.equity.dropna()
    dd = drawdown(equity)
    rows: list[dict[str, object]] = []
    in_episode = False
    start: pd.Timestamp | None = None
    for date, value in dd.items():
        date = pd.Timestamp(date)
        if not in_episode and value <= threshold:
            in_episode = True
            start = date
        recovered = value >= -0.01
        if in_episode and recovered and start is not None and date > start:
            rows.append(
                _drawdown_episode_row(
                    strategy=strategy,
                    family=family,
                    result=result,
                    prices=prices,
                    start=start,
                    end=date,
                )
            )
            in_episode = False
            start = None
    if in_episode and start is not None:
        rows.append(
            _drawdown_episode_row(
                strategy=strategy,
                family=family,
                result=result,
                prices=prices,
                start=start,
                end=pd.Timestamp(dd.index[-1]),
            )
        )
    return rows


def _drawdown_episode_row(
    *,
    strategy: str,
    family: str,
    result: BacktestResult,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    equity = result.equity.dropna()
    dd = drawdown(equity)
    window_dd = dd.loc[start:end]
    trough_date = pd.Timestamp(window_dd.idxmin())
    peak_date = pd.Timestamp(equity.loc[:start].idxmax())
    pre_trough = result.weights.reindex(prices.index).ffill().loc[peak_date:trough_date]
    asset_returns = daily_returns(prices).reindex(pre_trough.index).fillna(0.0)
    contributions = (pre_trough * asset_returns).sum().sort_values()
    dominant_ticker = str(contributions.index[0]) if not contributions.empty else ""
    risk_columns = [column for column in pre_trough.columns if column != "BIL"]
    defensive_weight = pre_trough.get("BIL", pd.Series(0.0, index=pre_trough.index))
    return {
        "strategy": strategy,
        "family": family,
        "episode_start": str(start.date()),
        "peak_date": str(peak_date.date()),
        "trough_date": str(trough_date.date()),
        "recovery_or_end_date": str(end.date()),
        "drawdown_depth": _safe_float(window_dd.min()),
        "calendar_days_peak_to_trough": int((trough_date - peak_date).days),
        "calendar_days_to_recovery_or_end": int((end - start).days),
        "dominant_loss_ticker": dominant_ticker,
        "dominant_loss_family": _ticker_family(dominant_ticker),
        "dominant_loss_contribution": _safe_float(
            contributions.iloc[0] if not contributions.empty else 0.0
        ),
        "average_risk_weight": _safe_float(pre_trough[risk_columns].sum(axis=1).mean()),
        "average_defensive_weight": _safe_float(defensive_weight.mean()),
        "average_ai_growth_weight": _safe_float(
            pre_trough[[column for column in pre_trough.columns if column in AI_GROWTH_TICKERS]]
            .sum(axis=1)
            .mean()
        ),
    }


def _apply_basket_scale(
    result: BacktestResult,
    prices: pd.DataFrame,
    budget: pd.Series,
    *,
    transaction_cost_bps: float,
    name: str,
) -> BacktestResult:
    weights = result.weights.reindex(prices.index).ffill().fillna(0.0)
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)
    budget = budget.reindex(weights.index).fillna(1.0).clip(0.0, 1.0)
    adjusted = weights.copy()
    basket_cols = [ticker for ticker in DEFENSIVE_BASKET if ticker in adjusted.columns]
    risk_columns = [column for column in adjusted.columns if column not in basket_cols]
    risk_weight = adjusted[risk_columns].sum(axis=1)
    released = risk_weight * (1.0 - budget)
    adjusted.loc[:, risk_columns] = adjusted[risk_columns].mul(budget, axis=0)
    for ticker, basket_weight in DEFENSIVE_BASKET.items():
        if ticker in adjusted:
            adjusted[ticker] = adjusted[ticker] + released * basket_weight
    return _result_from_weights(result, prices, adjusted, transaction_cost_bps, name)


def _apply_ai_concentration_cap(
    result: BacktestResult,
    prices: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    name: str,
    cap: float = 0.55,
) -> BacktestResult:
    weights = result.weights.reindex(prices.index).ffill().fillna(0.0)
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)
    ai_columns = [column for column in weights.columns if column in AI_GROWTH_TICKERS]
    stress = _ai_stress_signal(prices)
    adjusted = weights.copy()
    if ai_columns:
        ai_weight = adjusted[ai_columns].sum(axis=1)
        excess = (ai_weight - cap).clip(lower=0.0)
        reduction = pd.Series(0.0, index=adjusted.index)
        reduction.loc[stress] = excess.loc[stress]
        denominator = ai_weight.where(ai_weight.ne(0.0))
        scale = ((ai_weight - reduction) / denominator).fillna(1.0).astype(float)
        adjusted.loc[:, ai_columns] = adjusted[ai_columns].mul(scale, axis=0)
        if "BIL" in adjusted:
            adjusted["BIL"] = adjusted["BIL"] + reduction
    return _result_from_weights(result, prices, adjusted, transaction_cost_bps, name)


def _result_from_weights(
    base_result: BacktestResult,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float,
    name: str,
) -> BacktestResult:
    asset_returns = daily_returns(prices).reindex(weights.index).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    transaction_costs = turnover * transaction_cost_bps / 10000.0
    gross_returns = (weights * asset_returns).sum(axis=1)
    net_returns = gross_returns - transaction_costs
    initial = base_result.equity.iloc[0] / (1.0 + base_result.returns.iloc[0])
    equity = initial * (1.0 + net_returns).cumprod()
    return BacktestResult(
        name=name,
        equity=equity.rename(name),
        returns=net_returns.rename(name),
        gross_returns=gross_returns.rename(name),
        weights=weights,
        target_weights=base_result.target_weights,
        turnover=turnover.rename(name),
        transaction_costs=transaction_costs.rename(name),
    )


def _confirmed_trend_budget(prices: pd.DataFrame) -> pd.Series:
    qqq_below = prices["QQQ"] < moving_average(prices[["QQQ"]], 200)["QQQ"]
    spy_below = prices["SPY"] < moving_average(prices[["SPY"]], 200)["SPY"]
    credit_weak = _credit_weak(prices)
    confirmed = qqq_below & (spy_below | credit_weak)
    confirmed = confirmed.shift(1, fill_value=False).astype(bool)
    return pd.Series(1.0, index=prices.index).where(~confirmed, 0.70)


def _strategy_drawdown_budget(result: BacktestResult, prices: pd.DataFrame) -> pd.Series:
    strategy_dd = rolling_drawdown(result.equity.reindex(prices.index).ffill(), 126)
    qqq_dd = rolling_drawdown(prices["QQQ"].ffill(), 126)
    confirmed = (strategy_dd <= -0.08) & (qqq_dd <= -0.08)
    confirmed = confirmed.shift(1, fill_value=False).astype(bool)
    return pd.Series(1.0, index=prices.index).where(~confirmed, 0.65)


def _ai_stress_signal(prices: pd.DataFrame) -> pd.Series:
    qqq_dd = rolling_drawdown(prices["QQQ"].ffill(), 126) <= -0.08
    smh_below = prices["SMH"] < moving_average(prices[["SMH"]], 100)["SMH"]
    return (qqq_dd & smh_below).shift(1, fill_value=False).astype(bool)


def _credit_weak(prices: pd.DataFrame) -> pd.Series:
    if not {"HYG", "LQD"}.issubset(prices.columns):
        return pd.Series(False, index=prices.index)
    ratio = prices["HYG"].ffill() / prices["LQD"].ffill()
    return ratio < moving_average(ratio.to_frame("credit"), 100)["credit"]


def _strategy_metric_row(
    strategy: str,
    family: str,
    phase: str,
    metrics: PerformanceMetrics,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "family": family,
        "phase": phase,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe": metrics.sharpe,
        "average_turnover": metrics.average_turnover,
    }


def _architecture_metric_row(
    *,
    strategy: str,
    family: str,
    variant_name: str,
    base_result: BacktestResult,
    base_metrics: PerformanceMetrics,
    variant_metrics: PerformanceMetrics,
    variant_result: BacktestResult,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "family": family,
        "variant_name": variant_name,
        "cagr": variant_metrics.cagr,
        "max_drawdown": variant_metrics.max_drawdown,
        "calmar": variant_metrics.calmar,
        "sharpe": variant_metrics.sharpe,
        "average_turnover": variant_metrics.average_turnover,
        "delta_cagr_vs_base": variant_metrics.cagr - base_metrics.cagr,
        "delta_max_drawdown_vs_base": variant_metrics.max_drawdown - base_metrics.max_drawdown,
        "delta_calmar_vs_base": variant_metrics.calmar - base_metrics.calmar,
        "active_day_rate": _variant_active_day_rate(base_result.weights, variant_result.weights),
    }


def _architecture_read(group: pd.DataFrame) -> str:
    cagr_delta = pd.to_numeric(group["delta_cagr_vs_base"], errors="coerce").median()
    dd_delta = pd.to_numeric(group["delta_max_drawdown_vs_base"], errors="coerce").median()
    if cagr_delta > 0.002 and dd_delta >= -0.005:
        return "promising_growth"
    if dd_delta > 0.015 and cagr_delta > -0.015:
        return "promising_survivability"
    if cagr_delta < -0.015 and dd_delta < 0.005:
        return "reject_drag_without_protection"
    return "mixed_or_diagnostic"


def _variant_active_day_rate(base_weights: pd.DataFrame, variant_weights: pd.DataFrame) -> float:
    columns = sorted(set(base_weights.columns) | set(variant_weights.columns))
    base = base_weights.reindex(variant_weights.index).reindex(columns=columns, fill_value=0.0)
    variant = variant_weights.reindex(columns=columns, fill_value=0.0)
    difference = variant.sub(base, fill_value=0.0).abs().sum(axis=1)
    return float((difference > 0.001).mean())


def _signal_family(signal: str) -> str:
    if signal.startswith("health_"):
        return "market_structure"
    if signal.startswith("cycle_"):
        return "cycle_tracker"
    if signal.startswith("driver_"):
        return "scenario_drivers"
    if signal.startswith("confirmation_"):
        return "confirmation_matrix"
    if signal.startswith("instability_") or "large_move" in signal:
        return "regime_instability"
    if signal.startswith("portfolio_"):
        return "portfolio_stress"
    if signal in {"risk_score", "risk_status_score"}:
        return "risk_status"
    if "probability" in signal:
        return "probability_model"
    if "risk_budget" in signal or "risk_asset_weight" in signal:
        return "trade_bot_behavior"
    return "other"


def _ticker_family(ticker: str) -> str:
    if ticker in AI_GROWTH_TICKERS:
        return "ai_growth"
    if ticker in {"SPY", "VTI", "VT", "RSP", "IWM", "DIA", "MDY"}:
        return "broad_equity"
    if ticker in {"EFA", "EEM", "VEA", "VWO", "VGK", "EWJ", "INDA", "EWZ", "EWC"}:
        return "global_equity"
    if ticker in {"TLT", "IEF", "SHY", "BIL", "USFR", "SGOV"}:
        return "defensive_fixed_income"
    if ticker in {"GLD", "IAU", "DBC", "USO"}:
        return "real_assets"
    if ticker in {"HYG", "LQD", "JNK"}:
        return "credit"
    return "other"
