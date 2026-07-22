from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import calculate_metrics, metrics_frame
from trade_bot.backtest.windows import (
    calendar_year_metrics,
    rolling_window_metrics,
    summarize_walk_forward,
    summarize_windows,
    walk_forward_holdout_metrics,
)
from trade_bot.config import (
    BotConfig,
    DrawdownControlConfig,
    StrategyConfig,
    VolatilityTargetConfig,
    configured_tickers,
    required_strategy_tickers,
)
from trade_bot.data.market_data import load_or_fetch_yahoo_prices
from trade_bot.features.indicators import (
    daily_returns,
    lookback_returns,
    moving_average,
    realized_volatility,
    unusable_required_price_columns,
)
from trade_bot.strategies.momentum import build_strategy_weights

DEFAULT_UPSIDE_CAPTURE_OUTPUT_DIR = Path("reports/upside_capture_lab")
DEFAULT_PRIMARY_STRATEGY = "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145"


@dataclass(frozen=True)
class UpsideCaptureCandidate:
    round_id: int
    name: str
    hypothesis: str
    strategy: StrategyConfig
    overlay: dict[str, Any] | None = None


@dataclass(frozen=True)
class UpsideCaptureLabResult:
    summary: pd.DataFrame
    events: pd.DataFrame
    candidates: tuple[UpsideCaptureCandidate, ...]
    output_dir: Path


def run_upside_capture_lab(
    config: BotConfig,
    *,
    output_dir: str | Path = DEFAULT_UPSIDE_CAPTURE_OUTPUT_DIR,
    primary_strategy: str = DEFAULT_PRIMARY_STRATEGY,
    refresh_data: bool = False,
) -> UpsideCaptureLabResult:
    """Run targeted upside-capture experiments inspired by 42 Macro gaps."""

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    base_strategy = config.strategies.get(primary_strategy)
    if base_strategy is None:
        raise ValueError(f"Primary strategy not found: {primary_strategy}")
    candidates = _candidate_rounds(base_strategy)
    tickers = sorted(set(configured_tickers(config)) | _candidate_tickers(candidates))
    prices = load_or_fetch_yahoo_prices(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        adjusted=config.data.adjusted,
        refresh=refresh_data,
    )
    prices = prices.sort_index().ffill()

    results: dict[str, BacktestResult] = {}
    rows = []
    base_result: BacktestResult | None = None
    for candidate in candidates:
        candidate_prices = _strategy_prices(prices, candidate.strategy)
        weights = build_strategy_weights(candidate_prices, candidate.strategy)
        if candidate.overlay:
            weights = _apply_constructive_overlay(candidate_prices, weights, candidate.overlay)
        result = run_backtest(
            candidate.name,
            candidate_prices,
            weights,
            config.execution,
            volatility_target=candidate.strategy.volatility_target,
            drawdown_control=candidate.strategy.drawdown_control,
        )
        results[candidate.name] = result
        if candidate.name == "r0_i111_current_baseline":
            base_result = result
        metric = calculate_metrics(
            name=result.name,
            returns=result.returns,
            equity=result.equity,
            turnover=result.turnover,
            transaction_costs=result.transaction_costs,
        )
        rows.append(metric)
    if base_result is None:
        raise RuntimeError("Upside capture lab did not produce the baseline result.")

    summary = metrics_frame(rows).reset_index().rename(columns={"name": "candidate"})
    summary = _add_candidate_metadata(summary, candidates)
    event_metrics = _event_metrics_frame(results, prices, base_result)
    summary = summary.merge(event_metrics, on="candidate", how="left")
    window_metrics = rolling_window_metrics(results)
    window_summary = summarize_windows(window_metrics)
    walk_forward_folds = walk_forward_holdout_metrics(results)
    walk_forward_summary = summarize_walk_forward(walk_forward_folds)
    calendar_metrics = calendar_year_metrics(results)
    summary = _add_robustness_metrics(
        summary,
        window_summary=window_summary,
        walk_forward_summary=walk_forward_summary,
        calendar_metrics=calendar_metrics,
        baseline_name="r0_i111_current_baseline",
    )
    summary = _score_candidates(summary)
    summary = summary.sort_values(["round_id", "research_score"], ascending=[True, False])
    events = _event_observations(results, prices, base_result)

    summary.to_csv(report_dir / "upside_capture_summary.csv", index=False)
    events.to_csv(report_dir / "upside_capture_event_observations.csv", index=False)
    window_summary.reset_index().to_csv(
        report_dir / "upside_capture_rolling_windows.csv",
        index=False,
    )
    walk_forward_summary.reset_index().to_csv(
        report_dir / "upside_capture_walk_forward.csv",
        index=False,
    )
    calendar_metrics.to_csv(report_dir / "upside_capture_calendar_years.csv", index=False)
    (report_dir / "upside_capture_findings.md").write_text(
        _findings_markdown(summary, events),
        encoding="utf-8",
    )
    return UpsideCaptureLabResult(
        summary=summary,
        events=events,
        candidates=candidates,
        output_dir=report_dir,
    )


def _candidate_rounds(base: StrategyConfig) -> tuple[UpsideCaptureCandidate, ...]:
    ai_plus_global = [
        "QQQ",
        "SMH",
        "SOXX",
        "IGV",
        "NVDA",
        "AVGO",
        "MSFT",
        "META",
        "AMZN",
        "PLTR",
        "SPY",
        "IWM",
        "RSP",
        "VEA",
        "VGK",
        "EWJ",
        "VWO",
        "GLD",
    ]
    high_beta_plus_breadth = [
        "QQQ",
        "SMH",
        "SOXX",
        "IGV",
        "NVDA",
        "AVGO",
        "MSFT",
        "META",
        "AMZN",
        "PLTR",
        "SPY",
        "IWM",
        "RSP",
        "VEA",
        "GLD",
    ]
    candidates: list[UpsideCaptureCandidate] = [
        UpsideCaptureCandidate(
            round_id=0,
            name="r0_i111_current_baseline",
            hypothesis="Current i111 baseline for measuring incremental upside capture.",
            strategy=base,
        )
    ]

    def clone(**updates: object) -> StrategyConfig:
        data = base.model_dump()
        data.update(updates)
        return StrategyConfig.model_validate(data)

    round_specs = [
        (
            1,
            "threshold relaxation",
            [
                (
                    "r1_lower_min_return",
                    "Lower absolute momentum hurdle to avoid sitting in cash during early runups.",
                    clone(min_return=0.01),
                    None,
                ),
                (
                    "r1_no_trend_filter",
                    "Remove the slower trend filter and let cross-sectional momentum re-enter earlier.",
                    clone(min_return=0.02, trend_filter_days=None),
                    None,
                ),
                (
                    "r1_top5_broader",
                    "Hold one more winner so upside capture is less dependent on a narrow top four.",
                    clone(top_n=5, min_return=0.02, max_asset_weight=0.30),
                    None,
                ),
                (
                    "r1_return_trend_quality",
                    "Reward persistent trend quality instead of pure risk-adjusted return.",
                    clone(ranking_metric="return_trend_quality", min_return=0.02),
                    None,
                ),
                (
                    "r1_higher_vol_target",
                    "Permit slightly more risk when the existing i111 model is already constructive.",
                    clone(
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        )
                    ),
                    None,
                ),
            ],
        ),
        (
            2,
            "constructive floors",
            [
                (
                    "r2_risk_on_floor_50",
                    "Keep at least 50% risk when QQQ/SPY trend and credit breadth are constructive.",
                    clone(min_return=0.02, trend_filter_days=None),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.50,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 4,
                    },
                ),
                (
                    "r2_risk_on_floor_65",
                    "More assertive participation floor in confirmed risk-on tape.",
                    clone(min_return=0.02, trend_filter_days=None),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.65,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 4,
                    },
                ),
                (
                    "r2_credit_breadth_floor",
                    "Only add risk when credit and equal-weight breadth confirm the move.",
                    clone(min_return=0.02, trend_filter_days=None),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.60,
                        "signal": "credit_breadth",
                        "lookback": 42,
                        "top_n": 4,
                    },
                ),
                (
                    "r2_ai_leadership_floor",
                    "Add risk when AI leadership is broadening, mirroring the 42 Macro upside emphasis.",
                    clone(min_return=0.02, trend_filter_days=None),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.60,
                        "signal": "ai_leadership",
                        "lookback": 42,
                        "top_n": 4,
                    },
                ),
                (
                    "r2_recovery_floor",
                    "Accelerate re-entry after drawdown repair but avoid adding during falling knives.",
                    clone(min_return=0.02, trend_filter_days=None),
                    {
                        "mode": "recovery_floor",
                        "floor": 0.55,
                        "signal": "balanced",
                        "lookback": 21,
                        "top_n": 4,
                    },
                ),
            ],
        ),
        (
            3,
            "broader upside pools",
            [
                (
                    "r3_global_floor",
                    "Let international equities participate when the trend is global rather than US-only.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.28,
                        trend_filter_days=None,
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.60,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
                (
                    "r3_global_recovery_floor",
                    "Combine global pool with repair-aware re-entry.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.28,
                        trend_filter_days=None,
                    ),
                    {
                        "mode": "recovery_floor",
                        "floor": 0.60,
                        "signal": "credit_breadth",
                        "lookback": 21,
                        "top_n": 5,
                    },
                ),
                (
                    "r3_high_beta_breadth_floor",
                    "Keep AI beta but require market breadth before raising the floor.",
                    clone(
                        tickers=high_beta_plus_breadth,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.30,
                        trend_filter_days=None,
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.65,
                        "signal": "credit_breadth",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
                (
                    "r3_global_trend_quality",
                    "Broader pool with trend-quality ranking to catch non-US leadership rotations.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.28,
                        ranking_metric="return_trend_quality",
                        trend_filter_days=None,
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.60,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
                (
                    "r3_broader_higher_vol",
                    "Broader pool with slightly higher volatility budget.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.28,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18, lookback_days=21
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.60,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
            ],
        ),
        (
            4,
            "best-of-both-worlds guardrails",
            [
                (
                    "r4_balanced_floor_drawdown_guard",
                    "Risk-on floor with a drawdown guard to avoid giving back left-tail protection.",
                    clone(
                        tickers=high_beta_plus_breadth,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.30,
                        trend_filter_days=None,
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.62,
                        "signal": "credit_breadth",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
                (
                    "r4_recovery_guard",
                    "Recovery accelerator with tighter drawdown throttle.",
                    clone(
                        tickers=high_beta_plus_breadth,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.30,
                        trend_filter_days=None,
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=63,
                            max_drawdown=-0.10,
                            risk_multiplier=0.50,
                        ),
                    ),
                    {
                        "mode": "recovery_floor",
                        "floor": 0.60,
                        "signal": "credit_breadth",
                        "lookback": 21,
                        "top_n": 5,
                    },
                ),
                (
                    "r4_global_floor_guard",
                    "Global participation floor with left-tail guard.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.28,
                        trend_filter_days=None,
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.62,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
                (
                    "r4_ai_leadership_guard",
                    "AI leadership floor, but only with drawdown guardrails.",
                    clone(
                        tickers=high_beta_plus_breadth,
                        top_n=5,
                        min_return=0.01,
                        max_asset_weight=0.30,
                        trend_filter_days=None,
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.60,
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.65,
                        "signal": "ai_leadership",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
                (
                    "r4_measured_vol_floor",
                    "The most pragmatic blend: modest floor, broader pool, normal volatility budget.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=5,
                        min_return=0.015,
                        max_asset_weight=0.28,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17, lookback_days=21
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.58,
                        "signal": "credit_breadth",
                        "lookback": 42,
                        "top_n": 5,
                    },
                ),
            ],
        ),
        (
            5,
            "no-filter parameter tuning",
            [
                (
                    "r5_no_filter_min015",
                    "Keep the no-trend-filter edge but test a slightly easier return hurdle.",
                    clone(min_return=0.015, trend_filter_days=None),
                    None,
                ),
                (
                    "r5_no_filter_min025",
                    "Keep the no-trend-filter edge but require a little more absolute momentum.",
                    clone(min_return=0.025, trend_filter_days=None),
                    None,
                ),
                (
                    "r5_top3_concentrated",
                    "Let the strongest three names carry more upside while retaining the normal vol cap.",
                    clone(top_n=3, min_return=0.02, trend_filter_days=None, max_asset_weight=0.40),
                    None,
                ),
                (
                    "r5_top4_cap40",
                    "Keep top-four selection but allow winners to reach 40% before volatility targeting.",
                    clone(min_return=0.02, trend_filter_days=None, max_asset_weight=0.40),
                    None,
                ),
                (
                    "r5_top5_cap30",
                    "Use a broader five-name basket with tighter per-asset caps to test smoother upside.",
                    clone(top_n=5, min_return=0.02, trend_filter_days=None, max_asset_weight=0.30),
                    None,
                ),
            ],
        ),
        (
            6,
            "signal timing",
            [
                (
                    "r6_lookback42_skip5",
                    "Shorten the momentum lookback to react faster to post-dip upside rotations.",
                    clone(lookback_days=42, skip_days=5, min_return=0.02, trend_filter_days=None),
                    None,
                ),
                (
                    "r6_lookback42_skip0",
                    "Remove the skip lag from the faster lookback to test maximum re-entry speed.",
                    clone(lookback_days=42, skip_days=0, min_return=0.02, trend_filter_days=None),
                    None,
                ),
                (
                    "r6_lookback84_skip5",
                    "Lengthen the lookback modestly to reduce churn while staying faster than quarterly.",
                    clone(lookback_days=84, skip_days=5, min_return=0.02, trend_filter_days=None),
                    None,
                ),
                (
                    "r6_lookback63_skip10",
                    "Keep the current lookback but add a longer skip window to avoid late exhaustion.",
                    clone(lookback_days=63, skip_days=10, min_return=0.02, trend_filter_days=None),
                    None,
                ),
                (
                    "r6_vol42_smoother",
                    "Smooth volatility ranking so temporary vol spikes do not over-penalize winners.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_lookback_days=42,
                    ),
                    None,
                ),
            ],
        ),
        (
            7,
            "higher participation with drawdown brakes",
            [
                (
                    "r7_vol17_guard12",
                    "Raise the volatility target modestly but add a 12% drawdown throttle.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    None,
                ),
                (
                    "r7_vol18_guard12",
                    "Test whether the higher upside from an 18% vol target survives a drawdown brake.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r7_vol17_guard10",
                    "Use an earlier 10% drawdown throttle to preserve left-tail behavior.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=63,
                            max_drawdown=-0.10,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r7_cap40_vol17_guard",
                    "Combine modest concentration and a 17% vol target, then brake on drawdown.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        max_asset_weight=0.40,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    None,
                ),
                (
                    "r7_top3_vol17_guard",
                    "Test a concentrated top-three version with a moderate vol lift and drawdown brake.",
                    clone(
                        top_n=3,
                        min_return=0.02,
                        trend_filter_days=None,
                        max_asset_weight=0.40,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            8,
            "global participation without forced global exposure",
            [
                (
                    "r8_global_core_top4",
                    "Add non-US equity choices but keep top-four selection and no forced global floor.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=4,
                        min_return=0.02,
                        max_asset_weight=0.35,
                        trend_filter_days=None,
                    ),
                    None,
                ),
                (
                    "r8_global_core_top5",
                    "Add non-US equity choices with a five-name basket to reduce US mega-cap dependence.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=5,
                        min_return=0.02,
                        max_asset_weight=0.30,
                        trend_filter_days=None,
                    ),
                    None,
                ),
                (
                    "r8_global_return_rank",
                    "Use raw return ranking in the global pool to test whether risk adjustment hides overseas upside.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=4,
                        min_return=0.02,
                        max_asset_weight=0.35,
                        ranking_metric="return",
                        trend_filter_days=None,
                    ),
                    None,
                ),
                (
                    "r8_global_vol17_guard",
                    "Global pool with the same moderate higher-participation guardrail as the US-focused winner.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=4,
                        min_return=0.02,
                        max_asset_weight=0.35,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    None,
                ),
                (
                    "r8_global_quality_guard",
                    "Global pool with trend-quality ranking and drawdown guardrails.",
                    clone(
                        tickers=ai_plus_global,
                        top_n=4,
                        min_return=0.02,
                        max_asset_weight=0.35,
                        ranking_metric="return_trend_quality",
                        trend_filter_days=None,
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            9,
            "frontier hurdle micro-tuning",
            [
                (
                    "r9_no_filter_min0225",
                    "Interpolate between the 2.0% and 2.5% momentum hurdles.",
                    clone(min_return=0.0225, trend_filter_days=None),
                    None,
                ),
                (
                    "r9_no_filter_min0275",
                    "Test whether the better 2.5% hurdle keeps improving with a slightly stricter gate.",
                    clone(min_return=0.0275, trend_filter_days=None),
                    None,
                ),
                (
                    "r9_no_filter_min030",
                    "Return to the original 3.0% hurdle but remove only the stale trend filter.",
                    clone(min_return=0.03, trend_filter_days=None),
                    None,
                ),
                (
                    "r9_min25_vol42_smoother",
                    "Combine the 2.5% hurdle with smoother volatility ranking.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_lookback_days=42,
                    ),
                    None,
                ),
                (
                    "r9_min25_cap40",
                    "Let the 2.5% hurdle concentrate winners modestly more.",
                    clone(min_return=0.025, trend_filter_days=None, max_asset_weight=0.40),
                    None,
                ),
            ],
        ),
        (
            10,
            "min25 higher participation",
            [
                (
                    "r10_min25_vol17_guard12",
                    "Apply the 2.5% hurdle to the 17% vol target drawdown-braked variant.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    None,
                ),
                (
                    "r10_min25_vol18_guard12",
                    "Apply the 2.5% hurdle to the current aggressive 18% vol target winner.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r10_min25_vol18_guard10",
                    "Keep 18% target exposure but brake earlier after a 10% drawdown.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=63,
                            max_drawdown=-0.10,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r10_min25_vol18_guard14",
                    "Let the aggressive variant breathe longer before throttling.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.60,
                        ),
                    ),
                    None,
                ),
                (
                    "r10_min25_vol19_guard12",
                    "Probe for the first signs of over-risking above the 18% volatility target.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.19,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.52,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            11,
            "drawdown brake tuning",
            [
                (
                    "r11_vol18_guard10_mult50",
                    "Brake the 18% target earlier and harder after a 10% drawdown.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=63,
                            max_drawdown=-0.10,
                            risk_multiplier=0.50,
                        ),
                    ),
                    None,
                ),
                (
                    "r11_vol18_guard10_mult60",
                    "Brake earlier but less severely so upside recovery is not cut off.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=63,
                            max_drawdown=-0.10,
                            risk_multiplier=0.60,
                        ),
                    ),
                    None,
                ),
                (
                    "r11_vol18_guard12_mult45",
                    "Keep the 12% trigger but cut risk more aggressively once breached.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.45,
                        ),
                    ),
                    None,
                ),
                (
                    "r11_vol18_guard14_mult55",
                    "Use a later drawdown trigger but still apply a meaningful throttle.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r11_vol18_guard12_lookback126",
                    "Use a slower drawdown lookback to avoid throttling on short-lived noise.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=126,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            12,
            "concentration and smoother aggressive variants",
            [
                (
                    "r12_min25_cap40_vol17_guard",
                    "Combine 2.5% hurdle, higher concentration, and a moderate 17% vol target.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        max_asset_weight=0.40,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.17,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.58,
                        ),
                    ),
                    None,
                ),
                (
                    "r12_min25_cap40_vol18_guard",
                    "Combine 2.5% hurdle, higher concentration, and the aggressive 18% vol target.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        max_asset_weight=0.40,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r12_min25_top3_vol18_guard",
                    "Test whether a concentrated top-three aggressive variant overfits or improves convexity.",
                    clone(
                        top_n=3,
                        min_return=0.025,
                        trend_filter_days=None,
                        max_asset_weight=0.40,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r12_min25_top5_vol18_guard",
                    "Test whether a broader top-five aggressive variant reduces fragility.",
                    clone(
                        top_n=5,
                        min_return=0.025,
                        trend_filter_days=None,
                        max_asset_weight=0.30,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r12_min25_vol42_vol18_guard",
                    "Combine the smoother volatility ranking with the aggressive vol target.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_lookback_days=42,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.12,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            13,
            "late drawdown trigger neighborhood",
            [
                (
                    "r13_vol18_guard13_mult55",
                    "Move the best 14% drawdown trigger one notch earlier.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.13,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r13_vol18_guard15_mult55",
                    "Move the best 14% drawdown trigger one notch later.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.15,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r13_vol18_guard14_mult50",
                    "Use the best 14% trigger but cut risk harder after breach.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.50,
                        ),
                    ),
                    None,
                ),
                (
                    "r13_vol18_guard14_mult60",
                    "Use the best 14% trigger but throttle less after breach.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.60,
                        ),
                    ),
                    None,
                ),
                (
                    "r13_vol18_guard14_lookback126",
                    "Use the best 14% trigger with a slower drawdown lookback.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=126,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            14,
            "late guard return hurdle neighborhood",
            [
                (
                    "r14_min0225_vol18_guard14",
                    "Pair the late drawdown trigger with a 2.25% momentum hurdle.",
                    clone(
                        min_return=0.0225,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r14_min0275_vol18_guard14",
                    "Pair the late drawdown trigger with a 2.75% momentum hurdle.",
                    clone(
                        min_return=0.0275,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r14_min030_vol18_guard14",
                    "Test whether a strict 3.0% hurdle still works with the late drawdown trigger.",
                    clone(
                        min_return=0.03,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r14_min025_vol185_guard14",
                    "Combine the stronger 2.5% hurdle with an 18.5% volatility target.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r14_min0225_vol185_guard14",
                    "Combine the interpolated 2.25% hurdle with an 18.5% volatility target.",
                    clone(
                        min_return=0.0225,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            15,
            "volatility envelope around late guard",
            [
                (
                    "r15_vol175_guard14_mult55",
                    "Step down from 18% to 17.5% target volatility with the late guard.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.175,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r15_vol185_guard14_mult55",
                    "Step up from 18% to 18.5% target volatility with the late guard.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r15_vol19_guard14_mult55",
                    "Test whether a 19% target remains compensated with the late guard.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.19,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r15_vol195_guard14_mult52",
                    "Probe above 19% with a slightly harder throttle.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.195,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.52,
                        ),
                    ),
                    None,
                ),
                (
                    "r15_vol20_guard14_mult50",
                    "Push to 20% target volatility to identify over-risking boundaries.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.20,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.50,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            16,
            "late guard structure variants",
            [
                (
                    "r16_cap40_vol18_guard14",
                    "Add modest concentration to the late-guard aggressive variant.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        max_asset_weight=0.40,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r16_top3_vol18_guard14",
                    "Concentrate the late-guard variant into the top three names.",
                    clone(
                        top_n=3,
                        min_return=0.02,
                        trend_filter_days=None,
                        max_asset_weight=0.40,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r16_top5_vol18_guard14",
                    "Broaden the late-guard variant into five names with tighter caps.",
                    clone(
                        top_n=5,
                        min_return=0.02,
                        trend_filter_days=None,
                        max_asset_weight=0.30,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r16_vol42_vol18_guard14",
                    "Use smoother volatility ranking with the late-guard aggressive variant.",
                    clone(
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_lookback_days=42,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r16_lookback84_vol18_guard14",
                    "Use a slower 84-day momentum lookback with the late-guard aggressive variant.",
                    clone(
                        lookback_days=84,
                        skip_days=5,
                        min_return=0.02,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.18,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            17,
            "winner micro mesh",
            [
                (
                    "r17_min024_vol185_guard14",
                    "Nudge the winning hurdle down to 2.4% while holding the 18.5% late guard.",
                    clone(
                        min_return=0.024,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r17_min026_vol185_guard14",
                    "Nudge the winning hurdle up to 2.6% while holding the 18.5% late guard.",
                    clone(
                        min_return=0.026,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r17_min025_vol1825_guard14",
                    "Nudge the winning volatility target down to 18.25%.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.1825,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r17_min025_vol1875_guard14",
                    "Nudge the winning volatility target up to 18.75%.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.1875,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r17_min025_vol185_guard135",
                    "Nudge the winning drawdown trigger earlier to 13.5%.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.135,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            18,
            "winner throttle mesh",
            [
                (
                    "r18_min025_vol185_guard145",
                    "Nudge the winning drawdown trigger later to 14.5%.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.145,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r18_min025_vol185_guard15",
                    "Nudge the winning drawdown trigger later to 15%.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.15,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r18_min025_vol185_mult50",
                    "Keep the winning trigger but throttle harder after breach.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.50,
                        ),
                    ),
                    None,
                ),
                (
                    "r18_min025_vol185_mult60",
                    "Keep the winning trigger but throttle less after breach.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.14,
                            risk_multiplier=0.60,
                        ),
                    ),
                    None,
                ),
                (
                    "r18_min025_vol185_lookback126",
                    "Keep the winning trigger but use a slower drawdown lookback.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=126,
                            max_drawdown=-0.14,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            19,
            "late exit mesh",
            [
                (
                    "r19_min025_vol185_guard155_mult60",
                    "Stay risk-on slightly longer with a 15.5% drawdown trigger and softer throttle.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.155,
                            risk_multiplier=0.60,
                        ),
                    ),
                    None,
                ),
                (
                    "r19_min025_vol185_guard16_mult60",
                    "Push the winning exit threshold to 16% while keeping the same vol cap.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.16,
                            risk_multiplier=0.60,
                        ),
                    ),
                    None,
                ),
                (
                    "r19_min025_vol185_guard17_mult65",
                    "Test whether a meaningfully later break line captures more pre-break upside.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.17,
                            risk_multiplier=0.65,
                        ),
                    ),
                    None,
                ),
                (
                    "r19_min025_vol185_guard18_mult70",
                    "Aggressive contrast: defer de-risking until an 18% trailing drawdown.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.18,
                            risk_multiplier=0.70,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            20,
            "vol stretch mesh",
            [
                (
                    "r20_min025_vol19_guard145_mult55",
                    "Raise the risk budget to 19% while preserving the best 14.5% guard.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.19,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.145,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r20_min025_vol19_guard15_mult60",
                    "Pair the 19% vol budget with a later 15% guard and softer throttle.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.19,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.15,
                            risk_multiplier=0.60,
                        ),
                    ),
                    None,
                ),
                (
                    "r20_min025_vol195_guard145_mult55",
                    "Probe whether 19.5% volatility adds upside without breaking the left tail.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.195,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.145,
                            risk_multiplier=0.55,
                        ),
                    ),
                    None,
                ),
                (
                    "r20_min025_vol20_guard145_mult50",
                    "Aggressive contrast: 20% vol target with harder post-break throttle.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.20,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.145,
                            risk_multiplier=0.50,
                        ),
                    ),
                    None,
                ),
            ],
        ),
        (
            21,
            "constructive floor on winner",
            [
                (
                    "r21_min025_vol185_guard145_floor60",
                    "Keep a 60% risk floor when broad trend, breadth, and credit are constructive.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.145,
                            risk_multiplier=0.55,
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.60,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 4,
                    },
                ),
                (
                    "r21_min025_vol185_guard15_floor65",
                    "Combine the later 15% guard with a 65% constructive risk floor.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.185,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.15,
                            risk_multiplier=0.55,
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.65,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 4,
                    },
                ),
                (
                    "r21_min025_vol19_guard145_floor60",
                    "Pair a modestly higher vol cap with the constructive risk floor.",
                    clone(
                        min_return=0.025,
                        trend_filter_days=None,
                        volatility_target=VolatilityTargetConfig(
                            annualized_volatility=0.19,
                            lookback_days=21,
                            max_leverage=1.0,
                        ),
                        drawdown_control=DrawdownControlConfig(
                            equity_lookback_days=84,
                            max_drawdown=-0.145,
                            risk_multiplier=0.55,
                        ),
                    ),
                    {
                        "mode": "risk_on_floor",
                        "floor": 0.60,
                        "signal": "balanced",
                        "lookback": 42,
                        "top_n": 4,
                    },
                ),
            ],
        ),
    ]
    for round_id, _label, specs in round_specs:
        for name, hypothesis, strategy, overlay in specs:
            candidates.append(
                UpsideCaptureCandidate(
                    round_id=round_id,
                    name=name,
                    hypothesis=hypothesis,
                    strategy=strategy,
                    overlay=overlay,
                )
            )
    return tuple(candidates)


def _apply_constructive_overlay(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    overlay: dict[str, Any],
) -> pd.DataFrame:
    defensive_ticker = str(overlay.get("defensive_ticker") or "BIL")
    risk_tickers = [ticker for ticker in weights.columns if ticker != defensive_ticker]
    available = [ticker for ticker in risk_tickers if ticker in prices]
    if not available or defensive_ticker not in weights:
        return weights
    mode = str(overlay.get("mode", "risk_on_floor"))
    floor = float(overlay.get("floor", 0.55))
    lookback = int(overlay.get("lookback", 42))
    top_n = int(overlay.get("top_n", min(4, len(available))))
    signal = _constructive_signal(prices, profile=str(overlay.get("signal", "balanced")))
    if mode == "recovery_floor":
        signal = signal & _recovery_signal(prices)
    signal = signal.reindex(weights.index).fillna(False)
    risk_weight = weights[available].sum(axis=1).clip(lower=0.0, upper=1.0)
    add_budget = (floor - risk_weight).clip(lower=0.0).where(signal, 0.0)
    if add_budget.max() <= 0.0:
        return weights
    mix = _momentum_mix(prices, available, lookback=lookback, top_n=top_n)
    output = weights.copy()
    output.loc[:, available] = output[available].add(mix.mul(add_budget, axis=0), fill_value=0.0)
    total_risk = output[available].sum(axis=1).clip(lower=0.0, upper=1.0)
    output.loc[:, defensive_ticker] = (1.0 - total_risk).clip(lower=0.0)
    return output.clip(lower=0.0).fillna(0.0)


def _constructive_signal(prices: pd.DataFrame, *, profile: str) -> pd.Series:
    filled = prices.ffill()
    index = filled.index
    qqq = _trend_ok(filled, "QQQ", 100) & (_return_ok(filled, "QQQ", 21, 0.00))
    spy = _trend_ok(filled, "SPY", 100) & (_return_ok(filled, "SPY", 21, -0.005))
    breadth = _relative_ok(filled, "RSP", "SPY", 21, -0.015)
    credit = _relative_ok(filled, "HYG", "LQD", 21, -0.010)
    ai = _relative_ok(filled, "SMH", "SPY", 21, -0.010) & _relative_ok(
        filled, "QQQ", "RSP", 21, -0.010
    )
    if profile == "credit_breadth":
        signal = qqq & spy & breadth & credit
    elif profile == "ai_leadership":
        signal = qqq & spy & ai & credit
    else:
        signal = qqq & spy & (breadth | ai | credit)
    return signal.reindex(index).fillna(False)


def _recovery_signal(prices: pd.DataFrame) -> pd.Series:
    filled = prices.ffill()
    qqq = filled["QQQ"] if "QQQ" in filled else filled.iloc[:, 0]
    qqq_drawdown = qqq / qqq.cummax() - 1.0
    repaired = qqq.pct_change(21, fill_method=None) > 0.035
    not_deepening = qqq.pct_change(5, fill_method=None) > -0.025
    recently_damaged = qqq_drawdown.rolling(63, min_periods=20).min() < -0.08
    return (repaired & not_deepening & recently_damaged).fillna(False)


def _momentum_mix(
    prices: pd.DataFrame,
    tickers: list[str],
    *,
    lookback: int,
    top_n: int,
) -> pd.DataFrame:
    momentum = lookback_returns(prices[tickers], lookback, 0).clip(lower=0.0)
    vol = realized_volatility(daily_returns(prices[tickers]), max(21, lookback)).replace(
        0.0, np.nan
    )
    score = momentum.div(vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ranks = score.rank(axis=1, ascending=False, method="first")
    selected = ranks <= min(top_n, len(tickers))
    raw = score.where(selected, 0.0).clip(lower=0.0)
    row_sum = raw.sum(axis=1)
    equal = selected.astype(float).div(
        selected.sum(axis=1).where(selected.sum(axis=1) > 0.0), axis=0
    )
    mix = raw.div(row_sum.where(row_sum > 0.0), axis=0).fillna(equal).fillna(0.0)
    return mix.reindex(prices.index).fillna(0.0)


def _trend_ok(prices: pd.DataFrame, ticker: str, days: int) -> pd.Series:
    if ticker not in prices:
        return pd.Series(False, index=prices.index)
    return (prices[ticker] > moving_average(prices[[ticker]], days)[ticker]).fillna(False)


def _return_ok(prices: pd.DataFrame, ticker: str, days: int, threshold: float) -> pd.Series:
    if ticker not in prices:
        return pd.Series(False, index=prices.index)
    return (prices[ticker].pct_change(days, fill_method=None) > threshold).fillna(False)


def _relative_ok(
    prices: pd.DataFrame,
    numerator: str,
    denominator: str,
    days: int,
    threshold: float,
) -> pd.Series:
    if numerator not in prices or denominator not in prices:
        return pd.Series(False, index=prices.index)
    relative = prices[numerator] / prices[denominator]
    return (relative.pct_change(days, fill_method=None) > threshold).fillna(False)


def _event_metrics_frame(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    base_result: BacktestResult,
) -> pd.DataFrame:
    spy = prices["SPY"].ffill()
    spy_returns_21 = spy.pct_change(21, fill_method=None)
    spy_forward_21 = spy.shift(-21) / spy - 1.0
    spy_forward_63 = spy.shift(-63) / spy - 1.0
    spy_dd_21 = _forward_drawdown(spy, 21)
    runup_mask = (spy_forward_21 > 0.04) | (spy_forward_63 > 0.09)
    dip_mask = (spy_forward_21 < -0.05) | (spy_dd_21 < -0.07)
    rows = []
    base_returns = base_result.returns.reindex(prices.index).fillna(0.0)
    for name, result in results.items():
        returns = result.returns.reindex(prices.index).fillna(0.0)
        risk_weight = result.weights.drop(columns=["BIL"], errors="ignore").sum(axis=1)
        row = {
            "candidate": name,
            "runup_capture_21d": _conditional_forward_return(returns, runup_mask, 21),
            "baseline_runup_capture_21d": _conditional_forward_return(base_returns, runup_mask, 21),
            "left_tail_loss_21d": _conditional_forward_return(returns, dip_mask, 21),
            "baseline_left_tail_loss_21d": _conditional_forward_return(base_returns, dip_mask, 21),
            "runup_risk_weight": float(
                risk_weight.reindex(runup_mask.index)[runup_mask.fillna(False)].mean()
            ),
            "dip_risk_weight": float(
                risk_weight.reindex(dip_mask.index)[dip_mask.fillna(False)].mean()
            ),
            "risk_weight_after_positive_month": float(
                risk_weight.reindex(spy_returns_21.index)[
                    (spy_returns_21 > 0.04).fillna(False)
                ].mean()
            ),
        }
        row["runup_capture_lift"] = row["runup_capture_21d"] - row["baseline_runup_capture_21d"]
        row["left_tail_loss_delta"] = row["left_tail_loss_21d"] - row["baseline_left_tail_loss_21d"]
        rows.append(row)
    return pd.DataFrame(rows)


def _event_observations(
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    base_result: BacktestResult,
) -> pd.DataFrame:
    spy = prices["SPY"].ffill()
    spy_forward_21 = spy.shift(-21) / spy - 1.0
    spy_dd_21 = _forward_drawdown(spy, 21)
    selected_dates = spy_forward_21.abs().sort_values(ascending=False).head(40).index
    rows = []
    base_returns = base_result.returns.reindex(prices.index).fillna(0.0)
    top_names = list(results)[:]
    for date in selected_dates:
        for name in top_names:
            returns = results[name].returns.reindex(prices.index).fillna(0.0)
            start = prices.index.get_loc(date)
            end = min(start + 21, len(prices.index) - 1)
            rows.append(
                {
                    "origin_date": pd.Timestamp(date).date().isoformat(),
                    "candidate": name,
                    "spy_forward_21d": float(spy_forward_21.loc[date]),
                    "spy_forward_dd_21d": float(spy_dd_21.loc[date]),
                    "strategy_forward_21d": float(
                        (1.0 + returns.iloc[start : end + 1]).prod() - 1.0
                    ),
                    "baseline_forward_21d": float(
                        (1.0 + base_returns.iloc[start : end + 1]).prod() - 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _score_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    output["max_drawdown_penalty"] = (output["max_drawdown"].abs() - 0.23).clip(lower=0.0) * 2.0
    output["left_tail_penalty"] = output["left_tail_loss_delta"].clip(upper=0.0).abs() * 3.0
    output["turnover_penalty"] = (output["average_turnover"] - 0.16).clip(lower=0.0) * 0.25
    output["robustness_penalty"] = (
        output["worst_3y_cagr_delta"].clip(upper=0.0).abs() * 0.30
        + output["walk_forward_worst_cagr_delta"].clip(upper=0.0).abs() * 0.25
        + output["calendar_underperform_years_delta"].clip(lower=0.0) * 0.01
    )
    output["research_score"] = (
        output["cagr"]
        + 1.75 * output["runup_capture_lift"]
        + 0.35 * output["calmar"]
        - output["max_drawdown_penalty"]
        - output["left_tail_penalty"]
        - output["turnover_penalty"]
        - output["robustness_penalty"]
    )
    output["beats_baseline_runup"] = output["runup_capture_lift"] > 0.0025
    output["left_tail_not_worse"] = output["left_tail_loss_delta"] >= -0.005
    output["promotion_candidate"] = (
        output["beats_baseline_runup"]
        & output["left_tail_not_worse"]
        & (output["max_drawdown"] >= -0.25)
        & (output["worst_3y_cagr_delta"] >= -0.01)
        & (output["walk_forward_worst_cagr_delta"] >= -0.01)
        & (
            output["cagr"]
            >= output.loc[output["candidate"].eq("r0_i111_current_baseline"), "cagr"].iloc[0] - 0.01
        )
    )
    return output


def _add_robustness_metrics(
    summary: pd.DataFrame,
    *,
    window_summary: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    calendar_metrics: pd.DataFrame,
    baseline_name: str,
) -> pd.DataFrame:
    output = summary.copy()
    if not window_summary.empty:
        windows = window_summary.reset_index().rename(columns={"name": "candidate"})
        for window in ("1y", "3y", "5y"):
            subset = windows[windows["window"].eq(window)][
                ["candidate", "worst_cagr", "positive_window_rate", "worst_drawdown"]
            ].rename(
                columns={
                    "worst_cagr": f"worst_{window}_cagr",
                    "positive_window_rate": f"{window}_positive_window_rate",
                    "worst_drawdown": f"worst_{window}_drawdown",
                }
            )
            output = output.merge(subset, on="candidate", how="left")
    if not walk_forward_summary.empty:
        walk = walk_forward_summary.reset_index().rename(columns={"name": "candidate"})
        output = output.merge(
            walk[
                [
                    "candidate",
                    "walk_forward_median_cagr",
                    "walk_forward_worst_cagr",
                    "walk_forward_positive_rate",
                    "walk_forward_worst_drawdown",
                ]
            ],
            on="candidate",
            how="left",
        )
    if not calendar_metrics.empty:
        baseline_years = calendar_metrics[calendar_metrics["name"].eq(baseline_name)][
            ["window", "total_return"]
        ].rename(columns={"total_return": "baseline_total_return"})
        joined = calendar_metrics.merge(baseline_years, on="window", how="left")
        underperform = (
            joined.assign(
                underperformed=joined["total_return"] < joined["baseline_total_return"] - 0.005
            )
            .groupby("name", observed=True)
            .agg(
                calendar_years=("window", "count"),
                negative_calendar_years=("total_return", lambda values: int((values < 0).sum())),
                calendar_underperform_years=("underperformed", "sum"),
            )
            .reset_index()
            .rename(columns={"name": "candidate"})
        )
        output = output.merge(underperform, on="candidate", how="left")
    baseline = output[output["candidate"].eq(baseline_name)].iloc[0]
    for column in (
        "worst_3y_cagr",
        "walk_forward_worst_cagr",
        "calendar_underperform_years",
    ):
        if column not in output:
            output[column] = np.nan
        output[f"{column}_delta"] = output[column] - baseline.get(column, np.nan)
    return output


def _add_candidate_metadata(
    summary: pd.DataFrame,
    candidates: tuple[UpsideCaptureCandidate, ...],
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        [
            {
                "candidate": candidate.name,
                "round_id": candidate.round_id,
                "hypothesis": candidate.hypothesis,
                "overlay": candidate.overlay or {},
                "tickers": ",".join(candidate.strategy.tickers),
            }
            for candidate in candidates
        ]
    )
    return summary.merge(metadata, on="candidate", how="left")


def _findings_markdown(summary: pd.DataFrame, events: pd.DataFrame) -> str:
    baseline = summary[summary["candidate"].eq("r0_i111_current_baseline")].iloc[0]
    top = summary.sort_values("research_score", ascending=False).head(8)
    lines = [
        "# Upside Capture Lab",
        "",
        "## Goal",
        "",
        (
            "Test whether trade-bot can keep its left-tail discipline while adding a "
            "controlled participation floor in confirmed constructive regimes."
        ),
        "",
        "## Baseline",
        "",
        (
            f"`r0_i111_current_baseline`: CAGR {baseline['cagr']:.2%}, "
            f"max drawdown {baseline['max_drawdown']:.2%}, "
            f"run-up capture {baseline['runup_capture_21d']:.2%}, "
            f"left-tail loss {baseline['left_tail_loss_21d']:.2%}."
        ),
        "",
        "## Best Candidates",
        "",
        "| candidate | round | CAGR | max DD | run-up lift | left-tail delta | score | promote? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['candidate']} | {int(row['round_id'])} | {row['cagr']:.2%} | "
            f"{row['max_drawdown']:.2%} | {row['runup_capture_lift']:.2%} | "
            f"{row['left_tail_loss_delta']:.2%} | {row['research_score']:.3f} | "
            f"{bool(row['promotion_candidate'])} |"
        )
    lines.extend(["", "## Robustness Check", ""])
    lines.append(
        "| candidate | worst 3y CAGR delta | walk-forward worst CAGR delta | calendar underperform years |"
    )
    lines.append("|---|---:|---:|---:|")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['worst_3y_cagr_delta']:.2%} | "
            f"{row['walk_forward_worst_cagr_delta']:.2%} | "
            f"{int(row['calendar_underperform_years'])} |"
        )

    lines.extend(["", "## Round Winners", ""])
    for round_id, frame in summary.groupby("round_id"):
        winner = frame.sort_values("research_score", ascending=False).iloc[0]
        lines.append(
            f"- Round {int(round_id)}: `{winner['candidate']}` "
            f"(score {winner['research_score']:.3f}, run-up lift {winner['runup_capture_lift']:.2%}, "
            f"left-tail delta {winner['left_tail_loss_delta']:.2%})."
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Positive run-up lift means the candidate captured more of strong forward SPY windows than the current i111 baseline.",
            "- Left-tail delta should be close to zero or positive; a negative value means the candidate lost more than baseline in fast drawdown windows.",
            "- Promotion candidates are not production changes. They are candidates for the next normal experiment shelf and paper monitoring.",
        ]
    )
    return "\n".join(lines) + "\n"


def _forward_drawdown(series: pd.Series, days: int) -> pd.Series:
    values = []
    for index in range(len(series)):
        end = min(index + days, len(series) - 1)
        window = series.iloc[index : end + 1]
        relative = window / float(window.iloc[0])
        values.append(float((relative / relative.cummax() - 1.0).min()))
    return pd.Series(values, index=series.index)


def _conditional_forward_return(returns: pd.Series, mask: pd.Series, days: int) -> float:
    values = []
    aligned_returns = returns.reindex(mask.index).fillna(0.0)
    for date in mask[mask.fillna(False)].index:
        start = aligned_returns.index.get_loc(date)
        end = min(start + days, len(aligned_returns.index) - 1)
        values.append(float((1.0 + aligned_returns.iloc[start : end + 1]).prod() - 1.0))
    return float(np.mean(values)) if values else float("nan")


def _candidate_tickers(candidates: tuple[UpsideCaptureCandidate, ...]) -> set[str]:
    tickers: set[str] = {"SPY", "QQQ", "RSP", "HYG", "LQD", "SMH", "BIL"}
    for candidate in candidates:
        tickers.update(required_strategy_tickers(candidate.strategy))
    return tickers


def _strategy_prices(
    prices: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    selected = required_strategy_tickers(strategy)
    unusable = unusable_required_price_columns(prices, selected)
    if unusable:
        raise ValueError(f"Missing, empty, or stale required price columns: {unusable}")
    return prices[selected].dropna(how="all")
