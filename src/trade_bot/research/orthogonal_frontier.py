from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.config import ExecutionConfig, load_config, required_strategy_tickers
from trade_bot.strategies.momentum import build_strategy_weights

COMMON_START = pd.Timestamp("2014-01-01")
PERIODS = {
    "development": (pd.Timestamp("2014-01-01"), pd.Timestamp("2018-12-31")),
    "validation": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")),
    "holdout": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
    "forward": (pd.Timestamp("2025-01-01"), None),
    "common_full": (COMMON_START, None),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    novelty: str
    existing_overlap: str
    hypothesis: str
    target_weights: pd.DataFrame
    research_wave: int = 1
    selection_eligible: bool = True


@dataclass(frozen=True)
class SurveyResult:
    candidate_metrics: pd.DataFrame
    control_metrics: pd.DataFrame
    family_selection: pd.DataFrame
    state_conditioned: pd.DataFrame
    rolling_evidence: pd.DataFrame
    correlation_evidence: pd.DataFrame
    blend_evidence: pd.DataFrame
    sleeve_sizing: pd.DataFrame
    post_holdout_diagnostics: pd.DataFrame
    data_frontiers: pd.DataFrame
    report_path: Path


def run_orthogonal_frontier_survey(
    *,
    prices_path: str | Path = "data/cache/yahoo_prices.parquet",
    fred_path: str | Path = "data/cache/fred_series.parquet",
    config_path: str | Path = "configs/baseline.yaml",
    rotation_returns_path: str | Path | None = None,
    swing_returns_path: str | Path | None = None,
    output_dir: str | Path = "reports/orthogonal_frontier",
) -> SurveyResult:
    prices = _load_prices(Path(prices_path))
    fred = pd.read_parquet(fred_path).sort_index()
    candidates = generate_candidates(prices, fred)
    controls = _control_returns(
        prices,
        config_path=Path(config_path),
        rotation_returns_path=(Path(rotation_returns_path) if rotation_returns_path else None),
        swing_returns_path=(Path(swing_returns_path) if swing_returns_path else None),
    )
    states = point_in_time_market_states(prices["SPY"])
    control_metrics = _control_metric_frame(controls)
    candidate_results = {
        candidate.name: _run_candidate(candidate, prices) for candidate in candidates
    }

    metric_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    rolling_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    blend_rows: list[dict[str, object]] = []
    metadata = {candidate.name: candidate for candidate in candidates}

    for name, result in candidate_results.items():
        candidate = metadata[name]
        for period, (start, end) in PERIODS.items():
            returns = _period_slice(result.returns, start, end)
            row = _metric_row(returns)
            metric_rows.append(
                {
                    "candidate": name,
                    "family": candidate.family,
                    "novelty": candidate.novelty,
                    "existing_overlap": candidate.existing_overlap,
                    "hypothesis": candidate.hypothesis,
                    "research_wave": candidate.research_wave,
                    "selection_eligible": candidate.selection_eligible,
                    "period": period,
                    **row,
                }
            )
        state_rows.extend(
            _state_rows(name, candidate.family, result.returns, states)
        )
        rolling_rows.extend(
            _rolling_rows(name, candidate.family, result.returns, controls["spy_buy_hold"])
        )
        correlation_rows.extend(
            _correlation_rows(name, candidate.family, result.returns, controls)
        )
        blend_rows.extend(
            _blend_rows(name, candidate.family, result.returns, controls)
        )

    candidate_metrics = pd.DataFrame(metric_rows)
    state_conditioned = pd.DataFrame(state_rows)
    rolling_evidence = pd.DataFrame(rolling_rows)
    correlation_evidence = pd.DataFrame(correlation_rows)
    blend_evidence = pd.DataFrame(blend_rows)
    sleeve_sizing = _sleeve_sizing_evidence(candidate_results, metadata, controls)
    family_selection = _select_families(
        candidate_metrics,
        rolling_evidence,
        correlation_evidence,
        blend_evidence,
    )
    post_holdout_diagnostics = _post_holdout_diagnostics(
        candidate_metrics,
        rolling_evidence,
        correlation_evidence,
        blend_evidence,
    )
    data_frontiers = _data_frontiers()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_metrics.to_csv(output / "candidate_metrics.csv", index=False)
    control_metrics.to_csv(output / "control_metrics.csv", index=False)
    family_selection.to_csv(output / "family_selection.csv", index=False)
    state_conditioned.to_csv(output / "state_conditioned.csv", index=False)
    rolling_evidence.to_csv(output / "rolling_evidence.csv", index=False)
    correlation_evidence.to_csv(output / "correlation_evidence.csv", index=False)
    blend_evidence.to_csv(output / "blend_evidence.csv", index=False)
    sleeve_sizing.to_csv(output / "sleeve_sizing.csv", index=False)
    post_holdout_diagnostics.to_csv(
        output / "post_holdout_diagnostics.csv",
        index=False,
    )
    data_frontiers.to_csv(output / "data_frontiers.csv", index=False)
    pd.DataFrame(
        {
            "market_state": states,
            "spy_return": prices["SPY"].pct_change(fill_method=None),
        }
    ).to_csv(output / "market_states.csv", index_label="date")
    report_path = output / "report.md"
    report_path.write_text(
        _render_report(
            family_selection,
            post_holdout_diagnostics,
            sleeve_sizing,
            control_metrics,
            state_conditioned,
            data_frontiers,
            variants_tested=len(candidates),
            data_end=prices.index.max(),
        ),
        encoding="utf-8",
    )
    manifest = {
        "study": "orthogonal_frontier",
        "data_start": str(prices.index.min().date()),
        "data_end": str(prices.index.max().date()),
        "common_start": str(COMMON_START.date()),
        "variants_tested": len(candidates),
        "families_tested": int(len({candidate.family for candidate in candidates})),
        "selection_policy": (
            "Family winners selected only from 2014-2021 development and validation; "
            "2022-2024 holdout and 2025+ forward rows are reported after selection."
        ),
        "price_hash": _frame_hash(prices),
        "fred_hash": _frame_hash(fred),
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return SurveyResult(
        candidate_metrics=candidate_metrics,
        control_metrics=control_metrics,
        family_selection=family_selection,
        state_conditioned=state_conditioned,
        rolling_evidence=rolling_evidence,
        correlation_evidence=correlation_evidence,
        blend_evidence=blend_evidence,
        sleeve_sizing=sleeve_sizing,
        post_holdout_diagnostics=post_holdout_diagnostics,
        data_frontiers=data_frontiers,
        report_path=report_path,
    )


def generate_candidates(prices: pd.DataFrame, fred: pd.DataFrame) -> tuple[Candidate, ...]:
    builders: tuple[Callable[[pd.DataFrame, pd.DataFrame], list[Candidate]], ...] = (
        _cash_hurdle_candidates,
        _residual_momentum_candidates,
        _threshold_rebalancing_candidates,
        _yield_curve_candidates,
        _macro_quadrant_candidates,
        _calendar_candidates,
        _dislocation_candidates,
        _correlation_defense_candidates,
        _structural_factor_candidates,
        _adaptive_threshold_candidates,
        _trend_gated_residual_candidates,
        _expanded_correlation_candidates,
        _cross_asset_calendar_candidates,
    )
    candidates = [candidate for builder in builders for candidate in builder(prices, fred)]
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Orthogonal frontier candidate names must be unique.")
    return tuple(candidates)


def point_in_time_market_states(spy: pd.Series) -> pd.Series:
    spy = spy.dropna().astype(float)
    returns = spy.pct_change(fill_method=None)
    drawdown = spy / spy.cummax() - 1.0
    fast_weak = spy < spy.rolling(63, min_periods=63).mean()
    slow_weak = spy < spy.rolling(200, min_periods=200).mean()
    volatility = returns.rolling(63, min_periods=42).std() * np.sqrt(252.0)
    high_volatility = volatility > volatility.rolling(756, min_periods=252).quantile(0.75)
    rapid_loss = (spy.pct_change(21, fill_method=None) <= -0.05) | (
        spy.pct_change(63, fill_method=None) <= -0.08
    )
    breaking = ((drawdown <= -0.10) & fast_weak & (high_volatility | rapid_loss)) | (
        rapid_loss & high_volatility & slow_weak
    )
    transition = ~breaking & ((drawdown <= -0.05) | fast_weak | high_volatility)
    states = pd.Series("normal", index=spy.index, dtype="object")
    states.loc[transition.fillna(False)] = "transition"
    states.loc[breaking.fillna(False)] = "break"
    return states.rename("market_state")


def _cash_hurdle_candidates(prices: pd.DataFrame, _: pd.DataFrame) -> list[Candidate]:
    universe = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC"]
    candidates: list[Candidate] = []
    for lookback in (63, 126, 252):
        for top_n in (2, 3):
            for weighting in ("equal", "inverse_volatility"):
                for frequency in ("W-FRI", "M"):
                    momentum = prices[universe].pct_change(lookback, fill_method=None)
                    cash_hurdle = prices["BIL"].pct_change(lookback, fill_method=None)
                    scores = momentum.sub(cash_hurdle, axis=0)
                    raw = _ranked_weights(
                        scores,
                        prices[universe],
                        top_n=top_n,
                        weighting=weighting,
                    )
                    raw["BIL"] = (1.0 - raw.sum(axis=1)).clip(lower=0.0)
                    weights = _scheduled_weights(raw, frequency)
                    name = f"cash_hurdle_l{lookback}_n{top_n}_{weighting}_{frequency}"
                    candidates.append(
                        Candidate(
                            name=name,
                            family="cash_hurdle_cross_asset",
                            novelty="low",
                            existing_overlap="Direct extension of Trade Bot dual momentum.",
                            hypothesis="Require leadership to beat the realized T-bill hurdle, not zero.",
                            target_weights=weights,
                        )
                    )
    return candidates


def _residual_momentum_candidates(prices: pd.DataFrame, _: pd.DataFrame) -> list[Candidate]:
    universe = ["RSP", "IWM", "VTV", "VUG", "QUAL", "USMV", "MTUM", "EFA"]
    returns = prices[["SPY", *universe]].pct_change(fill_method=None)
    candidates: list[Candidate] = []
    for beta_window in (63, 126):
        market_variance = returns["SPY"].rolling(beta_window).var()
        beta = pd.DataFrame(
            {
                ticker: returns[ticker].rolling(beta_window).cov(returns["SPY"])
                / market_variance
                for ticker in universe
            }
        )
        residual = returns[universe] - beta.mul(returns["SPY"], axis=0)
        for lookback in (63, 126):
            residual_score = (1.0 + residual.fillna(0.0)).rolling(lookback).apply(
                np.prod,
                raw=True,
            ) - 1.0
            for top_n in (2, 3):
                raw = _ranked_weights(
                    residual_score,
                    prices[universe],
                    top_n=top_n,
                    weighting="inverse_volatility",
                )
                raw["BIL"] = (1.0 - raw.sum(axis=1)).clip(lower=0.0)
                weights = _scheduled_weights(raw, "M")
                candidates.append(
                    Candidate(
                        name=f"residual_beta{beta_window}_l{lookback}_n{top_n}",
                        family="beta_residual_leadership",
                        novelty="medium",
                        existing_overlap="Momentum remains, but broad-market beta is removed before ranking.",
                        hypothesis="Residual leadership may diversify raw momentum and AI beta.",
                        target_weights=weights,
                    )
                )
    return candidates


def _threshold_rebalancing_candidates(
    prices: pd.DataFrame,
    _: pd.DataFrame,
) -> list[Candidate]:
    allocations = {
        "balanced": {"SPY": 0.55, "IEF": 0.25, "GLD": 0.15, "BIL": 0.05},
        "all_weather": {"SPY": 0.40, "IEF": 0.25, "GLD": 0.20, "BIL": 0.15},
        "inflation_aware": {
            "SPY": 0.40,
            "TIP": 0.20,
            "GLD": 0.15,
            "DBC": 0.15,
            "BIL": 0.10,
        },
    }
    candidates: list[Candidate] = []
    for allocation_name, allocation in allocations.items():
        for band in (0.03, 0.06, 0.10):
            weights = _threshold_rebalance_weights(prices, allocation, band)
            candidates.append(
                Candidate(
                    name=f"threshold_{allocation_name}_b{int(band * 100):02d}",
                    family="threshold_rebalancing",
                    novelty="high",
                    existing_overlap="No forecast; unlike Trade Bot, it harvests allocation drift.",
                    hypothesis="Tolerance-band rebalancing may monetize volatility without predicting direction.",
                    target_weights=weights,
                )
            )
    return candidates


def _yield_curve_candidates(prices: pd.DataFrame, fred: pd.DataFrame) -> list[Candidate]:
    data = fred.reindex(prices.index).ffill()
    candidates: list[Candidate] = []
    for lookback in (42, 84, 126):
        yield_change = data["DGS10"].diff(lookback)
        slope = data["DGS10"] - data["DGS2"]
        for slope_floor in (-0.25, 0.0, 0.50):
            for inflation_ceiling in (0.025, 0.030):
                weights = pd.DataFrame(0.0, index=prices.index, columns=["BIL", "IEF", "TLT", "TIP"])
                inflation_high = (data["T10YIE"] / 100.0) >= inflation_ceiling
                long_duration = (yield_change < -0.30) & (slope >= slope_floor)
                medium_duration = (yield_change < 0.0) & ~long_duration
                weights.loc[inflation_high, "TIP"] = 1.0
                weights.loc[~inflation_high & long_duration, "TLT"] = 1.0
                weights.loc[~inflation_high & medium_duration, "IEF"] = 1.0
                weights.loc[weights.sum(axis=1).eq(0.0), "BIL"] = 1.0
                weights = _scheduled_weights(weights, "W-FRI")
                candidates.append(
                    Candidate(
                        name=(
                            f"curve_l{lookback}_s{int((slope_floor + 1) * 100):03d}"
                            f"_i{int(inflation_ceiling * 1000):02d}"
                        ),
                        family="yield_curve_duration",
                        novelty="medium_high",
                        existing_overlap="Trade Bot rotates bond ETFs, but usually from prices rather than direct yield state.",
                        hypothesis="Direct curve, yield-change, and breakeven signals may improve duration timing.",
                        target_weights=weights,
                    )
                )
    return candidates


def _macro_quadrant_candidates(prices: pd.DataFrame, fred: pd.DataFrame) -> list[Candidate]:
    candidates: list[Candidate] = []
    for release_lag in (30, 60):
        macro = _lagged_macro_panel(fred, prices.index, release_lag)
        growth = pd.concat(
            [
                macro["INDPRO"].pct_change(180, fill_method=None),
                macro["PAYEMS"].pct_change(180, fill_method=None),
                macro["RSAFS"].pct_change(180, fill_method=None),
                macro["CFNAI"] / 10.0,
                -macro["UNRATE"].diff(180) / 10.0,
            ],
            axis=1,
        ).median(axis=1)
        inflation = pd.concat(
            [
                macro["CPIAUCSL"].pct_change(365, fill_method=None),
                macro["PCEPI"].pct_change(365, fill_method=None),
                macro["T10YIE"] / 100.0,
            ],
            axis=1,
        ).median(axis=1)
        for growth_floor in (0.0, 0.01):
            for inflation_floor in (0.025, 0.03):
                for template in ("balanced", "defensive"):
                    weights = _macro_quadrant_weights(
                        prices.index,
                        growth,
                        inflation,
                        growth_floor=growth_floor,
                        inflation_floor=inflation_floor,
                        template=template,
                    )
                    weights = _scheduled_weights(weights, "M")
                    candidates.append(
                        Candidate(
                            name=(
                                f"macro_lag{release_lag}_g{int(growth_floor * 100):02d}"
                                f"_i{int(inflation_floor * 1000):02d}_{template}"
                            ),
                            family="macro_growth_inflation",
                            novelty="medium",
                            existing_overlap="Trade Bot already monitors macro regimes; revised-history risk remains.",
                            hypothesis="Lagged growth/inflation quadrants may diversify price-only timing.",
                            target_weights=weights,
                        )
                    )
    return candidates


def _calendar_candidates(prices: pd.DataFrame, _: pd.DataFrame) -> list[Candidate]:
    candidates: list[Candidate] = []
    index = prices.index
    month_position = pd.Series(index.to_series().groupby(index.to_period("M")).cumcount(), index=index)
    month_size = pd.Series(index.to_series().groupby(index.to_period("M")).transform("size"), index=index)
    from_end = month_size - month_position - 1
    for before, after in ((1, 3), (3, 3), (5, 5)):
        active = (from_end <= before) | (month_position <= after)
        weights = _binary_risk_weights(index, active, risk_ticker="SPY")
        candidates.append(
            Candidate(
                name=f"turn_month_b{before}_a{after}",
                family="calendar_anomaly",
                novelty="high",
                existing_overlap="No equivalent calendar mechanism is registered in Trade Bot.",
                hypothesis="Persistent institutional month-boundary flows may create a sparse equity premium.",
                target_weights=weights,
            )
        )
    for months in ((11, 12, 1, 2, 3, 4), (10, 11, 12, 1, 2, 3, 4, 5)):
        active = pd.Series(index.month.isin(months), index=index)
        weights = _binary_risk_weights(index, active, risk_ticker="SPY")
        candidates.append(
            Candidate(
                name=f"seasonal_{len(months)}m",
                family="calendar_anomaly",
                novelty="high",
                existing_overlap="No equivalent calendar mechanism is registered in Trade Bot.",
                hypothesis="Broad seasonal risk premia may be independent of momentum and macro forecasts.",
                target_weights=weights,
            )
        )
    return candidates


def _dislocation_candidates(prices: pd.DataFrame, fred: pd.DataFrame) -> list[Candidate]:
    spy = prices["SPY"]
    returns = spy.pct_change(fill_method=None)
    volatility = returns.rolling(63).std() * np.sqrt(252.0)
    vix = fred["VIXCLS"].reindex(prices.index).ffill()
    candidates: list[Candidate] = []
    for shock in (-0.04, -0.06, -0.08):
        shock_event = spy.pct_change(5, fill_method=None) <= shock
        high_stress = vix >= vix.rolling(756, min_periods=252).quantile(0.80)
        for hold in (5, 10, 20):
            for confirmation in ("none", "positive_day"):
                event = shock_event & high_stress & (volatility > 0.15)
                if confirmation == "positive_day":
                    event = event & returns.gt(0.0)
                active = _event_holding_mask(event, hold)
                weights = _binary_risk_weights(prices.index, active, risk_ticker="SPY")
                candidates.append(
                    Candidate(
                        name=f"dislocation_s{abs(int(shock * 100)):02d}_h{hold}_{confirmation}",
                        family="dislocation_recovery",
                        novelty="low",
                        existing_overlap="Substantially overlaps Trade Bot dip-reentry and risk-repair research.",
                        hypothesis="Sparse post-capitulation exposure may earn rebound returns with limited market time.",
                        target_weights=weights,
                    )
                )
    return candidates


def _correlation_defense_candidates(prices: pd.DataFrame, _: pd.DataFrame) -> list[Candidate]:
    returns = prices[["SPY", "IEF", "GLD"]].pct_change(fill_method=None)
    candidates: list[Candidate] = []
    for correlation_window in (63, 126):
        correlations = pd.DataFrame(
            {
                asset: returns["SPY"].rolling(correlation_window).corr(returns[asset])
                for asset in ("IEF", "GLD")
            }
        )
        for trend_window in (126, 200):
            trends = prices[["SPY", "IEF", "GLD"]].div(
                prices[["SPY", "IEF", "GLD"]].rolling(trend_window).mean()
            ) - 1.0
            for risk_weight in (0.4, 0.6):
                weights = pd.DataFrame(
                    0.0,
                    index=prices.index,
                    columns=["SPY", "IEF", "GLD", "BIL"],
                )
                risk_on = trends["SPY"] > 0.0
                weights.loc[risk_on, "SPY"] = risk_weight
                defense_scores = -correlations + trends[["IEF", "GLD"]]
                best_defense = defense_scores.fillna(-np.inf).idxmax(axis=1)
                for asset in ("IEF", "GLD"):
                    choose = best_defense.eq(asset) & trends[asset].gt(0.0)
                    weights.loc[choose, asset] = 1.0 - weights.loc[choose, "SPY"]
                weights["BIL"] = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
                weights = _scheduled_weights(weights, "W-FRI")
                candidates.append(
                    Candidate(
                        name=f"corrdef_c{correlation_window}_t{trend_window}_r{int(risk_weight * 100)}",
                        family="correlation_adaptive_defense",
                        novelty="medium_high",
                        existing_overlap="Uses trend, but selects defenses from realized hedge behavior rather than fixed labels.",
                        hypothesis="The best hedge changes; correlation-aware defense may survive inflation and growth shocks.",
                        target_weights=weights,
                    )
                )
    return candidates


def _structural_factor_candidates(prices: pd.DataFrame, _: pd.DataFrame) -> list[Candidate]:
    allocations = {
        "quality_value_lowvol": {"QUAL": 0.34, "VTV": 0.33, "USMV": 0.33},
        "quality_dividend": {"QUAL": 0.50, "VIG": 0.50},
        "quality_value_market": {"QUAL": 0.30, "VTV": 0.30, "SPY": 0.40},
        "lowvol_value_market": {"USMV": 0.30, "VTV": 0.30, "SPY": 0.40},
    }
    candidates: list[Candidate] = []
    for name, allocation in allocations.items():
        weights = pd.DataFrame(0.0, index=prices.index, columns=list(allocation))
        ready = prices[list(allocation)].notna().all(axis=1)
        for ticker, weight in allocation.items():
            weights.loc[ready, ticker] = weight
        candidates.append(
            Candidate(
                name=f"structural_{name}",
                family="structural_factor_blend",
                novelty="medium",
                existing_overlap="Trade Bot rotates factor ETFs; this tests a non-timed structural sleeve instead.",
                hypothesis="A permanent quality/value/low-vol sleeve may diversify timing systems.",
                target_weights=weights,
            )
        )
    return candidates


def _adaptive_threshold_candidates(prices: pd.DataFrame, _: pd.DataFrame) -> list[Candidate]:
    allocations = {
        "balanced": {"SPY": 0.55, "IEF": 0.25, "GLD": 0.15, "BIL": 0.05},
        "all_weather": {"SPY": 0.40, "IEF": 0.25, "GLD": 0.20, "BIL": 0.15},
        "inflation_aware": {
            "SPY": 0.40,
            "TIP": 0.20,
            "GLD": 0.15,
            "DBC": 0.15,
            "BIL": 0.10,
        },
    }
    candidates: list[Candidate] = []
    for allocation_name, allocation in allocations.items():
        for volatility_window in (42, 126):
            for base_band in (0.03, 0.06):
                for mode in ("narrow_stress", "widen_stress"):
                    weights = _adaptive_threshold_weights(
                        prices,
                        allocation,
                        base_band=base_band,
                        volatility_window=volatility_window,
                        mode=mode,
                    )
                    candidates.append(
                        Candidate(
                            name=(
                                f"adaptive_threshold_{allocation_name}_v{volatility_window}"
                                f"_b{int(base_band * 100):02d}_{mode}"
                            ),
                            family="adaptive_threshold_rebalancing",
                            novelty="high",
                            existing_overlap="Post-holdout extension of the non-forecast rebalance family.",
                            hypothesis="Volatility-scaled bands may preserve rebalance convexity while reducing churn.",
                            target_weights=weights,
                            research_wave=2,
                            selection_eligible=False,
                        )
                    )
    return candidates


def _trend_gated_residual_candidates(
    prices: pd.DataFrame,
    _: pd.DataFrame,
) -> list[Candidate]:
    universe = ["RSP", "IWM", "VTV", "VUG", "QUAL", "USMV", "MTUM", "EFA"]
    returns = prices[["SPY", *universe]].pct_change(fill_method=None)
    candidates: list[Candidate] = []
    for beta_window in (63, 126):
        market_variance = returns["SPY"].rolling(beta_window).var()
        beta = pd.DataFrame(
            {
                ticker: returns[ticker].rolling(beta_window).cov(returns["SPY"])
                / market_variance
                for ticker in universe
            }
        )
        residual = returns[universe] - beta.mul(returns["SPY"], axis=0)
        for lookback in (63, 126):
            residual_score = (1.0 + residual.fillna(0.0)).rolling(lookback).apply(
                np.prod,
                raw=True,
            ) - 1.0
            for top_n in (2, 3):
                for trend_window in (126, 200):
                    trend_ok = prices[universe].gt(
                        prices[universe].rolling(trend_window).mean()
                    )
                    scores = residual_score.where(trend_ok, np.nan)
                    raw = _ranked_weights(
                        scores,
                        prices[universe],
                        top_n=top_n,
                        weighting="inverse_volatility",
                    )
                    raw["BIL"] = (1.0 - raw.sum(axis=1)).clip(lower=0.0)
                    weights = _scheduled_weights(raw, "M")
                    candidates.append(
                        Candidate(
                            name=(
                                f"residual_trend_b{beta_window}_l{lookback}_n{top_n}"
                                f"_t{trend_window}"
                            ),
                            family="trend_gated_residual_leadership",
                            novelty="medium",
                            existing_overlap="Post-holdout repair of residual momentum using a familiar trend gate.",
                            hypothesis="A trend gate may preserve residual leadership while reducing break losses.",
                            target_weights=weights,
                            research_wave=2,
                            selection_eligible=False,
                        )
                    )
    return candidates


def _expanded_correlation_candidates(
    prices: pd.DataFrame,
    _: pd.DataFrame,
) -> list[Candidate]:
    defense = ["IEF", "TIP", "GLD", "UUP"]
    returns = prices[["SPY", *defense]].pct_change(fill_method=None)
    candidates: list[Candidate] = []
    for correlation_window in (63, 126):
        correlations = pd.DataFrame(
            {
                asset: returns["SPY"].rolling(correlation_window).corr(returns[asset])
                for asset in defense
            }
        )
        for trend_window in (126, 200):
            trends = prices[["SPY", *defense]].div(
                prices[["SPY", *defense]].rolling(trend_window).mean()
            ) - 1.0
            for risk_weight in (0.4, 0.6):
                weights = pd.DataFrame(
                    0.0,
                    index=prices.index,
                    columns=["SPY", *defense, "BIL"],
                )
                risk_on = trends["SPY"] > 0.0
                weights.loc[risk_on, "SPY"] = risk_weight
                defense_scores = (-correlations + trends[defense]).where(
                    trends[defense] > 0.0
                )
                best_defense = defense_scores.fillna(-np.inf).idxmax(axis=1)
                any_defense = defense_scores.notna().any(axis=1)
                for asset in defense:
                    choose = best_defense.eq(asset) & any_defense
                    weights.loc[choose, asset] = 1.0 - weights.loc[choose, "SPY"]
                weights["BIL"] = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
                weights = _scheduled_weights(weights, "W-FRI")
                candidates.append(
                    Candidate(
                        name=(
                            f"expanded_corrdef_c{correlation_window}_t{trend_window}"
                            f"_r{int(risk_weight * 100)}"
                        ),
                        family="expanded_correlation_defense",
                        novelty="medium_high",
                        existing_overlap="Post-holdout expansion to inflation and dollar defenses.",
                        hypothesis="A wider hedge set may avoid relying on duration during inflation shocks.",
                        target_weights=weights,
                        research_wave=2,
                        selection_eligible=False,
                    )
                )
    return candidates


def _cross_asset_calendar_candidates(
    prices: pd.DataFrame,
    _: pd.DataFrame,
) -> list[Candidate]:
    index = prices.index
    month_position = pd.Series(
        index.to_series().groupby(index.to_period("M")).cumcount(),
        index=index,
    )
    month_size = pd.Series(
        index.to_series().groupby(index.to_period("M")).transform("size"),
        index=index,
    )
    from_end = month_size - month_position - 1
    candidates: list[Candidate] = []
    for before, after in ((1, 3), (3, 3), (5, 5)):
        active = (from_end <= before) | (month_position <= after)
        for asset in ("RSP", "QUAL", "IWM"):
            weights = _binary_risk_weights(index, active, risk_ticker=asset)
            candidates.append(
                Candidate(
                    name=f"calendar_{asset.lower()}_b{before}_a{after}",
                    family="cross_asset_calendar",
                    novelty="high",
                    existing_overlap="Post-holdout calendar test outside cap-weighted SPY.",
                    hypothesis="If month-boundary flows are real, they should transfer across equity implementations.",
                    target_weights=weights,
                    research_wave=2,
                    selection_eligible=False,
                )
            )
    return candidates


def _ranked_weights(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_n: int,
    weighting: str,
) -> pd.DataFrame:
    selected = scores.gt(0.0) & scores.rank(axis=1, ascending=False, method="first").le(top_n)
    if weighting == "inverse_volatility":
        volatility = prices.pct_change(fill_method=None).rolling(63).std() * np.sqrt(252.0)
        raw = selected.astype(float).div(volatility.replace(0.0, np.nan))
    else:
        raw = selected.astype(float)
    return raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def _scheduled_weights(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    periods = weights.index.to_period(frequency)
    last_dates = pd.Series(weights.index, index=weights.index).groupby(periods).transform("max")
    scheduled = weights.loc[weights.index == last_dates]
    return scheduled.reindex(weights.index).ffill().fillna(0.0)


def _threshold_rebalance_weights(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    band: float,
) -> pd.DataFrame:
    tickers = list(allocation)
    returns = prices[tickers].pct_change(fill_method=None).fillna(0.0)
    target = pd.Series(allocation, dtype=float)
    weights = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    current = target.copy()
    ready_seen = False
    for date in prices.index:
        ready = bool(prices.loc[date, tickers].notna().all())
        if not ready:
            continue
        if not ready_seen:
            current = target.copy()
            ready_seen = True
        else:
            current = current * (1.0 + returns.loc[date])
            current = current / current.sum()
            if bool((current - target).abs().gt(band).any()):
                current = target.copy()
        weights.loc[date] = current
    return weights


def _adaptive_threshold_weights(
    prices: pd.DataFrame,
    allocation: dict[str, float],
    *,
    base_band: float,
    volatility_window: int,
    mode: str,
) -> pd.DataFrame:
    tickers = list(allocation)
    returns = prices[tickers].pct_change(fill_method=None).fillna(0.0)
    spy_volatility = (
        prices["SPY"].pct_change(fill_method=None).rolling(volatility_window).std()
        * np.sqrt(252.0)
    )
    stress = spy_volatility > spy_volatility.rolling(756, min_periods=252).quantile(0.75)
    band = pd.Series(base_band, index=prices.index)
    band.loc[stress.fillna(False)] = (
        base_band * 0.50 if mode == "narrow_stress" else base_band * 1.75
    )
    target = pd.Series(allocation, dtype=float)
    weights = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    current = target.copy()
    ready_seen = False
    for date in prices.index:
        ready = bool(prices.loc[date, tickers].notna().all())
        if not ready:
            continue
        if not ready_seen:
            current = target.copy()
            ready_seen = True
        else:
            current = current * (1.0 + returns.loc[date])
            current = current / current.sum()
            if bool((current - target).abs().gt(float(band.loc[date])).any()):
                current = target.copy()
        weights.loc[date] = current
    return weights


def _lagged_macro_panel(
    fred: pd.DataFrame,
    index: pd.DatetimeIndex,
    release_lag: int,
) -> pd.DataFrame:
    columns = ["INDPRO", "PAYEMS", "RSAFS", "CFNAI", "UNRATE", "CPIAUCSL", "PCEPI", "T10YIE"]
    panel = fred[columns].copy()
    slow = [column for column in columns if column != "T10YIE"]
    panel.loc[:, slow] = panel[slow].shift(release_lag)
    return panel.reindex(index).ffill()


def _macro_quadrant_weights(
    index: pd.DatetimeIndex,
    growth: pd.Series,
    inflation: pd.Series,
    *,
    growth_floor: float,
    inflation_floor: float,
    template: str,
) -> pd.DataFrame:
    weights = pd.DataFrame(
        0.0,
        index=index,
        columns=["SPY", "QQQ", "IEF", "TIP", "GLD", "DBC", "BIL"],
    )
    growth_up = growth >= growth_floor
    inflation_up = inflation >= inflation_floor
    if template == "balanced":
        allocations = {
            (True, False): {"SPY": 0.60, "QQQ": 0.20, "IEF": 0.20},
            (True, True): {"SPY": 0.45, "TIP": 0.20, "GLD": 0.15, "DBC": 0.20},
            (False, True): {"TIP": 0.30, "GLD": 0.30, "BIL": 0.40},
            (False, False): {"IEF": 0.50, "GLD": 0.20, "BIL": 0.30},
        }
    else:
        allocations = {
            (True, False): {"SPY": 0.50, "IEF": 0.25, "BIL": 0.25},
            (True, True): {"SPY": 0.30, "TIP": 0.25, "GLD": 0.20, "BIL": 0.25},
            (False, True): {"TIP": 0.25, "GLD": 0.25, "BIL": 0.50},
            (False, False): {"IEF": 0.40, "GLD": 0.20, "BIL": 0.40},
        }
    for state, allocation in allocations.items():
        mask = growth_up.eq(state[0]) & inflation_up.eq(state[1])
        for ticker, weight in allocation.items():
            weights.loc[mask, ticker] = weight
    weights.loc[weights.sum(axis=1).eq(0.0), "BIL"] = 1.0
    return weights


def _binary_risk_weights(
    index: pd.DatetimeIndex,
    active: pd.Series,
    *,
    risk_ticker: str,
) -> pd.DataFrame:
    active = active.reindex(index).fillna(False).astype(bool)
    weights = pd.DataFrame(0.0, index=index, columns=[risk_ticker, "BIL"])
    weights.loc[active, risk_ticker] = 1.0
    weights.loc[~active, "BIL"] = 1.0
    return weights


def _event_holding_mask(events: pd.Series, holding_days: int) -> pd.Series:
    event_values = events.fillna(False).to_numpy(dtype=bool)
    active = np.zeros(len(events), dtype=bool)
    remaining = 0
    for position, event in enumerate(event_values):
        if event:
            remaining = max(remaining, holding_days)
        if remaining > 0:
            active[position] = True
            remaining -= 1
    return pd.Series(active, index=events.index)


def _run_candidate(candidate: Candidate, prices: pd.DataFrame) -> BacktestResult:
    tickers = list(candidate.target_weights.columns)
    execution = ExecutionConfig(
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        rebalance="daily",
        signal_lag_days=2,
    )
    return run_backtest(
        candidate.name,
        prices[tickers],
        candidate.target_weights,
        execution,
    )


def _control_returns(
    prices: pd.DataFrame,
    *,
    config_path: Path,
    rotation_returns_path: Path | None,
    swing_returns_path: Path | None,
) -> dict[str, pd.Series]:
    controls: dict[str, pd.Series] = {
        "spy_buy_hold": prices["SPY"].pct_change(fill_method=None).fillna(0.0),
        "balanced_60_40": (
            0.60 * prices["SPY"].pct_change(fill_method=None)
            + 0.40 * prices["AGG"].pct_change(fill_method=None)
        ).fillna(0.0),
    }
    config = load_config(config_path)
    strategy = config.strategies[config.primary_strategy]
    tickers = required_strategy_tickers(strategy)
    strategy_prices = prices[tickers]
    weights = build_strategy_weights(strategy_prices, strategy)
    controls["trade_bot_primary"] = run_backtest(
        "trade_bot_primary",
        strategy_prices,
        weights,
        config.execution,
        volatility_target=strategy.volatility_target,
        drawdown_control=strategy.drawdown_control,
    ).returns
    if rotation_returns_path and rotation_returns_path.exists():
        rotation = pd.read_parquet(rotation_returns_path)
        for column in ("rotation_candidate", "regime_robust_challenger"):
            if column in rotation:
                controls[column] = rotation[column].astype(float)
    if swing_returns_path and swing_returns_path.exists():
        swing = pd.read_parquet(swing_returns_path)
        if "momentum_winner_12_2" in swing:
            controls["swing_momentum_factor"] = swing["momentum_winner_12_2"].astype(float)
    return controls


def _control_metric_frame(controls: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for control, returns in controls.items():
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "control": control,
                    "period": period,
                    **_metric_row(_period_slice(returns, start, end)),
                }
            )
    return pd.DataFrame(rows)


def _period_slice(
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.Series:
    sliced = returns.loc[returns.index >= start]
    if end is not None:
        sliced = sliced.loc[sliced.index <= end]
    return sliced.dropna()


def _metric_row(returns: pd.Series) -> dict[str, float | str | int]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "start": "",
            "end": "",
            "sessions": 0,
            "cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }
    growth = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    cagr = float(growth.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(returns.std() * np.sqrt(252.0))
    sharpe = float(returns.mean() * 252.0 / volatility) if volatility > 1e-12 else 0.0
    drawdown = growth / growth.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < -1e-12 else 0.0
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "sessions": len(returns),
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def _state_rows(
    candidate: str,
    family: str,
    returns: pd.Series,
    states: pd.Series,
) -> list[dict[str, object]]:
    aligned = pd.concat([returns.rename("return"), states], axis=1).dropna()
    aligned = aligned.loc[aligned.index >= COMMON_START]
    rows: list[dict[str, object]] = []
    for state, group in aligned.groupby("market_state"):
        rows.append(
            {
                "candidate": candidate,
                "family": family,
                "market_state": state,
                "sessions": len(group),
                "annualized_arithmetic_return": float(group["return"].mean() * 252.0),
                "annualized_volatility": float(group["return"].std() * np.sqrt(252.0)),
            }
        )
    return rows


def _rolling_rows(
    candidate: str,
    family: str,
    returns: pd.Series,
    spy: pd.Series,
) -> list[dict[str, object]]:
    aligned = pd.concat([returns.rename("candidate"), spy.rename("benchmark")], axis=1).dropna()
    aligned = aligned.loc[aligned.index >= COMMON_START]
    rows: list[dict[str, object]] = []
    for window in (252, 756):
        candidate_growth = (1.0 + aligned["candidate"]).rolling(window).apply(np.prod, raw=True)
        benchmark_growth = (1.0 + aligned["benchmark"]).rolling(window).apply(np.prod, raw=True)
        excess = candidate_growth - benchmark_growth
        clean = excess.dropna()
        rows.append(
            {
                "candidate": candidate,
                "family": family,
                "window_sessions": window,
                "windows": len(clean),
                "beat_share": float(clean.gt(0.0).mean()) if len(clean) else np.nan,
                "median_excess_growth": float(clean.median()) if len(clean) else np.nan,
                "latest_excess_growth": float(clean.iloc[-1]) if len(clean) else np.nan,
            }
        )
    return rows


def _correlation_rows(
    candidate: str,
    family: str,
    returns: pd.Series,
    controls: dict[str, pd.Series],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control_name, control in controls.items():
        aligned = pd.concat([returns.rename("candidate"), control.rename("control")], axis=1).dropna()
        aligned = aligned.loc[aligned.index >= COMMON_START]
        rows.append(
            {
                "candidate": candidate,
                "family": family,
                "control": control_name,
                "sessions": len(aligned),
                "correlation": float(aligned.corr().iloc[0, 1]) if len(aligned) > 2 else np.nan,
            }
        )
    return rows


def _blend_rows(
    candidate: str,
    family: str,
    returns: pd.Series,
    controls: dict[str, pd.Series],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control_name in ("trade_bot_primary", "rotation_candidate"):
        if control_name not in controls:
            continue
        aligned = pd.concat(
            [returns.rename("candidate"), controls[control_name].rename("control")],
            axis=1,
        ).dropna()
        aligned = aligned.loc[aligned.index >= COMMON_START]
        for period, (start, end) in PERIODS.items():
            label = "common_full" if period == "common_full" else period
            period_data = aligned.loc[aligned.index >= start]
            if end is not None:
                period_data = period_data.loc[period_data.index <= end]
            blend = 0.25 * period_data["candidate"] + 0.75 * period_data["control"]
            blend_metrics = _metric_row(blend)
            control_metrics = _metric_row(period_data["control"])
            rows.append(
                {
                    "candidate": candidate,
                    "family": family,
                    "control": control_name,
                    "period": label,
                    "blend_cagr": blend_metrics["cagr"],
                    "control_cagr": control_metrics["cagr"],
                    "cagr_delta": float(blend_metrics["cagr"]) - float(control_metrics["cagr"]),
                    "blend_sharpe": blend_metrics["sharpe"],
                    "control_sharpe": control_metrics["sharpe"],
                    "sharpe_delta": float(blend_metrics["sharpe"])
                    - float(control_metrics["sharpe"]),
                    "blend_max_drawdown": blend_metrics["max_drawdown"],
                    "control_max_drawdown": control_metrics["max_drawdown"],
                    "drawdown_delta": float(blend_metrics["max_drawdown"])
                    - float(control_metrics["max_drawdown"]),
                }
            )
    return rows


def _sleeve_sizing_evidence(
    candidate_results: dict[str, BacktestResult],
    metadata: dict[str, Candidate],
    controls: dict[str, pd.Series],
) -> pd.DataFrame:
    if "trade_bot_primary" not in controls:
        return pd.DataFrame()
    control = controls["trade_bot_primary"]
    rows: list[dict[str, object]] = []
    for name, result in candidate_results.items():
        aligned = pd.concat(
            [result.returns.rename("candidate"), control.rename("control")],
            axis=1,
        ).dropna()
        aligned = aligned.loc[aligned.index >= COMMON_START]
        for period in ("common_full", "holdout", "forward"):
            start, end = PERIODS[period]
            period_data = aligned.loc[aligned.index >= start]
            if end is not None:
                period_data = period_data.loc[period_data.index <= end]
            control_metrics = _metric_row(period_data["control"])
            for sleeve_weight in (0.05, 0.10, 0.15, 0.20, 0.25):
                blended = (
                    sleeve_weight * period_data["candidate"]
                    + (1.0 - sleeve_weight) * period_data["control"]
                )
                blend_metrics = _metric_row(blended)
                cagr_delta = float(blend_metrics["cagr"]) - float(control_metrics["cagr"])
                sharpe_delta = float(blend_metrics["sharpe"]) - float(
                    control_metrics["sharpe"]
                )
                drawdown_delta = float(blend_metrics["max_drawdown"]) - float(
                    control_metrics["max_drawdown"]
                )
                rows.append(
                    {
                        "candidate": name,
                        "family": metadata[name].family,
                        "research_wave": metadata[name].research_wave,
                        "selection_eligible": metadata[name].selection_eligible,
                        "period": period,
                        "sleeve_weight": sleeve_weight,
                        "blend_cagr": blend_metrics["cagr"],
                        "blend_sharpe": blend_metrics["sharpe"],
                        "blend_max_drawdown": blend_metrics["max_drawdown"],
                        "cagr_delta": cagr_delta,
                        "sharpe_delta": sharpe_delta,
                        "drawdown_delta": drawdown_delta,
                        "utility_score": (
                            sharpe_delta
                            + drawdown_delta
                            - 2.0 * max(0.0, -cagr_delta - 0.02)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _select_families(
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    correlations: pd.DataFrame,
    blends: pd.DataFrame,
) -> pd.DataFrame:
    eligible_metrics = metrics.loc[metrics["selection_eligible"].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    for candidate, candidate_rows in eligible_metrics.groupby("candidate"):
        source = candidate_rows.iloc[0]
        development = candidate_rows.loc[candidate_rows["period"].eq("development")].iloc[0]
        validation = candidate_rows.loc[candidate_rows["period"].eq("validation")].iloc[0]
        score = (
            0.30 * float(development["sharpe"])
            + 0.30 * float(validation["sharpe"])
            + 0.20 * float(development["calmar"])
            + 0.20 * float(validation["calmar"])
        )
        if min(float(development["cagr"]), float(validation["cagr"])) < 0.0:
            score -= 0.50
        rows.append(
            {
                "candidate": candidate,
                "family": source["family"],
                "novelty": source["novelty"],
                "existing_overlap": source["existing_overlap"],
                "preholdout_score": score,
            }
        )
    scored = pd.DataFrame(rows)
    winners = (
        scored.sort_values(["family", "preholdout_score"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .copy()
    )
    for period in PERIODS:
        period_metrics = metrics.loc[metrics["period"].eq(period)].set_index("candidate")
        for column in ("cagr", "sharpe", "max_drawdown", "calmar"):
            winners[f"{period}_{column}"] = winners["candidate"].map(period_metrics[column])
    roll = rolling.pivot(index="candidate", columns="window_sessions", values="beat_share")
    winners["rolling_1y_beat_spy"] = winners["candidate"].map(roll.get(252, pd.Series(dtype=float)))
    winners["rolling_3y_beat_spy"] = winners["candidate"].map(roll.get(756, pd.Series(dtype=float)))
    trade_corr = correlations.loc[correlations["control"].eq("trade_bot_primary")].set_index("candidate")
    rotation_corr = correlations.loc[correlations["control"].eq("rotation_candidate")].set_index("candidate")
    winners["correlation_trade_bot"] = winners["candidate"].map(trade_corr["correlation"])
    winners["correlation_rotation"] = winners["candidate"].map(rotation_corr["correlation"])
    blend = blends.loc[
        blends["control"].eq("trade_bot_primary") & blends["period"].eq("common_full")
    ].set_index("candidate")
    for column in ("cagr_delta", "sharpe_delta", "drawdown_delta"):
        winners[f"trade_bot_blend_{column}"] = winners["candidate"].map(blend[column])
    winners["research_read"] = winners.apply(_research_read, axis=1)
    return winners.sort_values("preholdout_score", ascending=False).reset_index(drop=True)


def _post_holdout_diagnostics(
    metrics: pd.DataFrame,
    rolling: pd.DataFrame,
    correlations: pd.DataFrame,
    blends: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = metrics.loc[~metrics["selection_eligible"].astype(bool)].copy()
    if diagnostics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for candidate, candidate_rows in diagnostics.groupby("candidate"):
        source = candidate_rows.iloc[0]
        common = candidate_rows.loc[candidate_rows["period"].eq("common_full")].iloc[0]
        holdout = candidate_rows.loc[candidate_rows["period"].eq("holdout")].iloc[0]
        forward = candidate_rows.loc[candidate_rows["period"].eq("forward")].iloc[0]
        score = (
            0.35 * float(common["sharpe"])
            + 0.20 * float(common["calmar"])
            + 0.20 * float(holdout["sharpe"])
            + 0.25 * float(forward["sharpe"])
        )
        rows.append(
            {
                "candidate": candidate,
                "family": source["family"],
                "novelty": source["novelty"],
                "diagnostic_score": score,
                "common_full_cagr": common["cagr"],
                "common_full_max_drawdown": common["max_drawdown"],
                "holdout_cagr": holdout["cagr"],
                "forward_cagr": forward["cagr"],
            }
        )
    summary = pd.DataFrame(rows)
    winners = (
        summary.sort_values(["family", "diagnostic_score"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .copy()
    )
    roll = rolling.pivot(index="candidate", columns="window_sessions", values="beat_share")
    winners["rolling_3y_beat_spy"] = winners["candidate"].map(
        roll.get(756, pd.Series(dtype=float))
    )
    trade_corr = correlations.loc[
        correlations["control"].eq("trade_bot_primary")
    ].set_index("candidate")
    winners["correlation_trade_bot"] = winners["candidate"].map(
        trade_corr["correlation"]
    )
    blend = blends.loc[
        blends["control"].eq("trade_bot_primary")
        & blends["period"].eq("common_full")
    ].set_index("candidate")
    for column in ("cagr_delta", "sharpe_delta", "drawdown_delta"):
        winners[f"trade_bot_blend_{column}"] = winners["candidate"].map(blend[column])
    winners["research_status"] = "post_holdout_diagnostic_only"
    return winners.sort_values("diagnostic_score", ascending=False).reset_index(drop=True)


def _research_read(row: pd.Series) -> str:
    forward = float(row.get("forward_cagr", np.nan))
    holdout = float(row.get("holdout_cagr", np.nan))
    blend_sharpe = float(row.get("trade_bot_blend_sharpe_delta", np.nan))
    blend_cagr = float(row.get("trade_bot_blend_cagr_delta", np.nan))
    blend_drawdown = float(row.get("trade_bot_blend_drawdown_delta", np.nan))
    rolling = float(row.get("rolling_3y_beat_spy", np.nan))
    correlation = float(row.get("correlation_trade_bot", np.nan))
    if (
        holdout > 0.0
        and forward > 0.0
        and blend_sharpe > 0.03
        and blend_cagr > -0.015
        and correlation < 0.65
    ):
        return "promising_incremental"
    if rolling >= 0.50 and holdout > 0.0 and forward > 0.0:
        return "promising_standalone"
    if correlation >= 0.80:
        return "duplicative_behavior"
    if blend_sharpe > 0.03 and blend_drawdown > 0.02 and blend_cagr <= -0.015:
        return "risk_diversifier_with_return_drag"
    if holdout <= 0.0 or forward <= 0.0:
        return "failed_out_of_sample"
    return "diagnostic_only"


def _data_frontiers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frontier": "sec_fundamental_change",
                "orthogonality": "high",
                "current_test_status": "not_tested",
                "required_data": "Point-in-time SEC XBRL facts, filing timestamps, delisted securities, and historical universe membership.",
                "reason": "ETF factor proxies cannot establish stock-level fundamental alpha.",
            },
            {
                "frontier": "corporate_event_drift",
                "orthogonality": "high",
                "current_test_status": "not_tested",
                "required_data": "Point-in-time earnings, 8-K, Form 4, 13D/13G, spin-off, and buyback event calendar.",
                "reason": "Price gaps are not a valid substitute for knowing the event at the time.",
            },
            {
                "frontier": "cftc_positioning",
                "orthogonality": "high",
                "current_test_status": "not_tested",
                "required_data": "Vintage-aware CFTC Commitments of Traders history mapped to executable ETFs.",
                "reason": "Positioning can provide non-price crowding information without trading futures.",
            },
            {
                "frontier": "options_implied_state",
                "orthogonality": "medium_high",
                "current_test_status": "not_tested",
                "required_data": "Historical VIX term structure, SKEW, put-call, and implied-versus-realized volatility.",
                "reason": "Options may be used as signals while trades remain long-only ETFs and cash.",
            },
            {
                "frontier": "fund_flows_and_positioning",
                "orthogonality": "medium_high",
                "current_test_status": "not_tested",
                "required_data": "Point-in-time ETF flows, short interest, margin, and fund positioning with stable identifiers.",
                "reason": "Current systems infer crowding mostly from prices and concentration.",
            },
        ]
    )


def _render_report(
    selection: pd.DataFrame,
    post_holdout: pd.DataFrame,
    sleeve_sizing: pd.DataFrame,
    controls: pd.DataFrame,
    states: pd.DataFrame,
    frontiers: pd.DataFrame,
    *,
    variants_tested: int,
    data_end: pd.Timestamp,
) -> str:
    family_count = len(set(selection["family"]) | set(post_holdout.get("family", [])))
    lines = [
        "# Orthogonal Strategy Frontier",
        "",
        f"Evidence through {data_end.date()}; {variants_tested} variants across {family_count} families.",
        "",
        "Family winners were selected from development (2014-2018) and validation (2019-2021) only. "
        "The 2022-2024 holdout and 2025+ forward periods were read after selection.",
        "",
        "## Family Winners",
        "",
        "| family | winner | novelty | holdout CAGR | forward CAGR | 3y beat SPY | corr Trade Bot | 25% blend Sharpe delta | read |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in selection.iterrows():
        lines.append(
            "| {family} | `{candidate}` | {novelty} | {holdout:.1%} | {forward:.1%} | "
            "{rolling:.1%} | {correlation:.2f} | {blend:+.2f} | {read} |".format(
                family=row["family"],
                candidate=row["candidate"],
                novelty=row["novelty"],
                holdout=float(row["holdout_cagr"]),
                forward=float(row["forward_cagr"]),
                rolling=float(row["rolling_3y_beat_spy"]),
                correlation=float(row["correlation_trade_bot"]),
                blend=float(row["trade_bot_blend_sharpe_delta"]),
                read=row["research_read"],
            )
        )
    lines.extend(
        [
            "",
            "## Same-Period Controls",
            "",
            "| control | common CAGR | Sharpe | max DD | holdout CAGR | forward CAGR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    control_pivot = controls.pivot(index="control", columns="period", values=["cagr", "sharpe", "max_drawdown"])
    for control in control_pivot.index:
        lines.append(
            "| {control} | {common:.1%} | {sharpe:.2f} | {drawdown:.1%} | {holdout:.1%} | {forward:.1%} |".format(
                control=control,
                common=float(control_pivot.loc[control, ("cagr", "common_full")]),
                sharpe=float(control_pivot.loc[control, ("sharpe", "common_full")]),
                drawdown=float(
                    control_pivot.loc[control, ("max_drawdown", "common_full")]
                ),
                holdout=float(control_pivot.loc[control, ("cagr", "holdout")]),
                forward=float(control_pivot.loc[control, ("cagr", "forward")]),
            )
        )
    if not sleeve_sizing.empty:
        common_sleeves = sleeve_sizing.loc[sleeve_sizing["period"].eq("common_full")]
        top_sleeves = common_sleeves.sort_values("utility_score", ascending=False).head(8)
        lines.extend(
            [
                "",
                "## Trade Bot Sleeve Utility",
                "",
                "This is portfolio construction evidence, not standalone strategy promotion. Post-holdout rows retain diagnostic-only status.",
                "",
                "| candidate | sleeve | blend CAGR | CAGR delta | Sharpe delta | max-DD improvement | wave |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in top_sleeves.iterrows():
            lines.append(
                "| `{candidate}` | {weight:.0%} | {cagr:.1%} | {cagr_delta:+.1%} | "
                "{sharpe_delta:+.2f} | {drawdown_delta:+.1%} | {wave} |".format(
                    candidate=row["candidate"],
                    weight=float(row["sleeve_weight"]),
                    cagr=float(row["blend_cagr"]),
                    cagr_delta=float(row["cagr_delta"]),
                    sharpe_delta=float(row["sharpe_delta"]),
                    drawdown_delta=float(row["drawdown_delta"]),
                    wave=int(row["research_wave"]),
                )
            )
    if not post_holdout.empty:
        lines.extend(
            [
                "",
                "## Post-Holdout Iteration",
                "",
                "These variants were designed after the first holdout read. They are diagnostics only and cannot replace the frozen family winners.",
                "",
                "| family | diagnostic winner | full CAGR | max DD | holdout CAGR | forward CAGR | 3y beat SPY | corr Trade Bot | 25% blend CAGR delta |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in post_holdout.iterrows():
            lines.append(
                "| {family} | `{candidate}` | {full:.1%} | {drawdown:.1%} | {holdout:.1%} | "
                "{forward:.1%} | {rolling:.1%} | {correlation:.2f} | {blend:+.1%} |".format(
                    family=row["family"],
                    candidate=row["candidate"],
                    full=float(row["common_full_cagr"]),
                    drawdown=float(row["common_full_max_drawdown"]),
                    holdout=float(row["holdout_cagr"]),
                    forward=float(row["forward_cagr"]),
                    rolling=float(row["rolling_3y_beat_spy"]),
                    correlation=float(row["correlation_trade_bot"]),
                    blend=float(row["trade_bot_blend_cagr_delta"]),
                )
            )
    lines.extend(
        [
            "",
            "## Regime-Break Behavior",
            "",
            "Annualized arithmetic returns on non-contiguous break sessions are diagnostic, not CAGRs.",
            "",
            "| family | selected candidate | break return | transition return | normal return |",
            "|---|---|---:|---:|---:|",
        ]
    )
    state_pivot = states.pivot(index="candidate", columns="market_state", values="annualized_arithmetic_return")
    for _, row in selection.iterrows():
        candidate = row["candidate"]
        values = state_pivot.loc[candidate] if candidate in state_pivot.index else pd.Series(dtype=float)
        lines.append(
            f"| {row['family']} | `{candidate}` | {float(values.get('break', np.nan)):.1%} | "
            f"{float(values.get('transition', np.nan)):.1%} | {float(values.get('normal', np.nan)):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Still-Unturned Data Frontiers",
            "",
            "These were not replaced with price-only proxies because that would answer a different question.",
            "",
            "| frontier | orthogonality | status | why it remains open |",
            "|---|---|---|---|",
        ]
    )
    for _, row in frontiers.iterrows():
        lines.append(
            f"| {row['frontier']} | {row['orthogonality']} | {row['current_test_status']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `cash_hurdle_cross_asset` and `dislocation_recovery` are extensions of existing Trade Bot mechanisms, not new systems.",
            "- Macro tests use release lags but revised FRED history, so they cannot earn production authority.",
            "- Structural factor ETFs are feasibility proxies, not tests of point-in-time company fundamentals.",
            "- No family is promoted from this survey. A new repository is justified only for a high-orthogonality mechanism that survives an independent data build and untouched holdout.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(path).sort_index()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index)).tz_localize(None)
    return prices.loc[~prices.index.duplicated(keep="last")]


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index().sort_index(axis=1)
    digest = hashlib.sha256()
    digest.update("|".join(map(str, ordered.columns)).encode())
    digest.update(pd.util.hash_pandas_object(ordered, index=True).values.tobytes())
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the orthogonal strategy frontier survey.")
    parser.add_argument("--prices", default="data/cache/yahoo_prices.parquet")
    parser.add_argument("--fred", default="data/cache/fred_series.parquet")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--rotation-returns")
    parser.add_argument("--swing-returns")
    parser.add_argument("--output", default="reports/orthogonal_frontier")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_orthogonal_frontier_survey(
        prices_path=args.prices,
        fred_path=args.fred,
        config_path=args.config,
        rotation_returns_path=args.rotation_returns,
        swing_returns_path=args.swing_returns,
        output_dir=args.output,
    )
    print(result.report_path)


if __name__ == "__main__":
    main()
