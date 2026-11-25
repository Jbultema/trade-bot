"""Fixed-slate adaptive response research; retrospective and zero authority."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.backtest.metrics import calculate_metrics
from trade_bot.config import BotConfig, ExecutionConfig, StrategyConfig, load_config
from trade_bot.features.indicators import bounded_forward_fill, daily_returns
from trade_bot.research.artifact_provenance import write_research_manifest
from trade_bot.research.backtest_pbo import estimate_probability_of_backtest_overfitting
from trade_bot.storage.run_store import RunStore
from trade_bot.strategies.momentum import build_strategy_weights

SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
DEFENSIVE_SECTORS = ("XLV", "XLP", "XLU")
ALTERNATIVES = ("XLV", "XLP", "XLU", "XLF", "XLI", "XLB", "XLE")
FAMILIES = (
    "adaptive_blend",
    "change_detection",
    "fast_protection",
    "early_replacement",
    "recovery",
)
PERIODS = {
    "development": (None, "2011-12-31"),
    "validation": ("2012-01-01", "2018-12-31"),
    "historical_test": ("2019-01-01", "2023-12-31"),
    "recent_diagnostic": ("2024-01-01", None),
    "current_transition": ("2026-06-01", None),
    "full": (None, None),
}
EPISODES = {
    "dotcom": ("2000-03-01", "2003-03-31"),
    "gfc": ("2007-07-01", "2009-06-30"),
    "china_oil": ("2015-06-01", "2016-06-30"),
    "q4_2018": ("2018-09-01", "2019-03-31"),
    "covid": ("2020-02-01", "2020-08-31"),
    "rates_2022": ("2022-01-01", "2023-06-30"),
    "tariff_2025": ("2025-02-01", "2025-08-31"),
    "current_unresolved": ("2026-06-01", None),
}


@dataclass(frozen=True)
class Policy:
    name: str
    family: str
    round: int = 1
    parent: str = ""
    selection: str = "slow"
    fast_days: int = 21
    slow_days: int = 63
    max_fast: float = 0.75
    stress_days: int = 21
    cusum_threshold: float = 6.0
    cusum_drift: float = 0.25
    cusum_decay: float = 1.0
    protection: str = "none"
    reduction: float = 0.50
    replacement: str = "none"
    replacement_days: int = 21
    replacement_share: float = 0.50
    destination_top_n: int = 2
    exit_days: int = 2
    release_days: int = 3
    release_threshold: float = 0.30
    reentry_step: float = 0.20
    require_breadth: bool = False
    static_scale: float = 1.0
    cadence: str = "weekly"
    emergency_daily: bool = True


@dataclass
class Context:
    name: str
    prices: pd.DataFrame
    strategy: StrategyConfig
    benchmark: str
    evaluation_start: pd.Timestamp
    source_path: str
    source_sha256: str


class Features:
    def __init__(self, context: Context):
        self.context = context
        self.prices = bounded_forward_fill(context.prices)
        self._weights: dict[int, pd.DataFrame] = {}
        self._stress: dict[int, pd.Series] = {}
        self._destinations: dict[tuple[str, int, int], pd.DataFrame] = {}

    def weights(self, days: int) -> pd.DataFrame:
        if days not in self._weights:
            strategy = self.context.strategy.model_copy(
                update={"lookback_days": days, "defensive_ticker": "CASH"}
            )
            self._weights[days] = build_strategy_weights(self.context.prices, strategy)
        return self._weights[days].copy()

    def stress(self, days: int) -> pd.Series:
        if days not in self._stress:
            benchmark = self.prices[self.context.benchmark]
            returns = benchmark.pct_change(fill_method=None)
            loss = (-benchmark.pct_change(days, fill_method=None) / 0.08).clip(0, 1)
            drawdown = (-(benchmark / benchmark.rolling(63).max() - 1) / 0.12).clip(0, 1)
            trend = benchmark.lt(benchmark.rolling(days).mean()).astype(float)
            vol_ratio = returns.rolling(10).std().div(returns.rolling(63).std())
            vol = ((vol_ratio - 1.0) / 0.75).clip(0, 1)
            persistence = returns.lt(0).rolling(10).mean().sub(0.40).div(0.35).clip(0, 1)
            self._stress[days] = (
                (0.30 * loss + 0.20 * drawdown + 0.20 * trend + 0.15 * vol + 0.15 * persistence)
                .fillna(0)
                .clip(0, 1)
            )
        return self._stress[days]

    def destination(self, kind: str, days: int, top_n: int) -> pd.DataFrame:
        key = kind, days, top_n
        if key not in self._destinations:
            tickers = DEFENSIVE_SECTORS if kind == "defensive" else ALTERNATIVES
            available = [ticker for ticker in tickers if ticker in self.prices]
            prices = self.prices[available]
            momentum = prices.pct_change(days, fill_method=None)
            relative = prices.div(self.prices["SPY"], axis=0).pct_change(days, fill_method=None)
            old_rank = prices.pct_change(63, fill_method=None).rank(axis=1, ascending=False)
            velocity = old_rank.shift(21).sub(old_rank).div(max(len(available), 1))
            vol = prices.pct_change(fill_method=None).rolling(21).std() * np.sqrt(252)
            score = momentum.div(vol.where(vol > 0)) + relative + 0.10 * velocity
            eligible = momentum.gt(0) & prices.gt(prices.rolling(21).mean())
            score = score.where(eligible & relative.gt(0)).clip(lower=0)
            chosen = score.rank(axis=1, ascending=False, method="first").le(top_n)
            raw = score.where(chosen, 0).fillna(0)
            self._destinations[key] = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        return self._destinations[key].copy()


def downside_cusum(prices: pd.Series, *, drift: float = 0.25, decay: float = 1.0) -> pd.Series:
    """One-sided standardized CUSUM using moments available before each return."""
    returns = prices.pct_change(fill_method=None)
    mean = returns.rolling(63).mean().shift(1)
    vol = returns.rolling(63).std().shift(1).clip(lower=0.003)
    surprises = returns.sub(mean).div(vol).fillna(0).to_numpy()
    values = np.zeros(len(prices))
    state = 0.0
    for position, surprise in enumerate(surprises):
        state = max(0.0, decay * state - float(surprise) - drift)
        values[position] = state
    return pd.Series(values, index=prices.index)


def recovery_exposure(stress: pd.Series, repair: pd.Series, policy: Policy) -> pd.Series:
    values = np.ones(len(stress))
    active = False
    bad_days = good_days = 0
    exposure = 1.0
    for position, (pressure, repaired) in enumerate(zip(stress, repair, strict=True)):
        bad_days = bad_days + 1 if pressure >= 0.50 else 0
        good_days = good_days + 1 if pressure <= policy.release_threshold and repaired else 0
        if bad_days >= policy.exit_days:
            active = True
            good_days = 0
        if active and good_days >= policy.release_days:
            active = False
        if active:
            exposure = min(exposure, 1.0 - policy.reduction)
        else:
            exposure = min(1.0, exposure + policy.reentry_step)
        values[position] = exposure
    return pd.Series(values, index=stress.index)


def _fund_cash(weights: pd.DataFrame, max_weight: float | None = None) -> pd.DataFrame:
    output = weights.clip(lower=0).copy()
    risky = [column for column in output if column != "CASH"]
    if max_weight is not None:
        output.loc[:, risky] = output[risky].clip(upper=max_weight)
    total = output[risky].sum(axis=1)
    scale = total.where(total > 1, 1)
    output.loc[:, risky] = output[risky].div(scale, axis=0)
    output["CASH"] = (1.0 - output[risky].sum(axis=1)).clip(lower=0)
    return output.fillna(0)


def schedule_targets(weights: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Keep actual weekday decisions; never invent a last-partial-week rebalance."""
    result = weights.where(mask, np.nan, axis=0).ffill().fillna(0)
    return _fund_cash(result)


def build_policy_targets(
    features: Features, policy: Policy, *, weekday: int = 2, force_daily: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = features.context
    prices = features.prices
    stress = features.stress(policy.stress_days)
    slow = features.weights(policy.slow_days)
    alpha = pd.Series(0.0, index=prices.index)
    if policy.selection == "fast":
        alpha[:] = 1.0
    elif policy.selection == "constant_blend":
        alpha[:] = policy.max_fast
    elif policy.selection == "adaptive":
        alpha = stress.mul(policy.max_fast)
    elif policy.selection == "cusum":
        alpha = (
            downside_cusum(
                prices[context.benchmark], drift=policy.cusum_drift, decay=policy.cusum_decay
            )
            .div(policy.cusum_threshold)
            .clip(0, 1)
            .mul(policy.max_fast)
        )
    if policy.selection == "buy_spy":
        source = slow * 0.0
        source["SPY"] = 1.0
    elif policy.selection == "buy_qqq":
        source = slow * 0.0
        source["QQQ"] = 1.0
    elif policy.selection == "cash":
        source = slow * 0.0
    elif policy.selection == "equal":
        source = slow * 0.0
        for ticker in context.strategy.tickers:
            source[ticker] = prices[ticker].notna().astype(float) / len(context.strategy.tickers)
    else:
        fast = features.weights(policy.fast_days)
        source = slow.mul(1.0 - alpha, axis=0).add(fast.mul(alpha, axis=0))
    source = _fund_cash(source)
    risky = [column for column in source if column != "CASH"]
    scheduled = pd.Series(prices.index.weekday == weekday, index=prices.index)
    slow_cadence = schedule_targets(source, scheduled)
    if policy.cadence == "daily" or force_daily:
        scheduled[:] = True
    elif policy.cadence == "adaptive":
        scheduled |= stress.ge(0.40) | alpha.ge(0.40)
    if policy.protection == "continuous":
        exposure = 1.0 - policy.reduction * stress
    elif policy.protection == "recovery":
        benchmark = prices[context.benchmark]
        repair = benchmark.gt(benchmark.rolling(21).mean()) & benchmark.pct_change(5).gt(0)
        if policy.require_breadth:
            asset_prices = prices[context.strategy.tickers]
            repair &= asset_prices.gt(asset_prices.rolling(21).mean()).mean(axis=1).ge(0.5)
        exposure = recovery_exposure(stress, repair, policy)
    else:
        exposure = pd.Series(1.0, index=prices.index)
    if policy.protection != "none":
        # Protection responds daily while ordinary source selection retains its cadence.
        source = slow_cadence if policy.selection == "slow" and not force_daily else source
        if policy.emergency_daily:
            scheduled |= exposure.ne(exposure.shift(1))
    source.loc[:, risky] = source[risky].mul(exposure, axis=0)
    removed = pd.Series(0.0, index=prices.index)
    if policy.replacement != "none":
        eligible_prices = prices[risky]
        recent = eligible_prices.pct_change(21, fill_method=None)
        relative = eligible_prices.div(prices["SPY"], axis=0).pct_change(21, fill_method=None)
        weak = recent.lt(0) | relative.lt(-0.01)
        reductions = source[risky].where(weak, 0).mul(stress * policy.replacement_share, axis=0)
        source.loc[:, risky] = source[risky] - reductions
        removed = reductions.sum(axis=1)
        if policy.replacement != "cash":
            mix = features.destination(
                policy.replacement, policy.replacement_days, policy.destination_top_n
            )
            source.loc[:, mix.columns] = source[mix.columns].add(mix.mul(removed, axis=0))
        if policy.emergency_daily:
            scheduled |= stress.ge(0.40)
    source.loc[source.index < context.evaluation_start, risky] = 0.0
    cap = None if policy.family == "benchmark" else context.strategy.max_asset_weight
    source = _fund_cash(source, cap)
    targets = schedule_targets(source, scheduled)
    diagnostics = pd.DataFrame(
        {
            "stress": stress,
            "fast_weight": alpha,
            "exposure_multiplier": exposure,
            "replacement_budget": removed,
            "scheduled": scheduled,
            "target_risk": targets.drop(columns="CASH").sum(axis=1),
        },
        index=prices.index,
    )
    return targets, diagnostics


def initial_policies() -> list[Policy]:
    policies = []
    for days in (21, 42):
        for maximum in (0.60, 1.0):
            policies.append(
                Policy(
                    f"blend_f{days}_a{maximum:.1f}",
                    "adaptive_blend",
                    selection="adaptive",
                    fast_days=days,
                    max_fast=maximum,
                    cadence="adaptive",
                )
            )
    for days in (21, 42):
        for threshold in (4.0, 8.0):
            policies.append(
                Policy(
                    f"cusum_f{days}_h{threshold:.0f}",
                    "change_detection",
                    selection="cusum",
                    fast_days=days,
                    cusum_threshold=threshold,
                    cadence="adaptive",
                )
            )
    for days in (10, 21):
        for strength in (0.35, 0.60):
            policies.append(
                Policy(
                    f"protect_d{days}_r{strength:.2f}",
                    "fast_protection",
                    stress_days=days,
                    reduction=strength,
                    protection="continuous",
                )
            )
    for destination in ("defensive", "emerging"):
        for share in (0.35, 0.65):
            policies.append(
                Policy(
                    f"replace_{destination}_s{share:.2f}",
                    "early_replacement",
                    replacement=destination,
                    replacement_share=share,
                )
            )
    for exits in (1, 3):
        for release in (2, 5):
            policies.append(
                Policy(
                    f"recovery_exit{exits}_release{release}",
                    "recovery",
                    protection="recovery",
                    reduction=0.50,
                    exit_days=exits,
                    release_days=release,
                )
            )
    return policies


def controls() -> list[Policy]:
    return [
        Policy("slow_reference", "control", round=0),
        Policy("fixed_fast21", "control", round=0, selection="fast", fast_days=21),
        Policy("fixed_fast42", "control", round=0, selection="fast", fast_days=42),
        Policy("fixed_slow126", "control", round=0, slow_days=126),
        Policy("constant_blend50", "control", round=0, selection="constant_blend", max_fast=0.5),
        Policy("static75", "control", round=0, static_scale=0.75),
        Policy("static50", "control", round=0, static_scale=0.50),
        Policy("equal_universe", "benchmark", round=0, selection="equal", cadence="daily"),
        Policy("buy_hold_spy", "benchmark", round=0, selection="buy_spy", cadence="daily"),
        Policy("buy_hold_qqq", "benchmark", round=0, selection="buy_qqq", cadence="daily"),
        Policy("cash", "benchmark", round=0, selection="cash", cadence="daily"),
    ]


def followups(parent: Policy, round_number: int) -> list[Policy]:
    common = {"round": round_number, "parent": parent.name}
    if parent.family == "adaptive_blend":
        updates = (
            [{"max_fast": 0.35}, {"stress_days": 10}, {"cadence": "weekly"}]
            if round_number == 2
            else [{"fast_days": 10}, {"slow_days": 126}, {"max_fast": 0.85}]
        )
    elif parent.family == "change_detection":
        updates = (
            [{"cusum_decay": 0.97}, {"cusum_drift": 0.50}, {"max_fast": 0.50}]
            if round_number == 2
            else [{"cusum_threshold": 12.0}, {"fast_days": 10}, {"cadence": "weekly"}]
        )
    elif parent.family == "fast_protection":
        updates = (
            [{"reduction": 0.20}, {"reduction": 0.80}, {"stress_days": 5}]
            if round_number == 2
            else [{"stress_days": 42}, {"reduction": 0.45}, {"static_scale": 0.90}]
        )
    elif parent.family == "early_replacement":
        updates = (
            [{"replacement_days": 10}, {"replacement_days": 42}, {"destination_top_n": 3}]
            if round_number == 2
            else [{"replacement_share": 0.20}, {"replacement_share": 0.85}, {"replacement": "cash"}]
        )
    else:
        updates = (
            [{"reentry_step": 0.10}, {"reentry_step": 1.0}, {"require_breadth": True}]
            if round_number == 2
            else [
                {"release_threshold": 0.40},
                {"reduction": 0.35},
                {"exit_days": 2, "release_days": 3},
            ]
        )
    return [
        replace(parent, **common, **update, name=f"{parent.family}_r{round_number}_v{position}")
        for position, update in enumerate(updates, start=1)
    ]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contexts(config: BotConfig, rotation_root: Path) -> list[Context]:
    latest, manifest = RunStore(
        "data/run_store/trade_bot.duckdb", read_only=True
    ).load_latest_snapshot(require_matching_config=False)
    run_manifest = json.loads((rotation_root / "reports/latest/run_manifest.json").read_text())
    price_manifest = rotation_root / run_manifest["price_manifest"]
    meta = json.loads(price_manifest.read_text())
    price_path = price_manifest.parent / meta["data_file"]
    if _hash(price_path) != meta["data_file_sha256"]:
        raise ValueError("Rotation source failed archived price SHA256 verification")
    rotation = pd.read_parquet(price_path)["Close"].sort_index()
    end = min(rotation.index.max(), latest.prices.index.max())
    contexts = []
    primary = config.strategies[config.primary_strategy]
    for name, prices, tickers, benchmark, strategy, source in (
        (
            "technology",
            latest.prices.loc[:end].copy(),
            primary.tickers,
            "QQQ",
            primary,
            Path(manifest.artifact_path),
        ),
        (
            "sectors",
            rotation.loc[:end].copy(),
            list(SECTORS),
            "SPY",
            StrategyConfig(
                type="dual_momentum",
                tickers=list(SECTORS),
                lookback_days=63,
                skip_days=5,
                top_n=3,
                defensive_ticker="CASH",
                min_return=0.0,
                ranking_metric="risk_adjusted_return",
                weighting="risk_adjusted_score",
                max_asset_weight=0.35,
            ),
            price_path,
        ),
    ):
        annual_yield = bounded_forward_fill(rotation["^IRX"].reindex(prices.index)).shift(1)
        if annual_yield.iloc[1:].isna().any():
            raise ValueError(f"Missing lagged cash yield in {name}")
        cash_returns = annual_yield.clip(lower=0).div(100).add(1).pow(1 / 252).sub(1).fillna(0)
        prices["CASH"] = 100 * cash_returns.add(1).cumprod()
        cash_ticker = ["BIL"] if "BIL" in prices else []
        required = list(dict.fromkeys([*tickers, *SECTORS, "SPY", "QQQ", "CASH", *cash_ticker]))
        prices = prices[required]
        contexts.append(
            Context(
                name, prices, strategy, benchmark, prices.index[252], str(source), _hash(source)
            )
        )
    return contexts


def execute(
    context: Context,
    targets: pd.DataFrame,
    *,
    lag: int = 2,
    cost: float = 10.0,
    controlled: bool = True,
    risk_scale: float = 1.0,
) -> BacktestResult:
    result = run_backtest(
        "research",
        context.prices,
        targets,
        ExecutionConfig(rebalance="D", signal_lag_days=lag, transaction_cost_bps=cost),
        volatility_target=context.strategy.volatility_target if controlled else None,
        drawdown_control=context.strategy.drawdown_control if controlled else None,
    )
    # Value idle cash after the engine has fixed the lagged risk holdings and controls.
    weights = result.weights.copy()
    risky = [column for column in weights if column != "CASH"]
    weights.loc[:, risky] = weights[risky] * risk_scale
    weights = _fund_cash(weights)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * cost / 10000.0
    gross = (weights * daily_returns(context.prices)).sum(axis=1)
    net = gross - costs
    return replace(
        result,
        weights=weights,
        turnover=turnover,
        transaction_costs=costs,
        gross_returns=gross,
        returns=net,
        equity=100000 * net.add(1).cumprod(),
    )


def period_metrics(
    result: BacktestResult, start: str | pd.Timestamp | None, end: str | None
) -> dict[str, float]:
    returns = result.returns.loc[start:end]
    if len(returns) < 10:
        return {}
    equity = 100000 * returns.add(1).cumprod()
    measured = calculate_metrics(
        "research", returns, equity, result.turnover, result.transaction_costs
    )
    # Include the capital just before the first return in peak-to-trough calculations.
    augmented = np.r_[100000.0, equity.to_numpy()]
    dd = augmented / np.maximum.accumulate(augmented) - 1
    underwater = maximum_underwater = 0
    for value in dd:
        underwater = underwater + 1 if value < -1e-12 else 0
        maximum_underwater = max(maximum_underwater, underwater)
    threshold = returns.quantile(0.05)
    return {
        "cagr": measured.cagr,
        "max_drawdown": float(dd.min()),
        "sharpe": measured.sharpe,
        "volatility": measured.annualized_volatility,
        "total_return": float(returns.add(1).prod() - 1),
        "es95_daily": float(returns[returns <= threshold].mean()),
        "ulcer": float(np.sqrt(np.mean(dd**2))),
        "maximum_underwater_sessions": maximum_underwater,
        "average_risk": float(
            result.weights.drop(columns="CASH").sum(axis=1).reindex(returns.index).mean()
        ),
        "annual_turnover": float(result.turnover.reindex(returns.index).mean() * 252),
        "sessions": len(returns),
    }


def metrics_rows(context: Context, policy: Policy, result: BacktestResult) -> list[dict]:
    rows = []
    for period, (start, end) in PERIODS.items():
        start = (
            max(pd.Timestamp(start), context.evaluation_start)
            if start
            else context.evaluation_start
        )
        measured = period_metrics(result, start, end)
        if measured:
            rows.append(
                {
                    "context": context.name,
                    "candidate": policy.name,
                    "family": policy.family,
                    "round": policy.round,
                    "period": period,
                    **measured,
                }
            )
    return rows


def select_winners(metrics: pd.DataFrame) -> pd.DataFrame:
    train = metrics[metrics.period.isin(["development", "validation"])].copy()
    reference = train[train.candidate.eq("slow_reference")].set_index(["context", "period"])
    rows = []
    for (context, name, family), group in train[train.family.isin(FAMILIES)].groupby(
        ["context", "candidate", "family"]
    ):
        scores = []
        for row in group.itertuples():
            base = reference.loc[(context, row.period)]
            # Freeze a downside-sensitive score with a return/opportunity-cost penalty.
            scores.append(
                0.5 * (row.sharpe - base.sharpe)
                + 3 * (row.max_drawdown - base.max_drawdown)
                + 2 * (row.cagr - base.cagr)
                - 0.002 * max(0.0, row.annual_turnover - base.annual_turnover)
            )
        if len(scores) == 2:
            rows.append(
                {
                    "context": context,
                    "candidate": name,
                    "family": family,
                    "training_score": min(scores) + 0.25 * np.mean(scores),
                }
            )
    ranked = pd.DataFrame(rows).sort_values(
        ["training_score", "candidate"], ascending=[False, True]
    )
    return ranked.groupby(["context", "family"], sort=False).head(1).reset_index(drop=True)


def combination_policies(winners: dict[str, Policy]) -> list[Policy]:
    adaptive = winners["adaptive_blend"]
    detector = winners["change_detection"]
    protect = winners["fast_protection"]
    destination = winners["early_replacement"]
    recovery = winners["recovery"]
    # If cash won the destination experiment, retain that negative evidence in combinations.
    variants = []
    for index, (selection, defense, replacement) in enumerate(
        (
            (adaptive, protect, False),
            (detector, protect, False),
            (adaptive, protect, True),
            (adaptive, recovery, True),
            (detector, recovery, True),
            (winners["fast_protection"], protect, True),
        ),
        start=1,
    ):
        variants.append(
            replace(
                selection,
                name=f"combination_{index}",
                family="combination",
                round=4,
                parent=f"{selection.name}+{defense.name}"
                + (f"+{destination.name}" if replacement else ""),
                protection=defense.protection,
                reduction=defense.reduction,
                stress_days=defense.stress_days,
                exit_days=defense.exit_days,
                release_days=defense.release_days,
                release_threshold=defense.release_threshold,
                reentry_step=defense.reentry_step,
                require_breadth=defense.require_breadth,
                replacement=destination.replacement if replacement else "none",
                replacement_share=destination.replacement_share,
                replacement_days=destination.replacement_days,
                destination_top_n=destination.destination_top_n,
            )
        )
    return variants


def bootstrap_difference(
    candidate: pd.Series, reference: pd.Series, *, samples: int = 1000, block: int = 63
) -> dict[str, float]:
    pair = pd.concat([candidate, reference], axis=1).dropna()
    differences = np.log1p(pair.iloc[:, 0]).sub(np.log1p(pair.iloc[:, 1])).to_numpy()
    rng = np.random.default_rng(20261125)
    n = len(differences)
    offsets = np.arange(block)
    draws = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        positions = ((starts[:, None] + offsets) % n).ravel()[:n]
        draws[sample] = differences[positions].mean() * 252
    return {
        "annual_log_excess": float(differences.mean() * 252),
        "lower95": float(np.quantile(draws, 0.025)),
        "upper95": float(np.quantile(draws, 0.975)),
        "positive_share": float((draws > 0).mean()),
    }


def _save_registry(output: Path, name: str, policies: dict[str, list[Policy]]) -> None:
    path = output / name
    payload = {context: [asdict(policy) for policy in rows] for context, rows in policies.items()}
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(f"Frozen registry differs; use a new output directory: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_lab(config: BotConfig, *, rotation_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts(config, rotation_root)
    features = {context.name: Features(context) for context in contexts}
    policies: dict[str, dict[str, Policy]] = {context.name: {} for context in contexts}
    results: dict[str, dict[str, BacktestResult]] = {context.name: {} for context in contexts}
    diagnostics: dict[str, dict[str, pd.DataFrame]] = {context.name: {} for context in contexts}
    rows: list[dict] = []
    checkpoints: list[dict] = []
    for round_number in (1, 2, 3, 4):
        additions = {}
        if round_number == 1:
            additions = {context.name: [*controls(), *initial_policies()] for context in contexts}
        else:
            winners = select_winners(pd.DataFrame(rows))
            for context in contexts:
                parents = {
                    str(row.family): policies[context.name][row.candidate]
                    for row in winners[winners.context.eq(context.name)].itertuples()
                }
                additions[context.name] = (
                    combination_policies(parents)
                    if round_number == 4
                    else [
                        child
                        for parent in parents.values()
                        for child in followups(parent, round_number)
                    ]
                )
        _save_registry(output, f"round_{round_number}_frozen_registry.json", additions)
        for context in contexts:
            for policy in additions[context.name]:
                target, diagnostic = build_policy_targets(features[context.name], policy)
                result = execute(
                    context,
                    target,
                    controlled=policy.family != "benchmark",
                    risk_scale=policy.static_scale,
                )
                policies[context.name][policy.name] = policy
                results[context.name][policy.name] = result
                diagnostics[context.name][policy.name] = diagnostic
                rows.extend(metrics_rows(context, policy, result))
            print(
                f"round {round_number}: {context.name}: {len(additions[context.name])} paths completed",
                flush=True,
            )
        metrics = pd.DataFrame(rows)
        winners = select_winners(metrics)
        winners.to_csv(output / f"round_{round_number}_training_winners.csv", index=False)
        checkpoints.extend(
            {"round": round_number, **record} for record in winners.to_dict("records")
        )
        metrics.to_csv(output / "candidate_metrics.csv", index=False)
    selected = select_winners(pd.DataFrame(rows))
    for context in contexts:
        combo = pd.DataFrame(rows)
        combo = combo[combo.context.eq(context.name) & combo.family.eq("combination")]
        combo = combo[combo.period.isin(["development", "validation"])]
        # Combination selection also reads only the two old training periods.
        score = combo.groupby("candidate").agg(sharpe=("sharpe", "min"), dd=("max_drawdown", "min"))
        best = (score.sharpe + 3 * score.dd).idxmax()
        selected = pd.concat(
            [
                selected,
                pd.DataFrame(
                    [
                        {
                            "context": context.name,
                            "family": "combination",
                            "candidate": best,
                            "training_score": np.nan,
                        }
                    ]
                ),
            ]
        )
    selected.to_csv(output / "final_training_selection.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(output / "round_checkpoints.csv", index=False)
    evaluate_selected(
        contexts, features, policies, results, diagnostics, pd.DataFrame(rows), selected, output
    )
    roster = {
        name: [asdict(policy) for policy in mapping.values()] for name, mapping in policies.items()
    }
    (output / "all_trials.json").write_text(json.dumps(roster, indent=2, sort_keys=True) + "\n")
    primary_names = [
        f"{name}:{policy['name']}" for name, entries in roster.items() for policy in entries
    ]
    paired = pd.read_csv(output / "paired_ablations.csv")[["context", "ablation"]].drop_duplicates()
    supplemental_names = [f"{row.context}:{row.ablation}" for row in paired.itertuples()]
    risk_matches = pd.read_csv(output / "risk_matched_comparison.csv")
    counts = {
        "main_grid": len(primary_names),
        "paired_ablations": len(paired),
        "execution_profiles": len(pd.read_csv(output / "execution_stress.csv")),
        "development_risk_matched_controls": len(
            risk_matches[["context", "candidate"]].drop_duplicates()
        ),
        "original_operating_reference": 1,
    }
    write_research_manifest(
        output,
        study="adaptive_response_five_experiments",
        config=config,
        prices=contexts[0].prices,
        parameters={
            "candidate_roster": roster,
            "candidate_names": [*primary_names, *supplemental_names],
            "completed_backtest_evaluations": counts,
            "rounds_per_family": 3,
            "selection_periods": ["development", "validation"],
            "all_history_previously_observed": True,
            "input_sources": [
                {
                    "context": c.name,
                    "path": c.source_path,
                    "sha256": c.source_sha256,
                    "start": str(c.evaluation_start.date()),
                    "end": str(c.prices.index[-1].date()),
                }
                for c in contexts
            ],
            "cash": "lagged IRX rate proxy; idle capital funded after risk controls",
            "execution": "close signals, two-session lag, 10bps per absolute weight change",
        },
        artifacts=[
            path.name
            for path in output.iterdir()
            if path.is_file() and path.name != "manifest.json"
        ],
    )
    print(output / "report.md", flush=True)


def evaluate_selected(
    contexts, features, policies, results, diagnostics, metrics, selected, output
):
    risk_match_rows = []
    stress_rows = []
    episode_rows = []
    bootstrap_rows = []
    behavior_rows = []
    current_rows = []
    pbo_rows = []
    ablation_rows = []
    ablation_registry = {}
    state_rows = []
    bridge_rows = []
    for context in contexts:
        name = context.name
        reference = results[name]["slow_reference"]
        reference_risk = reference.weights.drop(columns="CASH").sum(axis=1)
        names = selected[selected.context.eq(name)].candidate.tolist()
        ablation_registry[name] = []
        matrix = pd.DataFrame(
            {
                key: value.returns.loc[context.evaluation_start :]
                for key, value in results[name].items()
                if policies[name][key].family not in {"control", "benchmark"}
            }
        )
        matrix.to_parquet(output / f"{name}_all_candidate_returns.parquet")
        distinct = matrix.loc[:, ~matrix.round(12).T.duplicated()]
        pbo = estimate_probability_of_backtest_overfitting(distinct, partitions=8)
        pbo_rows.extend(
            {
                "context": name,
                "registered_variants": matrix.shape[1],
                "distinct_return_paths": distinct.shape[1],
                **row,
            }
            for row in pbo.summary.to_dict("records")
        )
        for candidate in names:
            policy = policies[name][candidate]
            result = results[name][candidate]
            risk = result.weights.drop(columns="CASH").sum(axis=1)
            training = (risk.index >= context.evaluation_start) & (
                risk.index <= pd.Timestamp("2011-12-31")
            )
            scale = float(
                (risk.loc[training].mean() / reference_risk.loc[training].mean()).clip(0, 1)
            )
            match = replace(Policy("matched", "control"), static_scale=scale)
            target, _ = build_policy_targets(features[name], match)
            matched = execute(context, target, risk_scale=scale)
            ablations = [
                replace(
                    policy, name=f"{candidate}_weekly_only", cadence="weekly", emergency_daily=False
                )
            ]
            if policy.selection in {"adaptive", "cusum"}:
                mean_alpha = float(
                    diagnostics[name][candidate]
                    .fast_weight.loc[context.evaluation_start : "2011-12-31"]
                    .mean()
                )
                ablations.extend(
                    [
                        replace(policy, name=f"{candidate}_no_fast", selection="slow"),
                        replace(
                            policy,
                            name=f"{candidate}_constant_fast",
                            selection="constant_blend",
                            max_fast=mean_alpha,
                        ),
                    ]
                )
            if policy.replacement != "none":
                ablations.append(
                    replace(policy, name=f"{candidate}_cash_destination", replacement="cash")
                )
            if policy.protection == "recovery":
                ablations.append(
                    replace(policy, name=f"{candidate}_no_latch", protection="continuous")
                )
            if policy.static_scale != 1.0:
                ablations.append(
                    replace(policy, name=f"{candidate}_no_static_cut", static_scale=1.0)
                )
            ablation_registry[name].extend(asdict(policy) for policy in ablations)
            _save_registry(
                output, f"{name}_{candidate}_paired_ablation_registry.json", {name: ablations}
            )
            for ablation in ablations:
                ablated_target, _ = build_policy_targets(features[name], ablation)
                ablated = execute(context, ablated_target, risk_scale=ablation.static_scale)
                for period, (start, end) in PERIODS.items():
                    start = (
                        max(pd.Timestamp(start), context.evaluation_start)
                        if start
                        else context.evaluation_start
                    )
                    am = period_metrics(ablated, start, end)
                    cm = period_metrics(result, start, end)
                    if am and cm:
                        ablation_rows.append(
                            {
                                "context": name,
                                "candidate": candidate,
                                "ablation": ablation.name,
                                "period": period,
                                "ablation_cagr": am["cagr"],
                                "ablation_max_drawdown": am["max_drawdown"],
                                "candidate_cagr_advantage": cm["cagr"] - am["cagr"],
                                "candidate_drawdown_advantage": cm["max_drawdown"]
                                - am["max_drawdown"],
                            }
                        )
            for period, (start, end) in PERIODS.items():
                start = (
                    max(pd.Timestamp(start), context.evaluation_start)
                    if start
                    else context.evaluation_start
                )
                cm = period_metrics(result, start, end)
                rm = period_metrics(matched, start, end)
                if cm and rm:
                    risk_match_rows.append(
                        {
                            "context": name,
                            "candidate": candidate,
                            "period": period,
                            "training_static_scale": scale,
                            "cagr_delta": cm["cagr"] - rm["cagr"],
                            "drawdown_delta": cm["max_drawdown"] - rm["max_drawdown"],
                        }
                    )
            for episode, (start, end) in EPISODES.items():
                start = max(pd.Timestamp(start), context.evaluation_start)
                cm = period_metrics(result, start, end)
                rm = period_metrics(reference, start, end)
                if cm and rm:
                    episode_rows.append(
                        {
                            "context": name,
                            "candidate": candidate,
                            "episode": episode,
                            "return": cm["total_return"],
                            "max_drawdown": cm["max_drawdown"],
                            "excess_return": cm["total_return"] - rm["total_return"],
                            "drawdown_improvement": cm["max_drawdown"] - rm["max_drawdown"],
                        }
                    )
            for period, start, end in (
                ("historical_test", "2019-01-01", "2023-12-31"),
                ("recent_diagnostic", "2024-01-01", None),
            ):
                for control, comparison in (
                    ("slow_reference", reference),
                    ("risk_matched", matched),
                ):
                    bootstrap_rows.append(
                        {
                            "context": name,
                            "candidate": candidate,
                            "period": period,
                            "control": control,
                            **bootstrap_difference(
                                result.returns.loc[start:end], comparison.returns.loc[start:end]
                            ),
                        }
                    )
            underlying = context.prices[context.benchmark]
            forward21 = underlying.shift(-21).div(underlying).sub(1)
            observed = forward21.notna() & (risk.index >= context.evaluation_start)
            defensive = risk < (reference_risk - 0.10)
            falsely_defensive = defensive & forward21.gt(0.03) & observed
            diagnostic = diagnostics[name][candidate]
            trigger = diagnostic.stress.ge(0.50)
            starts = trigger & ~trigger.shift(1, fill_value=False)
            lags = []
            late_losses = []
            eligible_starts = starts & (starts.index >= context.evaluation_start)
            for position in np.flatnonzero(eligible_starts.to_numpy()):
                later = np.flatnonzero(defensive.iloc[position : position + 21].to_numpy())
                if len(later):
                    lags.append(int(later[0]))
                    close = min(position + int(later[0]), len(underlying) - 1)
                    late_losses.append(
                        float(underlying.iloc[close] / underlying.iloc[position] - 1)
                    )
            behavior_rows.append(
                {
                    "context": name,
                    "candidate": candidate,
                    "defensive_day_share": float(defensive.loc[context.evaluation_start :].mean()),
                    "false_defense_share_of_observed_days": float(
                        falsely_defensive[observed].mean()
                    ),
                    "median_response_sessions": float(np.median(lags)) if lags else np.nan,
                    "triggers": int(starts.loc[context.evaluation_start :].sum()),
                    "triggers_with_defense_within21": len(lags),
                    "median_benchmark_return_before_defense": float(np.median(late_losses))
                    if late_losses
                    else np.nan,
                    "average_risk": float(risk.loc[context.evaluation_start :].mean()),
                }
            )
            regime = pd.cut(
                features[name].stress(21).shift(2),
                bins=[-0.01, 0.25, 0.50, 1.01],
                labels=["normal", "transition", "stress"],
            )
            for state in ("normal", "transition", "stress"):
                mask = regime.eq(state) & (regime.index >= context.evaluation_start)
                state_rows.append(
                    {
                        "context": name,
                        "candidate": candidate,
                        "state": state,
                        "sessions": int(mask.sum()),
                        "annualized_arithmetic_return": float(result.returns[mask].mean() * 252),
                        "reference_annualized_arithmetic_return": float(
                            reference.returns[mask].mean() * 252
                        ),
                        "average_risk": float(risk[mask].mean()),
                    }
                )
            intent = diagnostic.iloc[-1]
            current_rows.append(
                {
                    "context": name,
                    "candidate": candidate,
                    "market_date": str(context.prices.index[-1].date()),
                    "stress": intent.stress,
                    "fast_weight": intent.fast_weight,
                    "executed_risk": float(risk.iloc[-1]),
                    "executed_positions": _positions(result.weights.iloc[-1]),
                    "scheduled_target": _positions(result.target_weights.iloc[-1]),
                }
            )
            result.weights.to_parquet(output / f"{name}_{policy.family}_weights.parquet")
            diagnostic.to_parquet(output / f"{name}_{policy.family}_diagnostics.parquet")
        _save_registry(
            output,
            f"{name}_paired_ablation_registry.json",
            {name: [Policy(**row) for row in ablation_registry[name]]},
        )
        if name == "technology":
            raw = build_strategy_weights(context.prices, context.strategy)
            operational = run_backtest(
                "configured_primary",
                context.prices,
                raw,
                ExecutionConfig(rebalance="W-WED", signal_lag_days=2, transaction_cost_bps=5),
                volatility_target=context.strategy.volatility_target,
                drawdown_control=context.strategy.drawdown_control,
            )
            for label, bridge in (
                ("configured_primary_existing_engine_5bps", operational),
                ("matched_primary_lab_contract_10bps", reference),
            ):
                measure = replace(
                    bridge,
                    weights=bridge.weights.rename(columns={"BIL": "CASH"})
                    .T.groupby(level=0)
                    .sum()
                    .T,
                )
                bridge_rows.append(
                    {"reference": label, **period_metrics(measure, context.evaluation_start, None)}
                )
        for candidate in ["slow_reference", *names]:
            policy = policies[name][candidate]
            for profile, weekday, lag, cost, daily in (
                ("wed_lag2_10bps", 2, 2, 10, False),
                ("mon_lag2_10bps", 0, 2, 10, False),
                ("fri_lag2_10bps", 4, 2, 10, False),
                ("daily_lag2_10bps", 2, 2, 10, True),
                ("wed_lag3_25bps", 2, 3, 25, False),
                ("wed_lag5_40bps", 2, 5, 40, False),
            ):
                target, _ = build_policy_targets(
                    features[name], policy, weekday=weekday, force_daily=daily
                )
                result = execute(
                    context, target, lag=lag, cost=cost, risk_scale=policy.static_scale
                )
                stress_rows.append(
                    {
                        "context": name,
                        "candidate": candidate,
                        "profile": profile,
                        **period_metrics(result, context.evaluation_start, None),
                    }
                )
        figure = make_subplots(
            rows=2, cols=1, shared_xaxes=True, subplot_titles=("Growth Of $100,000", "Drawdown")
        )
        for candidate in ["slow_reference", "static75", "buy_hold_spy", *names]:
            returns = results[name][candidate].returns.loc[context.evaluation_start :]
            equity = 100000 * (1 + returns).cumprod()
            figure.add_trace(go.Scatter(x=equity.index, y=equity, name=candidate), row=1, col=1)
            figure.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.div(equity.cummax()).sub(1),
                    name=candidate,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        figure.update_yaxes(type="log", row=1, col=1)
        figure.update_yaxes(tickformat=".0%", row=2, col=1)
        figure.update_layout(
            title=f"{name.title()}: Adaptive Response Research",
            height=850,
            template="plotly_white",
            legend={"orientation": "h"},
        )
        figure.write_html(output / f"{name}_performance.html", include_plotlyjs=True)
        print(
            f"{name}: risk-matched, episode, bootstrap, PBO and execution checks complete",
            flush=True,
        )
    frames = {
        "risk_matched_comparison": risk_match_rows,
        "execution_stress": stress_rows,
        "crisis_episodes": episode_rows,
        "bootstrap": bootstrap_rows,
        "behavior": behavior_rows,
        "current_allocations": current_rows,
        "pbo": pbo_rows,
        "paired_ablations": ablation_rows,
        "state_conditioned": state_rows,
        "operating_reference_bridge": bridge_rows,
    }
    for key, rows in frames.items():
        pd.DataFrame(rows).to_csv(output / f"{key}.csv", index=False)
    (output / "report.md").write_text(render_report(metrics, selected, frames), encoding="utf-8")


def _positions(weights: pd.Series) -> str:
    return ", ".join(
        f"{ticker} {weight:.1%}"
        for ticker, weight in weights[weights > 0.005].sort_values(ascending=False).items()
    )


def render_report(metrics, selected, frames) -> str:
    lines = [
        "# Adaptive Response: Five Experiments And Two Follow-up Rounds",
        "",
        "Retrospective research only. No promotion or live allocation authority.",
        "",
        "20 initial mechanisms, 15 first follow-ups, 15 second follow-ups and 6 combinations per universe, plus 11 controls.",
        "Development through 2011 and validation 2012-2018 determine every follow-up and family selection. All historical periods were previously observed in other research; later periods are diagnostics, not pristine OOS.",
        "Cash uses lagged IRX yield. Signals use completed closes with two-session lag and 10bps per unit of absolute weight turnover. Weekly signals occur on actual weekdays, including no artificial partial-week endpoint.",
        "Technology carries the configured i111 volatility/drawdown controls; sector strategies do not. Benchmarks are unscaled. Evaluation begins after a common 252-session warmup, so comparisons with earlier reports are not direct.",
        "",
    ]
    for context in metrics.context.unique():
        lines += [
            f"## {context.title()}",
            "",
            "| family/control | candidate | full CAGR | max DD | 2019-23 CAGR | 2024+ CAGR | current-period return |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        names = [policy.name for policy in controls()]
        names += selected[selected.context.eq(context)].candidate.tolist()
        for name in names:
            subset = metrics[metrics.context.eq(context) & metrics.candidate.eq(name)].set_index(
                "period"
            )
            row = subset.loc["full"]
            lines.append(
                f"| {row['family']} | {name} | {row.cagr:.2%} | {row.max_drawdown:.2%} | "
                f"{subset.loc['historical_test', 'cagr']:.2%} | "
                f"{subset.loc['recent_diagnostic', 'cagr']:.2%} | "
                f"{subset.loc['current_transition', 'total_return']:.2%} |"
            )
        lines += ["", "### Follow-up Selections", ""]
        for round_number in (1, 2, 3):
            subset = metrics[metrics.context.eq(context) & metrics["round"].le(round_number)]
            winners = select_winners(subset)
            lines.append(
                f"Round {round_number}: "
                + "; ".join(f"{r.family}: `{r.candidate}`" for r in winners.itertuples())
            )
        lines += ["", "### Training Risk-Matched Controls", ""]
        matched = pd.DataFrame(frames["risk_matched_comparison"])
        for row in matched[
            matched.context.eq(context) & matched.period.eq("historical_test")
        ].itertuples():
            lines.append(
                f"- {row.candidate}: 2019-23 CAGR delta {row.cagr_delta:+.2%}; "
                f"drawdown improvement {row.drawdown_delta:+.2%}; training static scale {row.training_static_scale:.0%}."
            )
        lines += ["", "### Execution Robustness", ""]
        execution = pd.DataFrame(frames["execution_stress"])
        for candidate, group in execution[execution.context.eq(context)].groupby(
            "candidate", sort=False
        ):
            ordinary = group[
                group.profile.isin(
                    ["wed_lag2_10bps", "mon_lag2_10bps", "fri_lag2_10bps", "daily_lag2_10bps"]
                )
            ]
            lines.append(
                f"- {candidate}: ordinary-schedule median CAGR {ordinary.cagr.median():.2%}, "
                f"worst DD {ordinary.max_drawdown.min():.2%}; "
                f"including severe delay/cost stress, worst DD {group.max_drawdown.min():.2%}."
            )
        lines += [
            "",
            "### Paired Mechanism Ablations",
            "",
            "Positive deltas favor the candidate over its ablation. These are falsification diagnostics, not an additional selection round.",
            "",
            "| candidate | removed/changed feature | full CAGR advantage | full DD advantage |",
            "|---|---|---:|---:|",
        ]
        ablations = pd.DataFrame(frames["paired_ablations"])
        for row in ablations[
            ablations.context.eq(context) & ablations.period.eq("full")
        ].itertuples():
            lines.append(
                f"| {row.candidate} | {row.ablation} | "
                f"{row.candidate_cagr_advantage:+.2%} | {row.candidate_drawdown_advantage:+.2%} |"
            )
    lines += ["", "## Statistical And Operational Checks", ""]
    for row in pd.DataFrame(frames["pbo"]).itertuples():
        lines.append(
            f"- {row.context}: {row.registered_variants} registered mechanisms/combinations; "
            f"{row.distinct_return_paths} distinct return paths; family CSCV PBO {row.pbo_probability:.1%}. "
            "This excludes the earlier research trial history and does not establish statistical significance."
        )
    bootstrap = pd.DataFrame(frames["bootstrap"])
    matched = bootstrap[bootstrap.control.eq("risk_matched")]
    overlaps = ((matched.lower95 <= 0) & (matched.upper95 >= 0)).sum()
    lines += [
        "",
        f"{overlaps} of {len(matched)} risk-matched block-bootstrap confidence intervals include zero. "
        "Positive point estimates do not establish durable incremental return.",
        "",
        "The exact input dates and SHA256 identities are in manifest.json. The common warmup and cash/execution accounting differ from prior studies; current-period returns begin June 1, 2026.",
        "",
    ]
    for row in pd.DataFrame(frames["operating_reference_bridge"]).itertuples():
        lines.append(
            f"- Reference bridge `{row.reference}`: common-period CAGR {row.cagr:.2%}, max DD {row.max_drawdown:.2%}."
        )
    lines += [
        "",
        "## Interpretation Boundaries",
        "",
        "- Runtime targets are constructed natively inside each experimental policy. No operating strategy or config is modified.",
        "- Adaptive tests share a source universe with their controls. Replacement is the explicit broad-universe exception, with cash-destination ablations.",
        "- Technology is a modern selected-stock universe, not a survivor-free historical investment opportunity set.",
        "- Sector ETFs include the dot-com period but are today's surviving funds with changing underlying classifications.",
        "- CUSUM is a transparent online downside change statistic, not a fitted or calibrated probability.",
        "- Static risk matches are frozen using development average risk only. Bootstrap intervals are pointwise and do not erase the full research trial history.",
        "- False-defense diagnostics use matured forward returns only for evaluation; they never enter policy targets.",
        "- Crisis windows include recovery to penalize missed rebounds. The 2026 episode is unfinished.",
        "- Close-to-close lagged execution cannot model intraday fills, gap exits or discretionary manual timing exactly.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--rotation-root", type=Path, default=Path("../rotation-trader"))
    parser.add_argument("--output", type=Path, default=Path("reports/adaptive_response_lab"))
    args = parser.parse_args()
    run_lab(load_config(args.config), rotation_root=args.rotation_root, output=args.output)


if __name__ == "__main__":
    main()
