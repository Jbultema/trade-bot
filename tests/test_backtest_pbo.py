from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.config import BotConfig, DataConfig, ExecutionConfig, StrategyConfig
from trade_bot.research.backtest_pbo import (
    build_pbo_candidate_results,
    build_pbo_candidate_results_with_audit,
    estimate_probability_of_backtest_overfitting,
    run_backtest_pbo_gauntlet,
)


def test_probability_of_backtest_overfitting_estimates_split_ranks() -> None:
    index = pd.bdate_range("2020-01-01", periods=160)
    returns = pd.DataFrame(index=index)
    returns["steady"] = 0.0008
    returns["overfit_a"] = [0.004 if (idx // 40) in {0, 2} else -0.003 for idx in range(len(index))]
    returns["overfit_b"] = [0.004 if (idx // 40) in {1, 3} else -0.003 for idx in range(len(index))]

    result = estimate_probability_of_backtest_overfitting(
        returns,
        partitions=4,
        metric="mean_return",
    )

    summary = result.summary.iloc[0]
    assert summary["strategy_count"] == 3
    assert summary["valid_splits"] == 6
    assert 0.0 <= summary["pbo_probability"] <= 1.0
    assert {"selected_strategy", "relative_rank", "overfit"}.issubset(result.splits.columns)
    assert not result.strategy_selection.empty


def test_backtest_pbo_gauntlet_writes_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)

    gauntlet = run_backtest_pbo_gauntlet(
        config=config,
        prices=_prices(),
        output_dir=tmp_path / "pbo",
        experiment_root=tmp_path / "missing_experiments",
        top_n=3,
        partitions=4,
        metric="mean_return",
        min_observations=60,
    )

    assert gauntlet.artifacts["summary"].exists()
    assert gauntlet.artifacts["splits"].exists()
    assert gauntlet.artifacts["strategy_selection"].exists()
    assert gauntlet.artifacts["strategy_stats"].exists()
    assert gauntlet.artifacts["candidate_audit"].exists()
    assert gauntlet.result.summary.iloc[0]["strategy_count"] >= 2
    assert gauntlet.result.summary.iloc[0]["expected_candidate_count"] == 3
    assert gauntlet.result.summary.iloc[0]["evaluated_candidate_count"] == 3
    assert gauntlet.result.candidate_audit["status"].eq("evaluated").all()
    assert "PBO probability" in gauntlet.readout


def test_backtest_pbo_gauntlet_audits_missing_price_exclusion(tmp_path: Path) -> None:
    prices = _prices().drop(columns="VEA")

    gauntlet = run_backtest_pbo_gauntlet(
        config=_config(tmp_path),
        prices=prices,
        output_dir=tmp_path / "pbo_missing_price",
        experiment_root=tmp_path / "missing_experiments",
        top_n=3,
        partitions=4,
        metric="mean_return",
        min_observations=60,
    )

    missing = gauntlet.result.candidate_audit.loc[
        gauntlet.result.candidate_audit["strategy"].eq("global_momo")
    ].iloc[0]
    summary = gauntlet.result.summary.iloc[0]
    persisted = pd.read_csv(gauntlet.artifacts["candidate_audit"])

    assert missing["status"] == "excluded"
    assert missing["reconstruction_reason_code"] == "missing_price_inputs"
    assert missing["reason_code"] == "missing_price_inputs"
    assert missing["missing_price_columns"] == "VEA"
    assert summary["expected_candidate_count"] == 3
    assert summary["reconstructed_candidate_count"] == 2
    assert summary["evaluated_candidate_count"] == 2
    assert summary["excluded_candidate_count"] == 1
    assert persisted["strategy"].eq("global_momo").any()


def test_pbo_candidate_reconstruction_errors_are_audited_without_breaking_legacy_callers(
    tmp_path: Path,
) -> None:
    rows = pd.DataFrame(
        [
            {"strategy": "broken_candidate", "strategy_json": "{"},
            {"strategy": "", "strategy_json": "{}"},
        ]
    )

    build = build_pbo_candidate_results_with_audit(_config(tmp_path), _prices(), rows)
    legacy_results = build_pbo_candidate_results(_config(tmp_path), _prices(), rows)

    assert build.results == {}
    assert legacy_results == {}
    assert build.audit["status"].eq("excluded").all()
    assert build.audit["reason_code"].tolist() == [
        "catalog_reconstruction_error",
        "missing_strategy_name",
    ]
    assert build.audit["reason"].str.len().gt(0).all()


def _config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        data=DataConfig(start="2020-01-01", cache_dir=str(tmp_path)),
        execution=ExecutionConfig(
            initial_capital=1000.0,
            rebalance="W-WED",
            signal_lag_days=1,
            transaction_cost_bps=1.0,
        ),
        primary_strategy="qqq_momo",
        universe={"test": ["QQQ", "SMH", "SPY", "VEA", "BIL"]},
        strategies={
            "qqq_momo": StrategyConfig(
                type="dual_momentum",
                tickers=["QQQ", "SMH"],
                defensive_ticker="BIL",
                lookback_days=21,
                skip_days=1,
                top_n=1,
                min_return=-1.0,
            ),
            "global_momo": StrategyConfig(
                type="dual_momentum",
                tickers=["SPY", "VEA"],
                defensive_ticker="BIL",
                lookback_days=21,
                skip_days=1,
                top_n=1,
                min_return=-1.0,
            ),
            "spy_hold": StrategyConfig(
                type="buy_hold",
                tickers=["SPY"],
            ),
        },
    )


def _prices() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=260)
    frame = pd.DataFrame(index=index)
    for offset, ticker in enumerate(["QQQ", "SMH", "SPY", "VEA", "BIL"], start=1):
        drift = 0.00015 * offset
        cycle = pd.Series(range(len(index)), index=index).map(lambda value: (value % 23) / 12000)
        frame[ticker] = 100.0 * (1.0 + drift + cycle).cumprod()
    return frame
