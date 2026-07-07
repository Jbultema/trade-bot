from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from trade_bot.cli import app
from trade_bot.research.m6_lab import (
    M6LabConfig,
    M6Window,
    forecast_equal_probability,
    forecast_score_quintiles,
    load_m6_lab_config,
    portfolio_weights_from_forecast,
    ranked_probability_score,
    run_m6_lab,
)


def test_ranked_probability_score_rewards_better_quintile_forecasts() -> None:
    actual = pd.Series({"a": 1, "b": 3, "c": 5})
    equal = forecast_equal_probability(actual.index)
    perfect = pd.DataFrame(
        {
            "q1": [1.0, 0.0, 0.0],
            "q2": [0.0, 0.0, 0.0],
            "q3": [0.0, 1.0, 0.0],
            "q4": [0.0, 0.0, 0.0],
            "q5": [0.0, 0.0, 1.0],
        },
        index=actual.index,
    )

    assert ranked_probability_score(perfect, actual) == pytest.approx(0.0)
    assert ranked_probability_score(equal, actual) > ranked_probability_score(perfect, actual)


def test_run_m6_lab_scores_external_forecast_and_investment_models() -> None:
    prices = _synthetic_prices()
    config = M6LabConfig(
        name="test_m6",
        universe=tuple(prices.columns),
        windows=(
            M6Window("p1", pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-26")),
            M6Window("p2", pd.Timestamp("2024-04-29"), pd.Timestamp("2024-05-24")),
            M6Window("p3", pd.Timestamp("2024-05-27"), pd.Timestamp("2024-06-21")),
        ),
        train_lookback_days=60,
        min_train_observations=30,
        simulations=25,
        random_seed=5,
        cv_window_count=2,
        cv_top_n=1,
        forecast_models=(
            "equal_probability",
            "momentum_quintile",
            "covariance_sample",
            "gorelli_cv_covariance",
            "tradebot_composite",
        ),
        portfolio_constructors=("equal_weight", "forecast_long_short"),
        covariance_estimators=("sample", "diagonal"),
    )

    run = run_m6_lab(prices, config=config)

    assert not run.forecast_scores.empty
    assert not run.investment_scores.empty
    assert not run.model_comparison.empty
    assert {"equal_probability", "gorelli_cv_covariance"}.issubset(
        set(run.forecast_scores["model"])
    )
    assert run.forecasts[["q1", "q2", "q3", "q4", "q5"]].sum(axis=1).round(8).eq(1.0).all()
    assert (
        run.portfolio_weights.groupby(["period", "model", "portfolio"])["weight"]
        .apply(lambda values: values.abs().sum() <= 1.0000001)
        .all()
    )


def test_portfolio_weights_from_forecast_normalize_exposure() -> None:
    scores = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0, "e": -1.0})
    forecast = forecast_score_quintiles(scores)

    long_short = portfolio_weights_from_forecast(forecast, constructor="forecast_long_short")
    long_only = portfolio_weights_from_forecast(forecast, constructor="forecast_long_only_top")

    assert long_short.abs().sum() == pytest.approx(1.0)
    assert long_only.sum() == pytest.approx(1.0)
    assert long_only.ge(0.0).all()


def test_run_m6_lab_cli_writes_outputs(monkeypatch, tmp_path) -> None:
    prices = _synthetic_prices()
    config_path = tmp_path / "m6.yaml"
    config_path.write_text(
        """
name: cli_test_m6
settings:
  train_lookback_days: 60
  min_train_observations: 30
  simulations: 10
  random_seed: 11
  cv_window_count: 1
  cv_top_n: 1
  forecast_models: [equal_probability, momentum_quintile, gorelli_cv_covariance]
  portfolio_constructors: [equal_weight, forecast_long_short]
  covariance_estimators: [sample, diagonal]
windows:
  - {label: p1, start: "2024-04-01", end: "2024-04-26"}
  - {label: p2, start: "2024-04-29", end: "2024-05-24"}
universe: [AAA, BBB, CCC, DDD, EEE, FFF]
""",
        encoding="utf-8",
    )

    def fake_loader(*args: object, **kwargs: object) -> pd.DataFrame:
        return prices

    monkeypatch.setattr("trade_bot.cli.load_or_fetch_yahoo_prices", fake_loader)
    result = CliRunner().invoke(
        app,
        [
            "run-m6-lab",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "m6_forecast_scores.csv").exists()
    assert (tmp_path / "out" / "m6_investment_scores.csv").exists()
    assert (tmp_path / "out" / "m6_model_comparison.csv").exists()
    assert "M6-Style Forecast Scores" in result.output


def test_load_m6_lab_config_reads_yaml(tmp_path) -> None:
    config_path = tmp_path / "m6.yaml"
    config_path.write_text(
        """
name: demo
settings:
  simulations: 7
windows:
  - {label: one, start: "2024-01-02", end: "2024-01-31"}
universe: [AAA, BBB]
""",
        encoding="utf-8",
    )

    config = load_m6_lab_config(config_path)

    assert config.name == "demo"
    assert config.simulations == 7
    assert config.universe == ("AAA", "BBB")
    assert config.windows[0].label == "one"


def _synthetic_prices() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=140)
    daily_returns = pd.DataFrame(
        {
            "AAA": np.linspace(0.0004, 0.0025, len(index)),
            "BBB": np.full(len(index), 0.0010),
            "CCC": np.sin(np.arange(len(index)) / 9.0) * 0.004,
            "DDD": np.linspace(0.0015, -0.0010, len(index)),
            "EEE": np.full(len(index), -0.0003),
            "FFF": np.cos(np.arange(len(index)) / 7.0) * 0.003,
        },
        index=index,
    )
    prices = 100.0 * (1.0 + daily_returns).cumprod()
    return prices
