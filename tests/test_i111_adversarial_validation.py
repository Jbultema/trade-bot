from __future__ import annotations

import pandas as pd
import pytest

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import (
    BotConfig,
    DataConfig,
    DrawdownControlConfig,
    ExecutionConfig,
    StrategyConfig,
    VolatilityTargetConfig,
    required_strategy_tickers,
)
from trade_bot.research import i111_adversarial_validation as adversarial_validation
from trade_bot.research.i111_adversarial_validation import (
    AdversarialStrategySpec,
    _conditional_shift,
    _start_date_sensitivity,
    build_i111_adversarial_strategy_specs,
)
from trade_bot.research.i111_orthogonal_search import DEFAULT_I111_NATIVE_CHALLENGER


def test_adversarial_specs_include_native_challenger_and_primary_first() -> None:
    primary = _strategy()
    native = primary.model_copy(update={"type": "dual_momentum_risk_repair"})
    config = BotConfig(
        data=DataConfig(start="2020-01-01", end="2021-01-01"),
        execution=ExecutionConfig(),
        primary_strategy="i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145",
        universe={"core": ["SPY", "QQQ", "BIL"]},
        strategies={
            "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145": primary,
            DEFAULT_I111_NATIVE_CHALLENGER: native,
        },
    )

    specs = build_i111_adversarial_strategy_specs(config)
    names = [spec.name for spec in specs]

    assert names[0] == DEFAULT_I111_NATIVE_CHALLENGER
    assert "i111_reentry_vol_target_fast_21d_no_trend_vol185_guard145" in names


def test_conditional_shift_lags_trigger_and_preserves_long_only_weight_budget() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    weights = pd.DataFrame({"QQQ": [1.0, 1.0, 1.0, 1.0], "BIL": [0.0, 0.0, 0.0, 0.0]}, index=index)
    trigger = pd.Series([False, True, False, False], index=index)

    shifted = _conditional_shift(weights, trigger, 0.20, {"BIL": 1.0})

    assert shifted.loc[index[1], "BIL"] == 0.0
    assert shifted.loc[index[2], "BIL"] == 0.20
    assert shifted.sum(axis=1).le(1.0).all()
    assert shifted.ge(0.0).all().all()


def test_conditional_shift_transfers_mass_without_increasing_underinvested_exposure() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    weights = pd.DataFrame(
        {
            "QQQ": [0.30, 0.30, 0.30, 0.30],
            "SPY": [0.20, 0.20, 0.20, 0.20],
            "BIL": [0.10, 0.10, 0.10, 0.10],
        },
        index=index,
    )
    trigger = pd.Series([False, True, False, False], index=index)

    shifted = _conditional_shift(weights, trigger, 0.20, {"BIL": 1.0})

    pd.testing.assert_series_equal(
        shifted.sum(axis=1),
        weights.sum(axis=1),
        check_names=False,
    )
    assert shifted.loc[index[2], "QQQ"] == pytest.approx(0.18)
    assert shifted.loc[index[2], "SPY"] == pytest.approx(0.12)
    assert shifted.loc[index[2], "BIL"] == pytest.approx(0.30)


def test_adversarial_native_price_inputs_fail_closed_when_signal_is_missing() -> None:
    native = _strategy().model_copy(update={"type": "dual_momentum_risk_repair"})
    columns = [ticker for ticker in required_strategy_tickers(native) if ticker != "HYG"]
    prices = pd.DataFrame(
        100.0,
        index=pd.bdate_range("2024-01-01", periods=30),
        columns=columns,
    )

    with pytest.raises(KeyError, match="HYG"):
        adversarial_validation._strategy_prices(prices, native)


def test_start_date_sensitivity_carries_full_history_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.bdate_range("2020-01-02", periods=900)
    start = pd.Timestamp("2022-01-03")
    prices = pd.DataFrame({"QQQ": 100.0, "BIL": 100.0}, index=index)
    config = BotConfig(
        data=DataConfig(start=str(index[0].date()), end=str(index[-1].date())),
        execution=ExecutionConfig(),
        primary_strategy="candidate",
        universe={"core": ["QQQ", "BIL"]},
        strategies={"candidate": _strategy()},
    )
    spec = AdversarialStrategySpec(
        name="candidate",
        source_group="test",
        strategy=_strategy(),
    )
    call_starts: list[pd.Timestamp] = []
    full_returns = pd.Series(-0.001, index=index, name="candidate")
    full_returns.iloc[:63] = 0.0

    def fake_run_strategy(
        execution: ExecutionConfig,
        name: str,
        strategy: StrategyConfig,
        supplied_prices: pd.DataFrame,
    ) -> BacktestResult:
        del strategy
        call_starts.append(supplied_prices.index[0])
        returns = full_returns.reindex(supplied_prices.index)
        equity = execution.initial_capital * (1.0 + returns).cumprod()
        weights = pd.DataFrame(
            {"QQQ": 1.0, "BIL": 0.0},
            index=supplied_prices.index,
        )
        zeros = pd.Series(0.0, index=supplied_prices.index, name=name)
        return BacktestResult(
            name=name,
            equity=equity.rename(name),
            returns=returns.rename(name),
            gross_returns=returns.rename(name),
            weights=weights,
            target_weights=weights,
            turnover=zeros,
            transaction_costs=zeros,
        )

    monkeypatch.setattr(adversarial_validation, "_run_strategy", fake_run_strategy)
    monkeypatch.setattr(
        adversarial_validation,
        "DEFAULT_ADVERSARIAL_START_DATES",
        (str(start.date()),),
    )

    sensitivity = _start_date_sensitivity(config, (spec,), prices)

    assert call_starts == [index[0]]
    assert sensitivity.loc[0, "state_mode"] == "carried_state"
    carried_returns = full_returns.loc[full_returns.index >= start]
    years = (carried_returns.index[-1] - carried_returns.index[0]).days / 365.25
    expected_cagr = float((1.0 + carried_returns).prod() ** (1.0 / years) - 1.0)
    assert sensitivity.loc[0, "cagr"] == pytest.approx(expected_cagr)


def test_ai_monitor_audit_excludes_censored_terminal_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.bdate_range("2024-01-02", periods=12)
    benchmark = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 90.0, 89.0, 88.0, 87.0, 86.0, 85.0, 84.0],
        index=index,
    )
    prices = benchmark.to_frame("QQQ")
    horizon = 5

    def zero_score(supplied_prices: pd.DataFrame, profile: str) -> pd.Series:
        del profile
        return pd.Series(0.0, index=supplied_prices.index)

    def one_score(supplied_prices: pd.DataFrame, profile: str) -> pd.Series:
        del profile
        return pd.Series(1.0, index=supplied_prices.index)

    monkeypatch.setattr(adversarial_validation, "DEFAULT_MONITOR_HORIZONS", (horizon,))
    monkeypatch.setattr(adversarial_validation, "_ai_health_score", zero_score)
    monkeypatch.setattr(adversarial_validation, "_crash_onset_score", one_score)

    audit = adversarial_validation._ai_monitor_audit(prices)

    forward_return = benchmark.shift(-horizon) / benchmark - 1.0
    forward_drawdown = adversarial_validation._forward_drawdown(benchmark, horizon)
    eligible = forward_return.notna() & forward_drawdown.notna()
    expected_eligible_days = int(eligible.sum())
    expected_severe_rate = float((forward_drawdown.loc[eligible] <= -0.10).mean())
    expected_false_positive_rate = float(
        ((forward_return.loc[eligible] > 0.0) & (forward_drawdown.loc[eligible] > -0.05)).mean()
    )

    assert expected_eligible_days == len(index) - horizon
    assert audit["eligible_days"].eq(expected_eligible_days).all()
    assert audit["censored_days"].eq(horizon).all()
    assert audit["active_days"].eq(expected_eligible_days).all()
    assert audit["active_rate"].eq(1.0).all()
    assert audit["base_severe_forward_drawdown_rate"].eq(expected_severe_rate).all()
    assert audit["severe_forward_drawdown_rate"].eq(expected_severe_rate).all()
    assert audit["false_positive_rate"].eq(expected_false_positive_rate).all()


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        type="dual_momentum",
        tickers=["QQQ", "SMH", "SOXX", "IGV", "NVDA", "AVGO", "MSFT", "META", "AMZN", "PLTR"],
        defensive_ticker="BIL",
        lookback_days=63,
        skip_days=5,
        top_n=4,
        min_return=0.025,
        trend_filter_days=None,
        max_asset_weight=0.35,
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
    )
