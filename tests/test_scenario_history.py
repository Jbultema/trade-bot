from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from trade_bot.research.scenario_history import (
    clean_scenario_history,
    reconstruct_scenario_history_from_prices,
    scenario_history_from_snapshots,
)


def test_clean_scenario_history_accepts_as_of_date_alias() -> None:
    raw = pd.DataFrame(
        [
            {
                "as_of_date": "2024-01-31",
                "horizon": "1m",
                "scenario": "Risk-on",
                "risk_bucket": "risk_on",
                "probability": 0.65,
            }
        ]
    )

    history = clean_scenario_history(raw)

    assert len(history) == 1
    assert str(history.iloc[0]["market_date"]) == "2024-01-31"
    assert history.iloc[0]["probability"] == 0.65


def test_scenario_history_from_snapshots_exports_lattice_rows() -> None:
    scenario_lattice = pd.DataFrame(
        [
            {
                "horizon": "1m",
                "scenario": "Risk-on",
                "risk_bucket": "risk_on",
                "probability": 0.60,
            },
            {
                "horizon": "1m",
                "scenario": "Transition",
                "risk_bucket": "transition",
                "probability": 0.40,
            },
        ]
    )
    run = SimpleNamespace(
        current_state=SimpleNamespace(scenario_lattice=scenario_lattice),
    )
    manifest = SimpleNamespace(
        run_id="snapshot-1",
        market_date="2024-01-31",
        created_at_utc="2024-02-01T00:00:00+00:00",
    )

    class Store:
        def list_snapshots(self, *, limit: int) -> pd.DataFrame:
            return pd.DataFrame([{"run_id": "snapshot-1"}])

        def load_snapshot(self, run_id: str) -> object:
            assert run_id == "snapshot-1"
            return run, manifest

    history = scenario_history_from_snapshots(Store(), limit=10)

    assert len(history) == 2
    assert set(history["risk_bucket"]) == {"risk_on", "transition"}
    assert set(history["run_id"]) == {"snapshot-1"}


def test_reconstruct_scenario_history_from_prices_uses_past_origin_state() -> None:
    index = pd.bdate_range("2024-01-02", periods=320)
    tickers = ["SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "TLT", "GLD", "SMH", "VIXY", "UUP"]
    prices = pd.DataFrame(index=index)
    for position, ticker in enumerate(tickers):
        drift = 0.0008 - position * 0.00003
        path = (1.0 + pd.Series([drift] * len(index), index=index)).cumprod()
        prices[ticker] = 100.0 * path
    prices.loc[index[210] :, "VIXY"] *= 1.4
    prices.loc[index[210] :, "SPY"] *= 0.9

    history = reconstruct_scenario_history_from_prices(
        prices,
        origin_frequency="quarterly",
        min_train_days=126,
    )

    assert not history.empty
    assert history["market_date"].nunique() >= 2
    assert set(history["horizon"]) == {"1w", "1m", "3m", "6m"}
    assert history.groupby(["market_date", "horizon"])["probability"].sum().round(6).eq(1.0).all()
    assert set(history["source"]) == {"reconstructed_price_state"}
