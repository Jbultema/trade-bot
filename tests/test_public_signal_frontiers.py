from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from trade_bot.research.public_signal_frontiers import (
    _cross_sectional_weights,
    acquire_cboe_panel,
    acquire_cftc_panel,
    build_source_feasibility,
    finra_approximate_release_date,
    select_family_winners,
)


def test_cftc_tuesday_observation_is_not_available_until_friday(tmp_path, monkeypatch) -> None:
    row = {
        "Market_and_Exchange_Names": "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
        "Report_Date_as_YYYY-MM-DD": "2026-01-06",
        "Open_Interest_All": 1_000,
        "Asset_Mgr_Positions_Long_All": 400,
        "Asset_Mgr_Positions_Short_All": 200,
        "M_Money_Positions_Long_All": 0,
        "M_Money_Positions_Short_All": 0,
        "Conc_Gross_LE_4_TDR_Long_All": 20,
        "Conc_Gross_LE_4_TDR_Short_All": 10,
    }
    csv_payload = pd.DataFrame([row]).to_csv(index=False).encode()
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("cot.csv", csv_payload)

    def fake_get(url: str, path, *, refresh: bool):  # type: ignore[no-untyped-def]
        return zipped.getvalue(), "downloaded"

    monkeypatch.setattr(
        "trade_bot.research.public_signal_frontiers._cached_get",
        fake_get,
    )

    panel, _ = acquire_cftc_panel(
        tmp_path,
        start_year=2026,
        end_year=2026,
        refresh=True,
    )

    spy = panel.loc[panel["ticker"].eq("SPY")].iloc[0]
    assert spy["report_date"] == pd.Timestamp("2026-01-06")
    assert spy["available_date"] == pd.Timestamp("2026-01-09")


def test_finra_release_date_uses_third_friday_of_following_month() -> None:
    assert finra_approximate_release_date(pd.Timestamp("2026-01-01")) == pd.Timestamp("2026-02-20")
    assert finra_approximate_release_date(pd.Timestamp("2026-02-01")) == pd.Timestamp("2026-03-20")


def test_cross_sectional_weights_are_long_only_and_fully_invested() -> None:
    index = pd.bdate_range("2020-01-02", periods=4)
    scores = pd.DataFrame(
        {
            "SPY": [1.0, -1.0, 2.0, np.nan],
            "GLD": [0.5, -2.0, 1.0, np.nan],
            "IEF": [-1.0, -3.0, 0.5, np.nan],
        },
        index=index,
    )

    weights = _cross_sectional_weights(scores, top_n=2, minimum=0.0)

    assert weights.ge(0.0).all(axis=None)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights.iloc[-1]["BIL"] == 1.0


def test_family_selection_never_reads_holdout_or_forward() -> None:
    rows = []
    for candidate, dev, validation, holdout in (
        ("stable", 0.10, 0.10, -0.50),
        ("lucky_holdout", 0.01, 0.01, 2.00),
    ):
        for period, period_cagr in (
            ("development", dev),
            ("validation", validation),
            ("holdout", holdout),
            ("forward", holdout),
            ("common_full", (dev + validation + holdout) / 3.0),
        ):
            rows.append(
                {
                    "candidate": candidate,
                    "family": "test",
                    "period": period,
                    "cagr": period_cagr,
                    "sharpe": period_cagr * 5,
                    "max_drawdown": -0.20,
                }
            )

    selection = select_family_winners(pd.DataFrame(rows))

    assert selection.iloc[0]["candidate"] == "stable"


def test_cboe_parser_strips_column_whitespace(tmp_path, monkeypatch) -> None:
    payload = b"DATE, OPEN, HIGH, LOW, CLOSE\n01/02/2020,10,11,9,10.5\n"

    def fake_get(url: str, path, *, refresh: bool):  # type: ignore[no-untyped-def]
        return payload, "downloaded"

    monkeypatch.setattr(
        "trade_bot.research.public_signal_frontiers._cached_get",
        fake_get,
    )
    panel, inventory = acquire_cboe_panel(tmp_path, refresh=True)

    assert set(panel.columns) == {"vix", "vix3m", "skew"}
    assert float(panel.iloc[0]["vix"]) == 10.5
    assert inventory["status"].eq("downloaded").all()


def test_sec_frontier_fails_closed_without_governed_raw_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    inventory = pd.DataFrame([{"status": "cached", "source": "test", "bytes": 1}])

    feasibility = build_source_feasibility(
        tmp_path,
        cftc_inventory=inventory,
        options_inventory=inventory,
        margin_inventory=inventory,
    )
    sec = feasibility.loc[feasibility["frontier"].eq("sec_fundamental_events")].iloc[0]

    assert sec["status"] == "blocked"
    assert sec["backtest_authority"] == "none"
