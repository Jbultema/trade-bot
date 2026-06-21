from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.ml.diagnostics import run_ml_diagnostics


def test_ml_diagnostics_writes_metrics_probabilities_importance_and_drift(tmp_path) -> None:
    prices = _diagnostic_prices()

    run = run_ml_diagnostics(
        prices,
        output_dir=tmp_path,
        horizons=(21,),
        model_names=("sk_logit_l2", "sk_random_forest"),
    )

    assert not run.metrics.empty
    assert not run.latest_probabilities.empty
    assert not run.feature_importance.empty
    assert not run.family_importance.empty
    assert not run.drift.empty
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "latest_probabilities.csv").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "family_importance.csv").exists()
    assert (tmp_path / "drift.csv").exists()
    assert (tmp_path / "settings.json").exists()
    assert set(run.metrics["kind"]) >= {"future_state", "off_ramp", "strategy_family_router"}
    assert run.metrics["utility_score"].between(0.0, 1.0).all()


def _diagnostic_prices() -> pd.DataFrame:
    index = pd.bdate_range("2018-01-01", periods=620)
    trend = np.arange(len(index), dtype=float)
    cycle = np.sin(trend / 22.0)
    shock = np.where((trend > 260) & (trend < 340), -0.35 * (trend - 260), 0.0)
    repair = np.where(trend >= 340, 0.18 * (trend - 340), 0.0)
    spy = 100.0 + trend * 0.08 + cycle * 3.0 + shock + repair
    qqq = 100.0 + trend * 0.12 + cycle * 4.0 + shock * 1.2 + repair * 1.4
    rsp = 100.0 + trend * 0.07 + cycle * 2.2 + shock * 0.9 + repair
    smh = 100.0 + trend * 0.15 + cycle * 5.0 + shock * 1.3 + repair * 1.6
    safe = 100.0 + trend * 0.01
    frame = pd.DataFrame(
        {
            "SPY": spy,
            "QQQ": qqq,
            "RSP": rsp,
            "IWM": rsp * 0.96,
            "SMH": smh,
            "HYG": spy * 0.55 + 45.0,
            "LQD": safe,
            "TLT": 112.0 - trend * 0.015 + np.cos(trend / 45.0) * 2.0,
            "GLD": 100.0 + np.cos(trend / 38.0) * 5.0 + trend * 0.02,
            "USO": 80.0 + np.sin(trend / 18.0) * 8.0,
            "DBC": 90.0 + np.sin(trend / 24.0) * 5.0,
            "UUP": 100.0 + np.cos(trend / 30.0) * 2.0,
            "VIXY": 150.0 - spy * 0.25 + np.maximum(-shock, 0.0) * 0.55,
            "XLK": qqq * 1.01,
            "XLF": rsp * (0.95 + 0.02 * np.sin(trend / 50.0)),
            "XLE": 90.0 + np.sin(trend / 16.0) * 7.0 + trend * 0.03,
            "XLV": 95.0 + trend * 0.05 + np.cos(trend / 30.0),
            "XLI": rsp * 0.98,
            "XLY": qqq * 0.93,
            "XLP": safe * 1.02,
            "XLU": safe * 1.01,
            "XLRE": safe * 0.98,
            "XLC": qqq * 0.90,
            "BIL": safe,
        },
        index=index,
    )
    return frame.clip(lower=1.0)
