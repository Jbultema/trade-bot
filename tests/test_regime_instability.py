from __future__ import annotations

import numpy as np
import pandas as pd

from trade_bot.research.regime_instability import build_regime_instability_index


def test_regime_instability_index_detects_large_moves_and_dispersion() -> None:
    index = pd.bdate_range("2024-01-01", periods=260)
    calm_prices = _prices(index, shock_scale=0.001, alternating=False)
    unstable_prices = _prices(index, shock_scale=0.018, alternating=True)

    calm_summary, calm_components = build_regime_instability_index(calm_prices)
    unstable_summary, unstable_components = build_regime_instability_index(unstable_prices)

    assert not calm_summary.empty
    assert not calm_components.empty
    assert not unstable_summary.empty
    assert not unstable_components.empty
    assert float(unstable_summary.iloc[0]["regime_instability_score"]) > float(
        calm_summary.iloc[0]["regime_instability_score"]
    )
    assert "watch_only" in str(unstable_summary.iloc[0]["trading_use"])
    assert "large_move_share_21d" in set(unstable_components["component"])


def _prices(index: pd.DatetimeIndex, *, shock_scale: float, alternating: bool) -> pd.DataFrame:
    trend = np.linspace(100.0, 125.0, len(index))
    wave = np.sin(np.arange(len(index)) / 2.0) * shock_scale
    if alternating:
        wave = wave + ((np.arange(len(index)) % 2) * 2 - 1) * shock_scale * 0.7
    returns = pd.Series(0.0004 + wave, index=index)
    spy = 100.0 * (1.0 + returns).cumprod()
    qqq = spy * (1.0 + np.linspace(0.0, 0.18 if alternating else 0.03, len(index)))
    rsp = spy * (1.0 - np.linspace(0.0, 0.06 if alternating else 0.00, len(index)))
    smh = spy * (1.0 + np.linspace(0.0, 0.28 if alternating else 0.04, len(index)))
    hyg = spy * (0.85 - np.linspace(0.0, 0.03 if alternating else 0.00, len(index)))
    lqd = trend * 0.85
    vixy = 50.0 * (1.0 + returns.abs().rolling(5, min_periods=1).mean() * 25.0).cumprod()
    return pd.DataFrame(
        {
            "SPY": spy,
            "QQQ": qqq,
            "RSP": rsp,
            "IWM": rsp * 1.03,
            "SMH": smh,
            "XLK": qqq * 0.98,
            "XLF": rsp * 0.95,
            "HYG": hyg,
            "LQD": lqd,
            "TLT": trend * 0.75,
            "GLD": trend * 0.90,
            "VIXY": vixy,
        },
        index=index,
    )
