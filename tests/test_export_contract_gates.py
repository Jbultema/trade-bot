import numpy as np
import pandas as pd
import pytest

from trade_bot.research.portfolio_export import _publish


@pytest.mark.parametrize("kind", ["missing_target", "nan_target", "unknown_asset", "cash_gap", "infinite_price"])
def test_export_does_not_silently_impute_invalid_inputs(tmp_path, kind):
    dates = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame(100.0, index=dates, columns=["SPY"])
    targets = prices * 0 + 0.5
    cash = pd.Series(0.0, index=dates)
    if kind == "missing_target":
        targets = targets.iloc[:-1]
    elif kind == "nan_target":
        targets.iloc[1, 0] = np.nan
    elif kind == "unknown_asset":
        targets["UNKNOWN"] = 0.1
    elif kind == "cash_gap":
        cash.iloc[1] = np.nan
    else:
        prices.iloc[1, 0] = np.inf
    with pytest.raises(ValueError):
        _publish(
            tmp_path / "bundle", prices,
            {"primary": (targets, dates[0], "primary")},
            cash, [], "test_only", [], "primary",
        )
    assert not (tmp_path / "bundle/manifest.json").exists()
