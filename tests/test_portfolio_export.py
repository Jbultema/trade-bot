import json

import numpy as np
import pandas as pd

from trade_bot.config import StrategyConfig
from trade_bot.research.portfolio_export import _publish, research_targets


def prices():
    dates = pd.bdate_range("2010-01-01", periods=450)
    rng = np.random.default_rng(142)
    p = pd.DataFrame(
        100 * np.exp(rng.normal(0.0007, 0.01, (450, 4)).cumsum(axis=0)),
        index=dates,
        columns=["QQQ", "SPY", "XLV", "XLK"],
    )
    p["CASH"] = 1.0
    return p


def test_native_export_research_targets_are_causal():
    p = prices()
    config = StrategyConfig(
        type="dual_momentum",
        tickers=["QQQ", "XLK"],
        lookback_days=63,
        skip_days=5,
        top_n=2,
        defensive_ticker="CASH",
        max_asset_weight=0.5,
    )
    full = research_targets(p, config)
    prefix = research_targets(p.iloc[:337], config)
    assert set(full) == {"trade_primary", "trade_gentle_protection", "trade_staged_recovery"}
    for name in full:
        pd.testing.assert_frame_equal(full[name][0].iloc[:337], prefix[name][0])
        assert full[name][0].sum(axis=1).max() <= 1 + 1e-12


def test_bundle_has_no_authority_and_nonoverwriting_input_hashes(tmp_path):
    p = prices().drop(columns="CASH")
    source = tmp_path / "input.txt"
    source.write_text("frozen source")
    path = _publish(
        tmp_path / "bundle",
        p,
        {"test": (p * 0 + 0.1, p.index[252], "primary")},
        pd.Series(0.0, index=p.index),
        [source],
        "test_native",
        ["test"],
        "test",
    )
    m = json.loads(path.read_text())
    assert m["automatic_promotion_allowed"] is False
    assert m["authority"] == "research_only"
    assert m["source_inputs"][0]["sha256"]
