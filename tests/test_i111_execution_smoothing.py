from __future__ import annotations

import pandas as pd

from trade_bot.research.i111_execution_smoothing import (
    _promotion_gates,
    _summary_markdown,
    causal_smooth_weights,
    default_execution_smoothing_specs,
)


def test_execution_smoothing_candidate_set_is_fixed_and_small() -> None:
    specs = default_execution_smoothing_specs()

    assert [spec.name for spec in specs] == ["raw", "ewm5", "mean10"]
    assert [(spec.kind, spec.span_or_window) for spec in specs] == [
        ("raw", None),
        ("ewm", 5),
        ("mean", 10),
    ]


def test_execution_smoothing_is_causal() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    weights = pd.DataFrame(
        {
            "QQQ": [1.0] * 6 + [0.0] * 6,
            "BIL": [0.0] * 6 + [1.0] * 6,
        },
        index=dates,
    )
    changed_future = weights.copy()
    changed_future.loc[dates[8] :, ["QQQ", "BIL"]] = [1.0, 0.0]

    for spec in default_execution_smoothing_specs():
        original = causal_smooth_weights(weights, spec)
        perturbed = causal_smooth_weights(changed_future, spec)
        pd.testing.assert_frame_equal(original.loc[: dates[7]], perturbed.loc[: dates[7]])


def test_execution_smoothing_preserves_long_only_exposure_limit() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    weights = pd.DataFrame(
        {
            "QQQ": [1.2, 0.0, 0.7, 0.1],
            "BIL": [-0.2, 1.0, 0.7, 0.9],
        },
        index=dates,
    )

    for spec in default_execution_smoothing_specs():
        smoothed = causal_smooth_weights(weights, spec)
        assert smoothed.ge(0.0).all().all()
        assert smoothed.sum(axis=1).le(1.0 + 1e-12).all()


def test_execution_smoothing_pbo_gate_is_explicitly_family_level() -> None:
    rows: list[dict[str, object]] = []
    for is_base_cost, cost_adjustment in ((True, 0.0), (False, -0.02)):
        for transform in ("raw", "ewm5", "mean10"):
            rows.append(
                {
                    "transform": transform,
                    "is_base_cost": is_base_cost,
                    "median_schedule_cagr": 0.20 + cost_adjustment,
                    "worst_schedule_drawdown": -0.30 if transform == "raw" else -0.25,
                    "schedule_cagr_range": 0.008,
                    "wednesday_cagr": 0.20,
                    "wednesday_max_drawdown": -0.20,
                }
            )
    summary = pd.DataFrame(rows)
    rolling = pd.DataFrame(
        [
            {"name": f"{transform}__wed", "window": "3y", "cagr": 0.01}
            for transform in ("raw", "ewm5", "mean10")
        ]
    )
    pbo = pd.DataFrame(
        [
            {
                "strategy_count": 15,
                "valid_splits": 70,
                "pbo_probability": 0.20,
            }
        ]
    )

    gates = _promotion_gates(summary, rolling, pbo)
    readout = _summary_markdown(summary, gates, pbo)

    assert gates["family_pbo_gate"].all()
    assert gates["pbo_scope"].eq("family_15_strategies_70_splits").all()
    assert "local_pbo" not in gates
    assert "Family-level CSCV PBO across 15 strategies and 70 splits" in readout
    assert "eight retrospective gates" in readout
    assert "configured-Wednesday CAGR within 0.5 points" in readout
    assert "max drawdown within one point" in readout
    assert "same PBO gate applies to every row" in readout


def test_execution_smoothing_requires_wednesday_noninferiority() -> None:
    rows: list[dict[str, object]] = []
    for is_base_cost, cost_adjustment in ((True, 0.0), (False, -0.02)):
        rows.extend(
            [
                {
                    "transform": "raw",
                    "is_base_cost": is_base_cost,
                    "median_schedule_cagr": 0.20 + cost_adjustment,
                    "worst_schedule_drawdown": -0.30,
                    "schedule_cagr_range": 0.008,
                    "wednesday_cagr": 0.20 + cost_adjustment,
                    "wednesday_max_drawdown": -0.20,
                },
                {
                    "transform": "ewm5",
                    "is_base_cost": is_base_cost,
                    "median_schedule_cagr": 0.20 + cost_adjustment,
                    "worst_schedule_drawdown": -0.25,
                    "schedule_cagr_range": 0.008,
                    "wednesday_cagr": 0.18 + cost_adjustment,
                    "wednesday_max_drawdown": -0.22,
                },
            ]
        )
    rolling = pd.DataFrame(
        [
            {"name": f"{transform}__wed", "window": "3y", "cagr": 0.01}
            for transform in ("raw", "ewm5")
        ]
    )
    pbo = pd.DataFrame([{"strategy_count": 10, "valid_splits": 20, "pbo_probability": 0.20}])

    gates = _promotion_gates(pd.DataFrame(rows), rolling, pbo).set_index("transform")

    assert not bool(gates.loc["ewm5", "wednesday_edge_noninferiority"])
    assert not bool(gates.loc["ewm5", "wednesday_tail_noninferiority"])
    assert gates.loc["ewm5", "research_status"] == "research_only"
