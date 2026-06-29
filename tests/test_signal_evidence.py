from __future__ import annotations

import pandas as pd

from trade_bot.research.signal_evidence import (
    build_signal_family_evidence,
    build_signal_family_marginal_tests,
    tag_scorecard_signal_families,
)


def test_signal_family_marginal_tests_compare_child_to_parent() -> None:
    scorecards = pd.DataFrame(
        [
            {
                "iteration": 1,
                "strategy": "base_cycle",
                "phase": "cycle",
                "family": "broad_momentum",
                "role": "parent",
                "parent": "",
                "promotion_score": 0.60,
                "cagr": 0.11,
                "max_drawdown": -0.22,
                "calmar": 0.50,
                "average_turnover": 0.07,
                "reentry_score": 0.40,
                "left_tail_regime_return": -0.12,
                "hypothesis": "Broad trend momentum parent.",
            },
            {
                "iteration": 2,
                "strategy": "credit_reentry_child",
                "phase": "dip_reentry",
                "family": "credit_reentry",
                "role": "candidate",
                "parent": "base_cycle",
                "promotion_score": 0.75,
                "cagr": 0.14,
                "max_drawdown": -0.17,
                "calmar": 0.82,
                "average_turnover": 0.05,
                "reentry_score": 0.80,
                "left_tail_regime_return": -0.06,
                "hypothesis": "Credit repair gate improves re-entry after drawdowns.",
            },
        ]
    )

    tagged = tag_scorecard_signal_families(scorecards)
    tests = build_signal_family_marginal_tests(tagged)
    reentry = tests[tests["signal_family"] == "reentry_timing"].iloc[0]
    credit = tests[tests["signal_family"] == "credit"].iloc[0]

    assert "reentry_timing" in tagged.set_index("strategy").loc[
        "credit_reentry_child", "signal_families"
    ]
    assert reentry["child_strategy"] == "credit_reentry_child"
    assert round(float(reentry["delta_cagr"]), 6) == 0.03
    assert round(float(reentry["delta_max_drawdown"]), 6) == 0.05
    assert round(float(reentry["delta_reentry_score"]), 6) == 0.40
    assert round(float(reentry["delta_average_turnover"]), 6) == -0.02
    assert credit["parent_strategy"] == "base_cycle"


def test_signal_family_evidence_labels_validated_contributor_with_enough_pairs() -> None:
    rows = []
    for idx in range(3):
        parent = f"base_{idx}"
        child = f"child_{idx}"
        rows.extend(
            [
                {
                    "iteration": idx,
                    "strategy": parent,
                    "phase": "cycle",
                    "family": "broad_momentum",
                    "role": "parent",
                    "parent": "",
                    "promotion_score": 0.55,
                    "cagr": 0.10,
                    "max_drawdown": -0.23,
                    "calmar": 0.43,
                    "average_turnover": 0.08,
                    "reentry_score": 0.30,
                    "hypothesis": "Trend parent.",
                },
                {
                    "iteration": idx,
                    "strategy": child,
                    "phase": "dip_reentry",
                    "family": "credit_reentry",
                    "role": "candidate",
                    "parent": parent,
                    "promotion_score": 0.70,
                    "cagr": 0.13,
                    "max_drawdown": -0.18,
                    "calmar": 0.72,
                    "average_turnover": 0.06,
                    "reentry_score": 0.75,
                    "hypothesis": "Credit reentry after drawdown repair.",
                },
            ]
        )
    evidence = build_signal_family_evidence(pd.DataFrame(rows))
    reentry = evidence[evidence["signal_family"] == "reentry_timing"].iloc[0]

    assert reentry["paired_tests"] == 3
    assert reentry["evidence_label"] == "validated_contributor"
    assert float(reentry["net_evidence_score"]) >= 0.65
    assert "Keep in model-search" in reentry["recommendation"]


def test_signal_family_evidence_keeps_unpaired_thin_topics_context_only() -> None:
    scorecards = pd.DataFrame(
        [
            {
                "strategy": "earnings_proxy_candidate",
                "phase": "long_form_macro_process",
                "family": "earnings_revision_proxy",
                "role": "candidate",
                "promotion_score": 0.45,
                "cagr": 0.06,
                "max_drawdown": -0.11,
                "average_turnover": 0.04,
                "hypothesis": "Earnings revision and margin pressure proxy.",
            },
            {
                "strategy": "earnings_proxy_candidate_2",
                "phase": "long_form_macro_process",
                "family": "earnings_revision_proxy",
                "role": "candidate",
                "promotion_score": 0.46,
                "cagr": 0.07,
                "max_drawdown": -0.12,
                "average_turnover": 0.04,
                "hypothesis": "Earnings revisions without clean parent control.",
            },
            {
                "strategy": "earnings_proxy_candidate_3",
                "phase": "long_form_macro_process",
                "family": "earnings_revision_proxy",
                "role": "candidate",
                "promotion_score": 0.47,
                "cagr": 0.08,
                "max_drawdown": -0.13,
                "average_turnover": 0.04,
                "hypothesis": "Margin and FCF revision proxy.",
            },
        ]
    )

    evidence = build_signal_family_evidence(scorecards)
    earnings = evidence[evidence["signal_family"] == "earnings_revision"].iloc[0]

    assert earnings["paired_tests"] == 0
    assert earnings["evidence_label"] == "context_only"
    assert "thin_proxy" in earnings["data_status"]
