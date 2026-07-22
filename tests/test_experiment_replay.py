from __future__ import annotations

import pandas as pd

from trade_bot.config import load_config
from trade_bot.DEFAULTS import DEFAULT_CONFIG_PATH, DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION
from trade_bot.research.evaluation_contract import (
    build_strategy_evaluation_contract,
    evaluation_contract_sha256,
)
from trade_bot.research.experiment_replay import (
    load_saved_experiment_candidates,
    verify_experiment_library,
)


def test_evaluation_contract_includes_exact_price_frame_identity() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    dates = pd.bdate_range("2025-01-02", periods=4)
    base = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 103.0]}, index=dates)
    expanded = base.assign(QQQ=[100.0, 102.0, 104.0, 106.0])

    base_hash = evaluation_contract_sha256(build_strategy_evaluation_contract(config, base))
    expanded_hash = evaluation_contract_sha256(
        build_strategy_evaluation_contract(config, expanded)
    )

    assert base_hash != expanded_hash


def test_evaluation_contract_freezes_account_contribution_assumption() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.bdate_range("2025-01-02", periods=2),
    )

    contract = build_strategy_evaluation_contract(config, prices)

    assert DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION == 4_000.0
    assert contract["outcome_planning"]["annual_contribution"] == 4_000.0


def test_saved_candidate_loader_preserves_iteration_and_strategy_definitions(tmp_path) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    iteration = tmp_path / "iteration_77"
    iteration.mkdir()
    strategy = config.strategies[config.primary_strategy]
    pd.DataFrame(
        [
            {
                "strategy": "saved_candidate",
                "hypothesis": "Replay me exactly.",
                "role": "candidate_core",
                "phase": "broad",
                "family": "unit_test",
                "parent": "",
                "strategy_json": strategy.model_dump_json(),
                "scenario_sizing_json": "",
                "future_state_model_json": "",
                "strategy_drawdown_model_json": "",
                "decision_sanity_json": "",
            }
        ]
    ).to_csv(iteration / "candidates.csv", index=False)

    loaded = load_saved_experiment_candidates(tmp_path)

    assert tuple(loaded) == (77,)
    assert loaded[77][0].name == "saved_candidate"
    assert loaded[77][0].strategy == strategy


def test_library_verification_fails_closed_without_manifest(tmp_path) -> None:
    assert verify_experiment_library(tmp_path)["status"] == "missing_or_unreadable_manifest"
