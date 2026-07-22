from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from trade_bot.config import BotConfig
from trade_bot.research.artifact_provenance import build_runtime_provenance

EVALUATION_CONTRACT_SCHEMA_VERSION = 1


def build_strategy_evaluation_contract(
    config: BotConfig,
    prices: pd.DataFrame,
) -> dict[str, object]:
    """Return the frozen assumptions required for comparable strategy metrics."""

    provenance = build_runtime_provenance(prices)
    price_input = provenance["price_input"]
    return {
        "schema_version": EVALUATION_CONTRACT_SCHEMA_VERSION,
        "return_engine": "close_to_close_total_return",
        "feature_observation": "session_close",
        "first_eligible_fill": "strictly_after_feature_close",
        "annualization_periods": 252,
        "execution": config.execution.model_dump(mode="json"),
        "data": {
            "start": config.data.start,
            "end": config.data.end,
            "adjusted": config.data.adjusted,
        },
        # The exact frame is part of the contract. A matching date range is not
        # enough when a feature model can consume the available price columns.
        "price_input": price_input,
        "source_tree_sha256": provenance["source_tree_sha256"],
        "poetry_lock_sha256": provenance["poetry_lock_sha256"],
    }


def evaluation_contract_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
