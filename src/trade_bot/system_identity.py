"""Read the configured native strategy identity without refreshing data or trading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trade_bot.config import load_config


def identity(root=Path(".")):
    baseline = root / "configs/baseline.yaml"
    active = root / "configs/active_trading.yaml"
    a, b = load_config(baseline), load_config(active)
    return {"schema_version": 1, "system": "trade-bot", "role": "strategic_portfolio",
        "primary": a.primary_strategy, "operating_primary": b.primary_strategy,
        "configuration_consistent": a.primary_strategy == b.primary_strategy,
        "primary_source": "configs/baseline.yaml:primary_strategy",
        "native_clock": "configured_lagged_native_execution",
        "configuration_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (baseline, active)},
        "action_authorized": False, "portfolio_allocation_allowed": False}


if __name__ == "__main__":
    print(json.dumps(identity(), sort_keys=True))
