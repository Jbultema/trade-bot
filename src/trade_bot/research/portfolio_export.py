"""Versioned research export; never changes the operating strategy."""

from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot.backtest.engine import apply_drawdown_control, apply_volatility_target
from trade_bot.config import load_config
from trade_bot.features.indicators import daily_returns
from trade_bot.research.adaptive_response_lab import Context, Features, Policy, build_policy_targets
from trade_bot.storage.run_store import RunStore

SYSTEM = "trade-bot"


def research_targets(prices, strategy):
    context = Context("technology", prices, strategy, "QQQ", prices.index[252], "", "")
    features = Features(context)
    candidates = [
        Policy("trade_primary", "control"),
        Policy(
            "trade_gentle_protection", "fast_protection", protection="continuous", reduction=0.20
        ),
        Policy(
            "trade_staged_recovery",
            "recovery",
            protection="recovery",
            reduction=0.50,
            exit_days=1,
            release_days=5,
            reentry_step=0.10,
        ),
    ]
    output = {}
    for policy in candidates:
        target, _ = build_policy_targets(features, policy)
        returns = daily_returns(prices)
        if strategy.volatility_target:
            target = apply_volatility_target(target, returns, strategy.volatility_target)
        if strategy.drawdown_control:
            target = apply_drawdown_control(target, returns, strategy.drawdown_control)
        target = target.reindex(columns=strategy.tickers, fill_value=0).clip(lower=0)
        target = target.div(target.sum(axis=1).clip(lower=1), axis=0)
        output[policy.name] = (
            target,
            context.evaluation_start,
            "primary" if policy.name == "trade_primary" else "challenger",
        )
    return output


def export_bundle(output):
    config = load_config("configs/baseline.yaml")
    snapshot, manifest = RunStore(
        "data/run_store/trade_bot.duckdb", read_only=True
    ).load_latest_snapshot(require_matching_config=False)
    prices = snapshot.prices.copy()
    prices["CASH"] = 1.0
    cash = (
        prices["BIL"].pct_change(fill_method=None).fillna(0)
        if "BIL" in prices
        else pd.Series(0.0, index=prices.index)
    )
    targets = research_targets(prices, config.strategies[config.primary_strategy])
    return _publish(
        output,
        prices.drop(columns="CASH"),
        targets,
        cash,
        [Path(manifest.artifact_path), Path("configs/baseline.yaml")],
        "native close-to-close lagged execution; exported controls recomputed causally at signal close",
        [
            "Common-clock reconstruction, not identical native primary returns.",
            "Native volatility and drawdown controls retained, re-clocked before common fills.",
            "Modern selected-stock universe is not survivor-free.",
            "Prior adaptive protection studies are reused, not new independent discoveries.",
            "CASH and BIL defensive weights are exported as unallocated cash.",
        ],
        "trade_primary",
    )


def _publish(output, prices, targets, cash, source_inputs, native_execution, notes, primary):
    import hashlib
    import json
    from datetime import UTC, datetime
    from importlib.metadata import version

    output.mkdir(parents=True, exist_ok=False)
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or prices.index.hasnans or prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing or not prices.columns.is_unique
        or np.isinf(prices.to_numpy(dtype=float)).any() or prices.le(0).any().any()
    ):
        raise ValueError("Invalid source session index")
    if (
        not cash.index.equals(prices.index) or not np.isfinite(cash).all()
        or cash.le(-1).any()
    ):
        raise ValueError("Cash observations must be finite and aligned")
    prices.to_parquet(output / "prices.parquet")
    cash.reindex(prices.index).fillna(0).rename("cash_return").to_frame().to_parquet(
        output / "cash_returns.parquet"
    )
    experts = []
    for name, (weights, evaluation_start, role) in targets.items():
        if (
            not weights.index.equals(prices.index) or not weights.columns.is_unique
            or not set(weights) <= set(prices)
            or not np.isfinite(weights.to_numpy(dtype=float)).all()
            or weights.lt(-1e-12).any().any()
            or weights.sum(axis=1).gt(1 + 1e-8).any()
        ):
            raise ValueError(f"Invalid native targets: {name}")
        if (weights.gt(1e-12) & prices.reindex(columns=weights.columns).isna()).any().any():
            raise ValueError(f"Unpriced native target: {name}")
        filename = f"{name}.parquet"
        weights.to_parquet(output / filename)
        experts.append(
            {
                "id": name,
                "weights_file": filename,
                "evaluation_start": str(pd.Timestamp(evaluation_start).date()),
                "role": role,
                "evidence_status": "retrospective_research",
                "target_semantics": "close_known_risky_budgets",
            }
        )
    artifacts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    code = hashlib.sha256()
    for p in sorted(Path("src").rglob("*.py")):
        code.update(str(p).encode())
        code.update(p.read_bytes())
    source_inputs = [*source_inputs, *[
        p for p in (Path("pyproject.toml"), Path("poetry.lock"))
        if p.is_file() and p not in source_inputs
    ]]
    manifest = {
        "schema_version": 1,
        "runtime_versions": {name: version(name) for name in ("numpy", "pandas", "pyarrow")},
        "system": SYSTEM,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_end": str(prices.index.max().date()),
        "primary_expert": primary,
        "native_execution": native_execution,
        "authority": "research_only",
        "automatic_promotion_allowed": False,
        "code_sha256": code.hexdigest(),
        "experts": experts,
        "source_inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in source_inputs
        ],
        "artifacts": artifacts,
        "notes": notes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output / "manifest.json"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Research-only Portfolio Lab specialist export")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(export_bundle(args.output))


if __name__ == "__main__":
    main()
