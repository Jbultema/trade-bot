"""Read-only telemetry for the frozen Portfolio Lab control, not live orders."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import apply_drawdown_control, apply_volatility_target
from trade_bot.config import load_config
from trade_bot.features.indicators import daily_returns
from trade_bot.research.adaptive_response_lab import Context, Features, Policy, build_policy_targets
from trade_bot.research.macro_desk_export import external_research_context, macro_context
from trade_bot.storage.run_store import RunStore
from trade_bot.storage.warehouse import TradingWarehouse


def signals(prices, strategy):
    prices = prices.copy()
    prices["CASH"] = 1.0
    context = Context("technology", prices, strategy, "QQQ", prices.index[252], "", "")
    raw, diagnostic = build_policy_targets(Features(context), Policy("trade_primary", "control"))
    returns = daily_returns(prices)
    vol = apply_volatility_target(raw, returns, strategy.volatility_target) if strategy.volatility_target else raw
    guarded = apply_drawdown_control(vol, returns, strategy.drawdown_control) if strategy.drawdown_control else vol
    risky = strategy.tickers
    result = diagnostic[["stress", "scheduled"]].copy()
    result["raw_exposure"] = raw.reindex(columns=risky, fill_value=0).sum(axis=1)
    result["vol_exposure"] = vol.reindex(columns=risky, fill_value=0).sum(axis=1)
    result["guard_exposure"] = guarded.reindex(columns=risky, fill_value=0).sum(axis=1)
    result["vol_scale"] = result.vol_exposure.div(result.raw_exposure.replace(0, float("nan")))
    result["drawdown_scale"] = result.guard_exposure.div(result.vol_exposure.replace(0, float("nan")))
    result["reason"] = "scheduled_selection_held"
    result.loc[result.scheduled, "reason"] = "scheduled_selection_review"
    result.loc[result.vol_scale.lt(0.999), "reason"] = "volatility_budget_reduced"
    result.loc[result.drawdown_scale.lt(0.999), "reason"] = "shadow_equity_drawdown_guard"
    final = guarded.reindex(columns=risky, fill_value=0).clip(lower=0)
    final = final.div(final.sum(axis=1).clip(lower=1), axis=0)
    for ticker in final:
        result[f"target_{ticker}"] = final[ticker]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    config = load_config("configs/baseline.yaml")
    strategy = config.strategies[config.primary_strategy]
    prices = pd.read_parquet(args.bundle / "prices.parquet").loc[:args.cutoff]
    result = signals(prices, strategy)
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(args.output / "signals.parquet")
    snapshots = RunStore("data/run_store/trade_bot.duckdb", read_only=True).list_snapshots(limit=500)
    columns = [c for c in ("run_id", "market_date", "created_at_utc", "stored_at_utc", "risk_status", "recommended_action", "risk_budget_multiplier", "combined_config_hash") if c in snapshots]
    snapshots[columns].to_parquet(args.output / "operating_history.parquet", index=False)
    cycle = TradingWarehouse(read_only=True).cycle_tracker_runs(limit=10)
    cycle_columns = [c for c in ("market_date", "created_at_utc", "dominant_phase", "dominant_phase_probability", "readout") if c in cycle]
    (args.output / "cycle_context.json").write_text(cycle[cycle_columns].to_json(orient="records", indent=2) + "\n")
    (args.output / "macro_context.json").write_text(json.dumps(macro_context(), indent=2, allow_nan=False) + "\n")
    (args.output / "external_research_context.json").write_text(json.dumps(external_research_context(), indent=2, allow_nan=False) + "\n")
    (args.output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "system": "trade-bot",
        "signals_sha256": hashlib.sha256((args.output / "signals.parquet").read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
        "semantics": "frozen_control_reconstruction_not_operating_decision",
        "driver_columns": ["scheduled", "raw_exposure", "vol_scale", "drawdown_scale"],
        "monitor_columns": ["stress"],
        "rules": {"strategy": config.primary_strategy, "configuration": strategy.model_dump(mode="json"),
                  "control_selection_cadence": "Wednesday", "native_volatility_and_drawdown_lag_sessions": 1},
        "context_artifacts": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in ("operating_history.parquet", "cycle_context.json", "macro_context.json", "external_research_context.json")},
        "limitations": ["Stress is a monitor, not a trigger in this control.", "Selected modern technology universe is not survivor-free.", "Shadow-equity drawdown is not an executed account drawdown."],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
