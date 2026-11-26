"""Fixed native staged-risk slate on the actual primary and repaired challenger."""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import (
    _rebalance_weights,
    apply_drawdown_control,
    apply_volatility_target,
    run_backtest,
)
from trade_bot.config import load_config
from trade_bot.features.indicators import daily_returns
from trade_bot.storage.run_store import RunStore
from trade_bot.strategies.momentum import build_strategy_weights


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    config = load_config("configs/baseline.yaml")
    source, snapshot = RunStore("data/run_store/trade_bot.duckdb", read_only=True).load_latest_snapshot(require_matching_config=False)
    prices = source.prices.copy()
    parents = [config.primary_strategy, "i111_native_risk_repair_guard17_relief85_ai85_div"]
    specs = []
    for parent in parents:
        specs.append({"name": parent + "__unchanged", "parent": parent, "kind": "unchanged"})
        for mode in ("gentle", "confirmed", "repair"):
            for lookback in (126, 252):
                for guard in ("retained", "replaced"):
                    specs.append({"name": f"{parent}__{mode}_{lookback}_{guard}", "parent": parent,
                        "kind": "native_staged", "mode": mode, "lookback": lookback,
                        "incumbent_drawdown_guard": guard})
        for scale in (.65, .85):
            specs.append({"name": f"{parent}__static{int(scale*100)}", "parent": parent, "kind": "static", "scale": scale})
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"specifications": specs, "selection": "none_fixed_slate_all_results_reported",
        "historical_status": "previously_examined_not_new_holdout", "native_execution": config.execution.model_dump(),
        "comparison": "native_engine_separately_from_funded_common_clock",
        "signal_rule": "SPY trailing drawdown plus QQQ trend, credit and participation confirmation",
        "risk_objective": "reduce persistent hard defense without surrendering crash protection",
        "cost_bps": [10, 25, 50], "extra_delay_sessions": [0, 1],
        "primary_changed": False, "action_authorized": False}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, exports = [], []
    available = prices[["SPY", "QQQ", "HYG", "LQD", "RSP"]].notna().rolling(252).sum().eq(252).all(axis=1)
    if not available.any():
        raise ValueError("No common initialized market/credit/participation evaluation window")
    start = max(prices.index[252], available[available].index[0])
    for spec in specs:
        parent = config.strategies[spec["parent"]]
        strategy = parent
        if spec["kind"] == "native_staged":
            strategy = parent.model_copy(update={"type": "staged_dual_momentum",
                "staged_risk_base": parent.type, "staged_risk_mode": spec["mode"],
                "staged_risk_lookback": spec["lookback"]})
            if spec["incumbent_drawdown_guard"] == "replaced":
                strategy = strategy.model_copy(update={"drawdown_control": None})
        target = build_strategy_weights(prices, strategy)
        if spec["kind"] == "static":
            risk = [t for t in target if t != strategy.defensive_ticker]
            released = target[risk].sum(axis=1) * (1-spec["scale"])
            target[risk] *= spec["scale"]
            if strategy.defensive_ticker:
                target[strategy.defensive_ticker] += released
        for cost in (10, 25, 50):
            for delay in (0, 1):
                execution = config.execution.model_copy(update={"transaction_cost_bps": cost,
                    "signal_lag_days": config.execution.signal_lag_days + delay})
                result = run_backtest(spec["name"], prices, target, execution,
                    volatility_target=strategy.volatility_target, drawdown_control=strategy.drawdown_control)
                values = result.returns.loc[start:]
                for period, left, right in (("full", None, None), ("pre2019", None, "2018-12-31"), ("later", "2019-01-01", None)):
                    r = values.loc[left:right]
                    equity = r.add(1).cumprod()
                    rows.append({"candidate": spec["name"], "period": period, "cost_bps": cost,
                        "extra_delay": delay, "sessions": len(r), "cagr": equity.iloc[-1] ** (252/len(r))-1,
                        "max_drawdown": (equity/equity.cummax().clip(lower=1)-1).min()})
        scheduled = _rebalance_weights(target, config.execution.rebalance)
        returns = daily_returns(prices)
        if strategy.volatility_target:
            scheduled = apply_volatility_target(scheduled, returns, strategy.volatility_target)
        if strategy.drawdown_control:
            scheduled = apply_drawdown_control(scheduled, returns, strategy.drawdown_control)
        risk = [t for t in scheduled if t not in {strategy.defensive_ticker, "CASH"}]
        file = spec["name"] + ".parquet"
        scheduled[risk].loc[start:].to_parquet(output / file)
        exports.append(spec | {"weights_file": file})
    prices.loc[start:].drop(columns="CASH", errors="ignore").to_parquet(output / "prices.parquet")
    prices.BIL.pct_change(fill_method=None).fillna(0).loc[start:].rename("cash_return").to_frame().to_parquet(output / "cash.parquet")
    pd.DataFrame(rows).to_csv(output / "native_metrics.csv", index=False)
    sources = [*sorted(Path("src/trade_bot").rglob("*.py")), Path("configs/baseline.yaml"),
        Path("pyproject.toml"), Path("poetry.lock"), Path(snapshot.artifact_path)]
    manifest = {"kind": "native_staged_cash_v1", "candidates": exports,
        "sources": {str(p.resolve()): digest(p) for p in sources},
        "environment": {name: version(name) for name in ("pandas", "numpy", "pyarrow", "pydantic")},
        "artifacts": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "native_paths": len(specs)*6, "portfolio_allocation_allowed": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"candidates": len(specs), "native_paths": len(specs)*6, "end": str(prices.index[-1].date())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
