from __future__ import annotations

from pathlib import Path

import pandas as pd

from trade_bot.backtest.engine import BacktestResult
from trade_bot.config import StrategyConfig
from trade_bot.research.experiments import ExperimentCandidate, build_experiment_scorecard
from trade_bot.tax.account import TaxAccountProfile, TaxLossHarvestConfig
from trade_bot.tax.backtest import simulate_taxable_backtest
from trade_bot.tax.harvesting import find_tax_loss_harvest_candidates
from trade_bot.tax.lots import TaxLotLedger
from trade_bot.trading.journal import TradeJournal


def test_tax_lot_ledger_uses_tax_min_specific_id_and_classifies_term() -> None:
    ledger = TaxLotLedger(TaxAccountProfile(lot_selection_method="specific_id_tax_min"))
    ledger.process_execution(
        execution_id="buy_low",
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="BUY",
        quantity=10,
        price=100,
        executed_at="2024-01-01",
    )
    ledger.process_execution(
        execution_id="buy_high",
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="BUY",
        quantity=10,
        price=120,
        executed_at="2024-02-01",
    )

    ledger.process_execution(
        execution_id="sell",
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="SELL",
        quantity=10,
        price=130,
        executed_at="2025-03-01",
    )

    realized = ledger.realized_lots_frame().iloc[0]
    assert realized["source_execution_id"] == "buy_high"
    assert realized["term"] == "long"
    assert realized["realized_gain_loss"] == 100.0


def test_wash_sale_detection_disallows_replacement_loss() -> None:
    ledger = TaxLotLedger(TaxAccountProfile(wash_sale_window_days=30))
    ledger.process_execution(
        execution_id="buy_original",
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="BUY",
        quantity=10,
        price=100,
        executed_at="2026-01-01",
    )
    ledger.process_execution(
        execution_id="sell_loss",
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="SELL",
        quantity=10,
        price=90,
        executed_at="2026-01-10",
    )
    ledger.process_execution(
        execution_id="buy_replacement",
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="BUY",
        quantity=10,
        price=91,
        executed_at="2026-01-20",
    )

    ledger.apply_wash_sale_rules()

    realized = ledger.realized_lots_frame().iloc[0]
    assert realized["realized_gain_loss"] == -100.0
    assert realized["wash_sale_disallowed_loss"] == 100.0
    assert realized["taxable_gain_loss"] == 0.0


def test_tax_loss_harvest_candidates_filter_by_amount_and_pct() -> None:
    ledger = TaxLotLedger()
    ledger.process_execution(
        execution_id="buy",
        mode="paper",
        account="taxable",
        ticker="SPY",
        side="BUY",
        quantity=10,
        price=100,
        executed_at="2026-01-01",
    )

    candidates = find_tax_loss_harvest_candidates(
        ledger.open_lots_frame(),
        {"SPY": 90.0},
        TaxLossHarvestConfig(min_loss_amount=50, min_loss_pct=0.05),
        substitute_map={"SPY": ["VOO", "IVV"]},
        as_of="2026-02-01",
    )

    assert len(candidates) == 1
    assert candidates.iloc[0]["ticker"] == "SPY"
    assert candidates.iloc[0]["unrealized_gain_loss"] == -100.0
    assert "VOO" in candidates.iloc[0]["substitute_candidates"]


def test_taxable_backtest_reports_tax_drag_and_tax_deferred_profile_is_neutral() -> None:
    result, prices = _simple_gain_result()

    taxable = simulate_taxable_backtest(result, prices, TaxAccountProfile())
    deferred = simulate_taxable_backtest(result, prices, TaxAccountProfile(account_type="ira"))

    assert taxable.summary["tax_model_status"] == "taxable_estimated"
    assert taxable.summary["total_tax_liability"] > 0
    assert taxable.summary["after_tax_final_equity"] < result.equity.iloc[-1]
    assert taxable.summary["tax_drag_bps_per_year"] > 0
    assert deferred.summary["total_tax_liability"] == 0
    assert deferred.summary["after_tax_final_equity"] == result.equity.iloc[-1]


def test_scorecard_accepts_after_tax_metrics() -> None:
    candidate = ExperimentCandidate(
        name="tax_candidate",
        hypothesis="Tax enrichment smoke test.",
        role="candidate",
        family="tax",
        strategy=StrategyConfig(
            type="fixed_allocation", tickers=["QQQ"], allocation_weights={"QQQ": 1.0}
        ),
    )
    metrics = pd.DataFrame(
        {
            "cagr": [0.14],
            "sharpe": [1.0],
            "sortino": [1.2],
            "max_drawdown": [-0.20],
            "calmar": [0.7],
            "average_turnover": [0.05],
        },
        index=pd.Index(["tax_candidate"], name="name"),
    )
    window_summary = pd.DataFrame(
        {
            "worst_cagr": [-0.02, 0.01, 0.03],
            "positive_window_rate": [0.8, 0.8, 0.8],
        },
        index=pd.MultiIndex.from_product(
            [["tax_candidate"], ["1y", "3y", "5y"]],
            names=["strategy", "window"],
        ),
    )
    tax_metrics = pd.DataFrame(
        {
            "tax_model_status": ["taxable_estimated"],
            "tax_account_type": ["taxable"],
            "after_tax_final_equity": [125000.0],
            "after_tax_cagr": [0.12],
            "after_tax_max_drawdown": [-0.20],
            "after_tax_calmar": [0.6],
            "tax_drag_bps_per_year": [200.0],
            "total_tax_liability": [1000.0],
            "total_tax_benefit": [0.0],
            "net_estimated_tax_paid": [1000.0],
            "realized_short_term_gain": [1000.0],
            "realized_long_term_gain": [0.0],
            "realized_loss_harvested": [0.0],
            "wash_sale_disallowed_loss": [0.0],
            "loss_carryforward_end": [0.0],
            "short_term_gain_share": [1.0],
        },
        index=pd.Index(["tax_candidate"], name="strategy"),
    )

    scorecard = build_experiment_scorecard(
        (candidate,),
        metrics,
        window_summary,
        tax_metrics=tax_metrics,
    )

    assert scorecard.iloc[0]["tax_model_status"] == "taxable_estimated"
    assert "after_tax_growth_constrained_utility_score" in scorecard.columns


def test_trade_journal_rebuilds_tax_lots(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "journal.sqlite")
    journal.log_execution(
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="BUY",
        quantity=10,
        price=100,
        executed_at_utc="2026-01-01T16:00:00+00:00",
    )
    journal.log_execution(
        mode="paper",
        account="taxable",
        ticker="QQQ",
        side="SELL",
        quantity=4,
        price=110,
        executed_at_utc="2026-02-01T16:00:00+00:00",
    )

    rebuilt = journal.rebuild_tax_lots(mode="paper", account="taxable")
    stored_open = journal.load_tax_lots(mode="paper", account="taxable")
    stored_realized = journal.load_tax_realized_lots(mode="paper", account="taxable")

    assert rebuilt["open_lots"].iloc[0]["remaining_quantity"] == 6
    assert stored_open.iloc[0]["remaining_quantity"] == 6
    assert stored_realized.iloc[0]["realized_gain_loss"] == 40


def _simple_gain_result() -> tuple[BacktestResult, pd.DataFrame]:
    index = pd.to_datetime(["2026-01-02", "2026-06-01", "2026-12-31"])
    prices = pd.DataFrame({"QQQ": [100.0, 110.0, 120.0]}, index=index)
    weights = pd.DataFrame({"QQQ": [1.0, 1.0, 0.0]}, index=index)
    returns = pd.Series([0.0, 0.10, 1200.0 / 1100.0 - 1.0], index=index, name="tax_test")
    equity = pd.Series([1000.0, 1100.0, 1200.0], index=index, name="tax_test")
    zeros = pd.Series(0.0, index=index, name="tax_test")
    return (
        BacktestResult(
            name="tax_test",
            equity=equity,
            returns=returns,
            gross_returns=returns,
            weights=weights,
            target_weights=weights,
            turnover=zeros,
            transaction_costs=zeros,
        ),
        prices,
    )
