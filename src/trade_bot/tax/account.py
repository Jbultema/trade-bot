from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trade_bot.DEFAULTS import (
    DEFAULT_TAX_ACCOUNT_TYPE,
    DEFAULT_TAX_ANNUAL_LOSS_DEDUCTION_LIMIT,
    DEFAULT_TAX_CAPITAL_LOSS_CARRYFORWARD_LONG,
    DEFAULT_TAX_CAPITAL_LOSS_CARRYFORWARD_SHORT,
    DEFAULT_TAX_FEDERAL_LONG_TERM_RATE,
    DEFAULT_TAX_FEDERAL_SHORT_TERM_RATE,
    DEFAULT_TAX_HARVEST_COOLDOWN_DAYS,
    DEFAULT_TAX_LONG_TERM_HOLDING_DAYS,
    DEFAULT_TAX_LOT_SELECTION_METHOD,
    DEFAULT_TAX_MIN_LOSS_HARVEST_AMOUNT,
    DEFAULT_TAX_MIN_LOSS_HARVEST_PCT,
    DEFAULT_TAX_NIIT_APPLIES,
    DEFAULT_TAX_NIIT_RATE,
    DEFAULT_TAX_STATE_LONG_TERM_RATE,
    DEFAULT_TAX_STATE_SHORT_TERM_RATE,
    DEFAULT_TAX_WASH_SALE_ENFORCEMENT,
    DEFAULT_TAX_WASH_SALE_WINDOW_DAYS,
)

AccountType = Literal["ira", "roth", "taxable"]
LotSelectionMethod = Literal["fifo", "specific_id_tax_min", "highest_cost", "lowest_gain"]
WashSaleEnforcement = Literal["off", "warn", "block_loss_harvest", "strict"]


@dataclass(frozen=True)
class TaxAccountProfile:
    """Configurable tax assumptions for estimated research calculations."""

    account_type: AccountType = DEFAULT_TAX_ACCOUNT_TYPE  # type: ignore[assignment]
    federal_short_term_tax_rate: float = DEFAULT_TAX_FEDERAL_SHORT_TERM_RATE
    federal_long_term_tax_rate: float = DEFAULT_TAX_FEDERAL_LONG_TERM_RATE
    state_short_term_tax_rate: float = DEFAULT_TAX_STATE_SHORT_TERM_RATE
    state_long_term_tax_rate: float = DEFAULT_TAX_STATE_LONG_TERM_RATE
    niit_rate: float = DEFAULT_TAX_NIIT_RATE
    niit_applies: bool = DEFAULT_TAX_NIIT_APPLIES
    capital_loss_carryforward_short: float = DEFAULT_TAX_CAPITAL_LOSS_CARRYFORWARD_SHORT
    capital_loss_carryforward_long: float = DEFAULT_TAX_CAPITAL_LOSS_CARRYFORWARD_LONG
    annual_loss_deduction_limit: float = DEFAULT_TAX_ANNUAL_LOSS_DEDUCTION_LIMIT
    long_term_holding_period_days: int = DEFAULT_TAX_LONG_TERM_HOLDING_DAYS
    lot_selection_method: LotSelectionMethod = DEFAULT_TAX_LOT_SELECTION_METHOD  # type: ignore[assignment]
    wash_sale_window_days: int = DEFAULT_TAX_WASH_SALE_WINDOW_DAYS
    wash_sale_enforcement: WashSaleEnforcement = DEFAULT_TAX_WASH_SALE_ENFORCEMENT  # type: ignore[assignment]

    @property
    def is_taxable(self) -> bool:
        return self.account_type == "taxable"

    @property
    def short_term_tax_rate(self) -> float:
        if not self.is_taxable:
            return 0.0
        niit = self.niit_rate if self.niit_applies else 0.0
        return self.federal_short_term_tax_rate + self.state_short_term_tax_rate + niit

    @property
    def long_term_tax_rate(self) -> float:
        if not self.is_taxable:
            return 0.0
        niit = self.niit_rate if self.niit_applies else 0.0
        return self.federal_long_term_tax_rate + self.state_long_term_tax_rate + niit

    @property
    def starting_loss_carryforward(self) -> float:
        return max(0.0, self.capital_loss_carryforward_short) + max(
            0.0,
            self.capital_loss_carryforward_long,
        )


@dataclass(frozen=True)
class TaxLossHarvestConfig:
    min_loss_amount: float = DEFAULT_TAX_MIN_LOSS_HARVEST_AMOUNT
    min_loss_pct: float = DEFAULT_TAX_MIN_LOSS_HARVEST_PCT
    cooldown_days: int = DEFAULT_TAX_HARVEST_COOLDOWN_DAYS
    wash_sale_window_days: int = DEFAULT_TAX_WASH_SALE_WINDOW_DAYS
    wash_sale_enforcement: WashSaleEnforcement = DEFAULT_TAX_WASH_SALE_ENFORCEMENT  # type: ignore[assignment]
