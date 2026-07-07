"""Centralized reusable defaults for the local trade-bot project.

This module is the default-value registry for code that should stay coordinated
across the app, CLI, backtests, research jobs, ML diagnostics, and dashboard.

Use this file for reusable defaults that can affect behavior, performance,
risk, storage paths, model hyperparameters, or strategy universes. Keep purely
local schema versions, display labels, and one-off table help dictionaries in
their owning modules unless multiple modules need the same value.

Naming convention:
- `DEFAULT_*` means a configurable behavioral default.
- Shared static vocabularies can use descriptive names when the value is not a
  parameter, for example `TRADING_DAYS_PER_YEAR`.
- New defaults should be grouped into one of the sections below and given a name
  specific enough that a future agent can tell what behavior it controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

# Global market calendar assumptions.
TRADING_DAYS_PER_YEAR = 252


# Paths, local stores, and app/runtime behavior.
DEFAULT_CONFIG_PATH = Path("configs/baseline.yaml")
DEFAULT_EVENTS_PATH = Path("configs/events.yaml")
DEFAULT_MACRO_PATH = Path("configs/macro_fred.yaml")
DEFAULT_NEWS_PATH = Path("configs/news_sources.yaml")
DEFAULT_REPORT_PATH = Path("reports/baseline_report.html")
DEFAULT_EXPERIMENTS_DIR = Path("reports/experiments")
DEFAULT_RESET_EXPERIMENTS_DIR = Path("data/experiments_reset_v2")
DEFAULT_ML_DIAGNOSTICS_DIR = Path("data/ml_diagnostics/latest")
DEFAULT_SIGNAL_EVIDENCE_DIR = Path("reports/signal_evidence")
DEFAULT_JOURNAL_PATH = Path("data/trading_journal.sqlite")
DEFAULT_RUN_STORE_DB_PATH = Path("data/run_store/trade_bot.duckdb")
DEFAULT_RUN_STORE_ARTIFACT_DIR = Path("data/run_store/snapshots")
DEFAULT_RUN_STORE_JOB_LOG_DIR = Path("data/run_store/jobs")
DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS = 15
DEFAULT_SCENARIO_HISTORY_SNAPSHOT_LIMIT = 120
DEFAULT_MONITORING_TOP_N = 5
DEFAULT_MONITORING_COHORT_START_DATE = "2026-01-01"
DEFAULT_MONITORING_ENVELOPE_WATCH_SHARE = 0.50
DEFAULT_MONITORING_ENVELOPE_REVIEW_SHARE = 0.85
DEFAULT_MONITORING_ENVELOPE_BREACH_SHARE = 1.00
DEFAULT_EXPERIMENT_REGISTRY_LIMIT = 500
DEFAULT_CURATED_SHELF_LIMIT = 25
DEFAULT_REFERENCE_BASELINE_STRATEGIES = frozenset(
    {
        "buy_hold_spy",
        "buy_hold_qqq",
        "buy_hold_bil",
        "buy_hold_cash",
        "cash",
        "bil",
        "i41_ref_us_60_40",
        "i41_ref_global_risk_sleeves",
    }
)
DEFAULT_DEFAULT_APPROACH_RESEARCH_STATUSES = (
    "operational_candidate",
    "needs_iteration",
    "reference",
)
DEFAULT_DASHBOARD_SECTIONS = (
    "Command Center",
    "Risk & Scenarios",
    "News & Macro",
    "Research Lab",
    "Simulation Lab",
    "Launch Lab",
    "Performance",
    "Monitoring",
    "Forward Test",
)


# Paper/live ticketing and book-alignment defaults.
DEFAULT_TICKET_PRICE_BAND_PCT = 0.0075
DEFAULT_TICKET_SIZE_BAND_PCT = 0.20
DEFAULT_TICKET_MIN_TRADE_NOTIONAL = 25.0
DEFAULT_TICKET_WHOLE_SHARES = True
DEFAULT_FORWARD_TEST_ACCOUNT = "default_paper_account"
DEFAULT_FORWARD_TEST_STRATEGY = "scenario_adjusted_trade_decision"
DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT = 0.02

# Tax-aware account modeling defaults. These are configurable research
# assumptions, not tax advice. User-specific rates belong in local config, and
# broker-reported lots should be reconciled before real taxable decisions.
DEFAULT_TAX_ACCOUNT_TYPE = "ira"
DEFAULT_TAX_FEDERAL_SHORT_TERM_RATE = 0.24
DEFAULT_TAX_FEDERAL_LONG_TERM_RATE = 0.15
DEFAULT_TAX_STATE_SHORT_TERM_RATE = 0.0
DEFAULT_TAX_STATE_LONG_TERM_RATE = 0.0
DEFAULT_TAX_NIIT_RATE = 0.0
DEFAULT_TAX_NIIT_APPLIES = False
DEFAULT_TAX_CAPITAL_LOSS_CARRYFORWARD_SHORT = 0.0
DEFAULT_TAX_CAPITAL_LOSS_CARRYFORWARD_LONG = 0.0
DEFAULT_TAX_ANNUAL_LOSS_DEDUCTION_LIMIT = 3_000.0
DEFAULT_TAX_LONG_TERM_HOLDING_DAYS = 365
DEFAULT_TAX_LOT_SELECTION_METHOD = "specific_id_tax_min"
DEFAULT_TAX_WASH_SALE_WINDOW_DAYS = 30
DEFAULT_TAX_WASH_SALE_ENFORCEMENT = "warn"
DEFAULT_TAX_MIN_LOSS_HARVEST_AMOUNT = 250.0
DEFAULT_TAX_MIN_LOSS_HARVEST_PCT = 0.05
DEFAULT_TAX_HARVEST_COOLDOWN_DAYS = 31
DEFAULT_TAX_LOT_QUANTITY_EPSILON = 1e-8
DEFAULT_TAX_BACKTEST_MIN_RECONSTRUCTED_QUANTITY = 1e-6


# Market data loading defaults.
DEFAULT_DATA_CACHE_DIR = "data/cache"
DEFAULT_DATA_ADJUSTED = True
DEFAULT_FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# Execution and backtest defaults.
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_REBALANCE = "W-WED"
DEFAULT_SIGNAL_LAG_DAYS = 1

# Owner-directed investable exclusions. These are hard local constraints used
# for data loading, generated candidates, and paper/live recommendation paths.
# Excluded tickers may still appear in watch-only risk/event proxy lists when
# they are useful warning signals, but they should not be proposed as holdings.
DEFAULT_EXCLUDED_TICKERS = frozenset({"ORCL"})


# Strategy research operability defaults. Material trades are allocation changes
# large enough to matter for a human-reviewed swing system. Weekly material
# trading is acceptable; the "too_twitchy" blocker should be reserved for
# materially faster churn or unusually large/constant reallocations.
DEFAULT_OPERABILITY_MATERIAL_TRADE_TURNOVER_THRESHOLD = 0.05
DEFAULT_OPERABILITY_PAPER_OPERABLE_MAX_TRADES_PER_YEAR = 26.0
DEFAULT_OPERABILITY_WEEKLY_CADENCE_MAX_TRADES_PER_YEAR = 60.0
DEFAULT_OPERABILITY_REVIEW_CHURN_MAX_TRADES_PER_YEAR = 90.0
DEFAULT_OPERABILITY_GOOD_MEAN_GAP_TRADING_DAYS = 5.0
DEFAULT_OPERABILITY_SCORE_FULL_CADENCE_TRADES_PER_YEAR = 26.0
DEFAULT_OPERABILITY_SCORE_ZERO_CADENCE_TRADES_PER_YEAR = 100.0
DEFAULT_OPERABILITY_PAPER_OPERABLE_MAX_SINGLE_DAY_TURNOVER = 0.65
DEFAULT_OPERABILITY_WEEKLY_CADENCE_MAX_SINGLE_DAY_TURNOVER = 0.90
DEFAULT_OPERABILITY_REVIEW_CHURN_MAX_SINGLE_DAY_TURNOVER = 1.10
DEFAULT_OPERABILITY_PAPER_OPERABLE_MAX_AVERAGE_TURNOVER = 0.08
DEFAULT_OPERABILITY_WEEKLY_CADENCE_MAX_AVERAGE_TURNOVER = 0.12
DEFAULT_OPERABILITY_SCORE_MAX_TURNOVER_START = 0.35
DEFAULT_OPERABILITY_SCORE_MAX_TURNOVER_ZERO = 1.00
DEFAULT_OPERABILITY_SCORE_AVERAGE_TURNOVER_START = 0.04
DEFAULT_OPERABILITY_SCORE_AVERAGE_TURNOVER_ZERO = 0.18


# Dashboard decision-timeline defaults. The timeline should expose the material
# allocation moves a human reviewer would care about, not every small rebalance.
DEFAULT_DECISION_TIMELINE_CONTEXT_DAYS = 21
DEFAULT_DECISION_TIMELINE_FORWARD_DAYS = 63
DEFAULT_DECISION_TIMELINE_MAX_EVENTS = 35


# Growth-constrained outcome utility defaults. These encode the retirement-style
# research objective: prefer high terminal wealth when drawdowns remain within a
# survivable band, while retaining hard guardrails against left-tail damage.
DEFAULT_OUTCOME_OBJECTIVE = "growth_constrained"
DEFAULT_OUTCOME_HORIZON_YEARS = 15
DEFAULT_OUTCOME_STARTING_ACCOUNT_VALUE = 320_000.0
DEFAULT_OUTCOME_ANNUAL_CONTRIBUTION = 40_000.0
DEFAULT_OUTCOME_SOFT_DRAWDOWN_LIMIT = -0.22
DEFAULT_OUTCOME_HARD_DRAWDOWN_LIMIT = -0.30
DEFAULT_OUTCOME_FLOOR_CAGR = 0.05
DEFAULT_OUTCOME_TARGET_CAGR = 0.15
DEFAULT_OUTCOME_MIN_WALK_FORWARD_POSITIVE_RATE = 0.65
DEFAULT_OUTCOME_MIN_WORST_3Y_CAGR = -0.05
DEFAULT_OUTCOME_MIN_LEFT_TAIL_REGIME_RETURN = -0.15
DEFAULT_OUTCOME_OVERFIT_PENALTY_WEIGHT = 0.10
DEFAULT_OUTCOME_CHURN_PENALTY_WEIGHT = 0.06
DEFAULT_OUTCOME_PEER_CURVE_METRIC_LIMIT = 75
# Split the annual accumulation budget into period-end deposits. Monthly is the
# default because it better approximates ongoing 401k/paycheck contributions.
DEFAULT_OUTCOME_CONTRIBUTION_TIMING = "monthly"
DEFAULT_OUTCOME_TRADING_DAYS_PER_YEAR = 252
DEFAULT_OUTCOME_BOOTSTRAP_PATHS = 750
DEFAULT_OUTCOME_BOOTSTRAP_BLOCK_DAYS = 21
DEFAULT_OUTCOME_BOOTSTRAP_RANDOM_SEED = 20260705
DEFAULT_FORWARD_SIMULATION_PATHS = 600
DEFAULT_FORWARD_SIMULATION_BLOCK_DAYS = 21
DEFAULT_FORWARD_SIMULATION_RANDOM_SEED = 20260705
DEFAULT_FORWARD_SIMULATION_INITIAL_SCENARIO_WEIGHT = 0.75
DEFAULT_FORWARD_SIMULATION_TRANSITION_SCENARIO_WEIGHT = 0.20
DEFAULT_FORWARD_SIMULATION_MIN_REGIME_OBSERVATIONS = 63
DEFAULT_SIMULATION_REFERENCE_STRATEGIES = (
    ("buy_hold_spy", "Hold SPY"),
    ("buy_hold_qqq", "Hold QQQ"),
)
DEFAULT_FORWARD_SIMULATION_FALLBACK_PROBABILITIES = {
    "risk_off": 0.15,
    "transition": 0.25,
    "risk_on_fragile": 0.15,
    "risk_on": 0.45,
}


# Backtest evaluation-window defaults.
DEFAULT_ROLLING_WINDOW_YEARS = (1, 3, 5)
DEFAULT_ROLLING_STEP_MONTHS = 1
DEFAULT_WINDOW_MIN_OBSERVATION_RATIO = 0.80
DEFAULT_CALENDAR_YEAR_MIN_OBSERVATIONS = 60
DEFAULT_REGIME_MIN_OBSERVATIONS = 20
DEFAULT_WALK_FORWARD_TRAIN_YEARS = 5
DEFAULT_WALK_FORWARD_TEST_YEARS = 1
DEFAULT_WALK_FORWARD_STEP_MONTHS = 6
DEFAULT_ENTRY_HORIZONS: dict[str, int] = {
    "3m": 63,
    "6m": 126,
    "1y": 252,
    "3y": 756,
    "5y": 1260,
}


# Launch-readiness defaults. Launch Lab answers a different question than the
# daily operating book: whether new paper/live capital should begin following a
# strategy now, and whether the entry should be immediate or staged.
DEFAULT_LAUNCH_CAPITAL = 1_000.0
DEFAULT_LAUNCH_TARGET_FRACTION = 1.0
DEFAULT_LAUNCH_INITIAL_RAMP_FRACTION = 0.25
DEFAULT_LAUNCH_RAMP_WEEKS = (0, 4, 8, 12)
DEFAULT_LAUNCH_START_FREQUENCY = "M"
DEFAULT_LAUNCH_PRIMARY_HORIZON = "6m"
DEFAULT_LAUNCH_BAD_START_DRAWDOWN = -0.08
DEFAULT_LAUNCH_READY_SCORE = 0.75
DEFAULT_LAUNCH_SET_SCORE = 0.55
DEFAULT_LAUNCH_WAIT_SCORE = 0.35
DEFAULT_LAUNCH_MIN_WINDOWS = 12


# Volatility targeting and drawdown-control defaults.
DEFAULT_VOL_TARGET_ANNUALIZED_VOLATILITY = 0.12
DEFAULT_VOL_TARGET_LOOKBACK_DAYS = 63
DEFAULT_VOL_TARGET_MAX_LEVERAGE = 1.0

DEFAULT_DRAWDOWN_EQUITY_LOOKBACK_DAYS = 252
DEFAULT_DRAWDOWN_MAX_DRAWDOWN = -0.10
DEFAULT_DRAWDOWN_RISK_MULTIPLIER = 0.50


# Momentum, dip-reentry, and cycle-overlay strategy defaults.
DEFAULT_MOVING_AVERAGE_DAYS = 200
DEFAULT_MOMENTUM_LOOKBACK_DAYS = 126
DEFAULT_MOMENTUM_SKIP_DAYS = 21
DEFAULT_TOP_N = 1
DEFAULT_MIN_RETURN = 0.0
DEFAULT_RANKING_METRIC: Final[Literal["return"]] = "return"
DEFAULT_WEIGHTING: Final[Literal["equal"]] = "equal"
DEFAULT_VOLATILITY_LOOKBACK_DAYS = 63
DEFAULT_TREND_FILTER_DAYS = None
DEFAULT_MAX_ASSET_WEIGHT = None

DEFAULT_STRATEGY_AI_GROWTH_TICKERS = frozenset(
    {
        "AAPL",
        "AMD",
        "AMZN",
        "ANET",
        "APP",
        "ARKK",
        "ARM",
        "ASML",
        "AVGO",
        "BOTZ",
        "CLOU",
        "CRWD",
        "DDOG",
        "DELL",
        "GOOG",
        "GOOGL",
        "IGV",
        "META",
        "MRVL",
        "MSFT",
        "MU",
        "NET",
        "NVDA",
        "PLTR",
        "QQQ",
        "QQQM",
        "ROBO",
        "SKYY",
        "SMCI",
        "SMH",
        "SNOW",
        "SOXX",
        "TSLA",
        "XLK",
        "XLC",
        "XLY",
    }
)
DEFAULT_STRATEGY_CYCLICAL_TICKERS = frozenset(
    {
        "BNO",
        "COWZ",
        "CPER",
        "DBC",
        "DIA",
        "IWB",
        "IWM",
        "IYT",
        "KBE",
        "KRE",
        "MDY",
        "RSP",
        "SPHB",
        "SPMO",
        "USO",
        "VTI",
        "VTV",
        "XES",
        "XHB",
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XME",
        "XOP",
        "XRT",
    }
)
DEFAULT_STRATEGY_DEFENSIVE_EQUITY_TICKERS = frozenset(
    {
        "MOAT",
        "QUAL",
        "SCHD",
        "SPLV",
        "USMV",
        "VIG",
        "XLRE",
        "XLP",
        "XLU",
        "XLV",
    }
)
DEFAULT_STRATEGY_DEFENSIVE_ALT_TICKERS = frozenset(
    {
        "AGG",
        "BIL",
        "BND",
        "BSV",
        "EDV",
        "GLD",
        "IAU",
        "IEF",
        "IEI",
        "LQD",
        "MUB",
        "SGOV",
        "SHY",
        "TIP",
        "TLT",
        "UUP",
        "USFR",
        "VCIT",
        "VCSH",
        "VGIT",
        "VGSH",
        "VGLT",
        "VTIP",
    }
)
DEFAULT_STRATEGY_GLOBAL_TICKERS = frozenset(
    {
        "EEM",
        "EFA",
        "EWA",
        "EWC",
        "EWJ",
        "EWU",
        "EWW",
        "EWZ",
        "FXE",
        "FXF",
        "FXY",
        "INDA",
        "MCHI",
        "VEA",
        "VGK",
        "VT",
        "VWO",
    }
)
DEFAULT_STRATEGY_SPECULATIVE_TICKERS = frozenset(
    {
        "ARKK",
        "BITB",
        "ETHE",
        "FBTC",
        "IBIT",
        "LIT",
        "SVXY",
        "TAN",
        "URA",
        "VIXY",
        "XBI",
    }
)

DEFAULT_DIP_LOOKBACK_DAYS = 252
DEFAULT_DIP_TRIGGER_DRAWDOWN = -0.12
DEFAULT_DIP_DEEP_DRAWDOWN = -0.25
DEFAULT_DIP_RECOVERY_DAYS = 21
DEFAULT_DIP_CONFIRMATION_DAYS = 5
DEFAULT_DIP_MIN_RECOVERY_RETURN = 0.015
DEFAULT_DIP_STARTER_WEIGHT = 0.20
DEFAULT_DIP_STEP_WEIGHT = 0.20
DEFAULT_DIP_MAX_RISK_WEIGHT = 0.80
DEFAULT_DIP_VOLATILITY_CEILING = 0.32
DEFAULT_DIP_CREDIT_CONFIRMATION = True
DEFAULT_DIP_BREADTH_CONFIRMATION = True

DEFAULT_CYCLE_SATELLITE_MAX_WEIGHT = 0.45
DEFAULT_CYCLE_SATELLITE_RISK_ON_WEIGHT = 0.35
DEFAULT_CYCLE_SATELLITE_REENTRY_WEIGHT = 0.45
DEFAULT_CYCLE_MIN_REBALANCE_CHANGE = 0.02
DEFAULT_CYCLE_MAX_STEP_CHANGE = 1.0
DEFAULT_CYCLE_MIN_HOLD_DAYS = 0
DEFAULT_CYCLE_RISK_OFF_OVERRIDE_CHANGE = 0.20

DEFAULT_MOMENTUM_STATE_LOOKBACK_DAYS = 126
DEFAULT_MOMENTUM_STATE_VOL_DAYS = 63
DEFAULT_MOMENTUM_STATE_SKIP_DAYS = 5


# Macro and signal-inclusion defaults.
DEFAULT_MACRO_SIGNAL_LOOKBACK_DAYS = 1260
DEFAULT_SIGNAL_INCLUSION_LOOKBACK_DAYS = 756
DEFAULT_SIGNAL_INCLUSION_DEFENSIVE_TICKER = "BIL"
DEFAULT_SIGNAL_INCLUSION_PUBLICATION_LAG_DAYS = 21
DEFAULT_SIGNAL_INCLUSION_MIN_OBSERVATIONS = 252
DEFAULT_SIGNAL_INCLUSION_PRESSURE_THRESHOLD = 0.65
DEFAULT_SIGNAL_INCLUSION_RISK_MULTIPLIER = 0.50


# Scenario and future-state sizing defaults.
DEFAULT_SCENARIO_STRESS_MULTIPLIER = 0.35
DEFAULT_SCENARIO_TRANSITION_MULTIPLIER = 0.65
DEFAULT_SCENARIO_FRAGILE_UPSIDE_MULTIPLIER = 0.80
DEFAULT_SCENARIO_RISK_ON_MULTIPLIER = 1.0
DEFAULT_SCENARIO_MIN_MULTIPLIER = 0.20
DEFAULT_SCENARIO_MAX_MULTIPLIER = 1.0
DEFAULT_SCENARIO_SIZING_LOOKBACK_DAYS = 63

# Classical ML, Bayesian scenario, and diagnostics defaults.
DEFAULT_SKLEARN_MODEL_NAMES = frozenset(
    {
        "sk_logit_l2",
        "sk_logit_l1",
        "sk_random_forest",
        "sk_extra_trees",
        "sk_gradient_boosting",
        "sk_calibrated_logit",
        "sk_ensemble",
    }
)
DEFAULT_SKLEARN_RANDOM_STATE = 17
DEFAULT_SKLEARN_N_ESTIMATORS = 160
DEFAULT_SKLEARN_MAX_DEPTH = 5
DEFAULT_SKLEARN_MIN_SAMPLES_LEAF = 20
DEFAULT_SKLEARN_MAX_ITER = 1000
DEFAULT_SKLEARN_REGULARIZATION_C = 0.65
DEFAULT_SKLEARN_CALIBRATION_SPLITS = 3

DEFAULT_ML_STANDARD_MODEL_NAMES = ("sk_logit_l2", "sk_random_forest")
DEFAULT_ML_RESEARCH_MODEL_NAMES = (
    "sk_logit_l2",
    "sk_logit_l1",
    "sk_random_forest",
    "sk_extra_trees",
    "sk_gradient_boosting",
    "sk_calibrated_logit",
)
DEFAULT_ML_MODEL_NAMES = DEFAULT_ML_STANDARD_MODEL_NAMES
DEFAULT_ML_HORIZONS = (21, 63)
DEFAULT_ML_RESEARCH_HORIZONS = (5, 21, 63)
DEFAULT_ML_STANDARD_STEP_DAYS = 126
DEFAULT_ML_RESEARCH_STEP_DAYS = 42
DEFAULT_ML_TASK_MIN_TRAIN_OBSERVATIONS = 252
DEFAULT_ML_TASK_TRAIN_WINDOW_DAYS = 756
DEFAULT_ML_TASK_STEP_DAYS = 21
DEFAULT_ML_SECTOR_TICKERS = (
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLRE",
    "XLC",
)

DEFAULT_FUTURE_STATE_COLUMNS = ("risk_off", "transition", "risk_on_fragile", "risk_on")
DEFAULT_STRATEGY_DRAWDOWN_COLUMNS = ("stable", "drawdown")
DEFAULT_FUTURE_STATE_MODEL_HORIZON_DAYS = 21
DEFAULT_FUTURE_STATE_FEATURE_SET = "core"
DEFAULT_FUTURE_STATE_TRAIN_WINDOW_DAYS = 756
DEFAULT_FUTURE_STATE_MIN_TRAIN_OBSERVATIONS = 252
DEFAULT_FUTURE_STATE_REFIT_EVERY_DAYS = 21
DEFAULT_FUTURE_STATE_K_NEIGHBORS = 80
DEFAULT_FUTURE_STATE_BAG_COUNT = 9
DEFAULT_FUTURE_STATE_RIDGE_ALPHA = 0.15
DEFAULT_FUTURE_STATE_RIDGE_LEARNING_RATE = 0.08
DEFAULT_FUTURE_STATE_RIDGE_ITERATIONS = 140
DEFAULT_FUTURE_STATE_PROBABILITY_SMOOTHING = 0.08
DEFAULT_FUTURE_STATE_DIRICHLET_PRIOR_STRENGTH = 8.0
DEFAULT_FUTURE_STATE_RECENCY_HALF_LIFE_DAYS = 252
DEFAULT_FUTURE_STATE_BAYESIAN_VARIANCE_FLOOR = 0.25
DEFAULT_FUTURE_STATE_BAYESIAN_FEATURE_SHRINKAGE = 12.0
DEFAULT_FUTURE_STATE_SKLEARN_N_ESTIMATORS = 140
DEFAULT_FUTURE_STATE_SKLEARN_MAX_DEPTH = 5
DEFAULT_FUTURE_STATE_SKLEARN_MIN_SAMPLES_LEAF = 20
DEFAULT_FUTURE_STATE_SKLEARN_REGULARIZATION_C = 0.70
DEFAULT_FUTURE_STATE_SKLEARN_RANDOM_STATE = 17
DEFAULT_FUTURE_STATE_RISK_OFF_ACTIVATION_PROBABILITY = 0.0
DEFAULT_FUTURE_STATE_TRANSITION_ACTIVATION_PROBABILITY = 0.0
DEFAULT_FUTURE_STATE_FRAGILE_ACTIVATION_PROBABILITY = 0.0

DEFAULT_STRATEGY_DRAWDOWN_MODEL_HORIZON_DAYS = 21
DEFAULT_STRATEGY_DRAWDOWN_FEATURE_SET = "ai"
DEFAULT_STRATEGY_DRAWDOWN_TRAIN_WINDOW_DAYS = 756
DEFAULT_STRATEGY_DRAWDOWN_MIN_TRAIN_OBSERVATIONS = 252
DEFAULT_STRATEGY_DRAWDOWN_REFIT_EVERY_DAYS = 126
DEFAULT_STRATEGY_DRAWDOWN_FUTURE_DRAWDOWN_THRESHOLD = -0.08
DEFAULT_STRATEGY_DRAWDOWN_ACTIVATION_PROBABILITY = 0.42
DEFAULT_STRATEGY_DRAWDOWN_STRESS_MULTIPLIER = 0.62
DEFAULT_STRATEGY_DRAWDOWN_MIN_MULTIPLIER = 0.55
DEFAULT_STRATEGY_DRAWDOWN_PROBABILITY_SMOOTHING = 0.08
DEFAULT_STRATEGY_DRAWDOWN_SKLEARN_N_ESTIMATORS = 48
DEFAULT_STRATEGY_DRAWDOWN_SKLEARN_MAX_DEPTH = 4
DEFAULT_STRATEGY_DRAWDOWN_SKLEARN_MIN_SAMPLES_LEAF = 24
DEFAULT_STRATEGY_DRAWDOWN_SKLEARN_REGULARIZATION_C = 0.70
DEFAULT_STRATEGY_DRAWDOWN_SKLEARN_RANDOM_STATE = 29


# News and event-risk defaults.
DEFAULT_NEWS_CACHE_FILE = "news_items.json"
DEFAULT_NEWS_USER_AGENT = "trade-bot-news-monitor/0.1"
DEFAULT_NEWS_SOURCE_TYPE = "rss"
DEFAULT_NEWS_SOURCE_PRIORITY = 3
DEFAULT_NEWS_SOURCE_ENABLED = True
DEFAULT_NEWS_MAX_AGE_MINUTES = 120
DEFAULT_NEWS_LOOKBACK_DAYS = 7
DEFAULT_NEWS_ACTIVATION_THRESHOLD = 0.80
DEFAULT_NEWS_MAX_ITEMS_PER_SOURCE = 25
DEFAULT_NEWS_SOURCE_COVERAGE_BUCKETS = {
    "official_macro_releases": ("official_macro", "macro_release"),
    "monetary_policy_liquidity": ("monetary_policy", "liquidity"),
    "fiscal_treasury_policy": ("fiscal_policy", "treasury", "sanctions"),
    "earnings_revisions_fundamentals": ("earnings_revisions", "earnings", "corporate_profits"),
    "regulatory_filings_enforcement": ("regulatory_filings", "enforcement", "accounting"),
    "credit_private_markets": ("credit", "private_credit", "direct_lending"),
    "energy_geopolitics": ("oil", "energy", "geopolitics", "inventories"),
    "ai_semiconductors_supply_chain": ("ai", "ai_capex", "semiconductors", "chips"),
    "market_plumbing_volatility": ("market_plumbing", "options", "crowding"),
    "crypto_liquidity_risk_appetite": ("crypto_liquidity", "bitcoin", "risk_appetite"),
    "retail_social_sentiment": ("retail_sentiment", "meme_stocks"),
}

DEFAULT_EVENT_WINDOWS = (-5, 1, 5, 21, 63)
DEFAULT_EVENT_ASSET_PROXIES = (
    "SPY",
    "QQQ",
    "XLK",
    "IGV",
    "SOXX",
    "RSP",
    "IWM",
    "SMH",
    "NVDA",
    "MSFT",
    "AVGO",
    "ORCL",
    "PLTR",
    "XLE",
    "USO",
    "DBC",
    "GLD",
    "TLT",
    "HYG",
    "LQD",
    "UUP",
    "VIXY",
    "BIL",
    "BIZD",
    "SRLN",
    "BKLN",
)
DEFAULT_EVENT_ONLY_MAX_DEFENSIVE_ADD = 0.25
DEFAULT_EVENT_CONFIRMATION_REQUIRED_SIGNALS = 2
DEFAULT_EVENT_CONFIRMATION_THEMES = ("credit", "volatility", "breadth", "trend")

# Narrative signal defaults. These research-only diagnostics turn recurring
# investor/commentary themes into visible signals that can be backtested before
# they are allowed to affect sizing directly.
DEFAULT_NARRATIVE_SIGNAL_WARNING_SCORE = 0.45
DEFAULT_NARRATIVE_SIGNAL_ACTIVE_SCORE = 0.70
DEFAULT_NARRATIVE_SIGNAL_NEWS_URGENCY_THRESHOLD = 0.70
DEFAULT_NARRATIVE_SIGNAL_RELATIVE_STRENGTH_THRESHOLD = 0.05
DEFAULT_NARRATIVE_SIGNAL_STRONG_RELATIVE_STRENGTH = 0.15
DEFAULT_NARRATIVE_SIGNAL_ABSORPTION_LOOKBACK_DAYS = 5
DEFAULT_NARRATIVE_SIGNAL_MEDIUM_LOOKBACK_DAYS = 21
DEFAULT_NARRATIVE_SIGNAL_LONG_LOOKBACK_DAYS = 63
DEFAULT_NARRATIVE_SIGNAL_DECISION_ROLE = "explainer_research_only"
DEFAULT_NARRATIVE_SIGNAL_MODEL_AUTHORITY = "no_direct_sizing_authority"
DEFAULT_NARRATIVE_SIGNAL_PROMOTION_REQUIREMENT = (
    "Must pass ablation, walk-forward, regime, churn, and paper-monitoring tests "
    "before it can affect allocation sizing."
)
DEFAULT_NARRATIVE_OPERATING_DATA_SUPPORT = ("direct", "proxy")
DEFAULT_NARRATIVE_RESEARCH_ONLY_DATA_SUPPORT = ("thin_proxy",)
DEFAULT_NARRATIVE_UNSUPPORTED_DATA_SUPPORT = ("unsupported_watchlist",)
DEFAULT_NARRATIVE_HYPERSCALER_TICKERS = (
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "ORCL",
    "QQQ",
    "IGV",
)
DEFAULT_NARRATIVE_AI_SUPPLIER_TICKERS = (
    "SMH",
    "SOXX",
    "MU",
    "NVDA",
    "AMD",
    "AVGO",
    "VRT",
    "ETN",
    "GEV",
)
DEFAULT_NARRATIVE_AI_INFRASTRUCTURE_TICKERS = ("VRT", "ETN", "PWR", "CEG", "GEV", "NRG")
DEFAULT_NARRATIVE_GLOBAL_CHIP_TICKERS = ("TSM", "ASML", "SMH", "SOXX", "MU")
DEFAULT_NARRATIVE_SPECULATIVE_TICKERS = ("ARKK", "SPHB", "IWM", "IBIT", "VIXY")
DEFAULT_NARRATIVE_DEFENSIVE_CONFIRMATION_TICKERS = ("HYG", "LQD", "VIXY", "UUP", "TLT")

# Driver-rotation dashboard defaults. This view separates proven historical
# relevance from what is currently firing so unproven narrative context remains
# visible without silently becoming an allocation input.
DEFAULT_DRIVER_ROTATION_ML_FAMILY_IMPORTANCE_PATH = (
    DEFAULT_ML_DIAGNOSTICS_DIR / "family_importance.csv"
)
DEFAULT_DRIVER_ROTATION_ACTIVE_THRESHOLD = 0.45
DEFAULT_DRIVER_ROTATION_PROVEN_THRESHOLD = 0.45
DEFAULT_DRIVER_ROTATION_EMERGING_DELTA_THRESHOLD = 0.20
DEFAULT_DRIVER_ROTATION_FADING_DELTA_THRESHOLD = -0.20
DEFAULT_DRIVER_ROTATION_SHORT_LOOKBACK_DAYS = 30
DEFAULT_DRIVER_ROTATION_LONG_LOOKBACK_DAYS = 90
DEFAULT_DRIVER_ROTATION_FALLBACK_RELEVANCE = {
    "credit": 0.72,
    "volatility": 0.72,
    "breadth": 0.68,
    "trend": 0.68,
    "ai_leadership": 0.60,
    "commodities": 0.58,
    "duration_rates": 0.55,
    "dollar_liquidity": 0.52,
    "drawdown": 0.50,
    "concentration": 0.45,
    "positioning": 0.38,
    "regime_instability": 0.35,
    "private_credit": 0.32,
    "ai_capex": 0.30,
    "equity_supply": 0.22,
}
DEFAULT_DRIVER_ROTATION_PRICE_PROXY_SPECS = (
    ("trend", "Broad equity trend", "SPY", None, 63, 0.12),
    ("ai_leadership", "AI / semis leadership", "SMH", "SPY", 63, 0.15),
    ("ai_leadership", "Nasdaq leadership", "QQQ", "RSP", 63, 0.12),
    ("breadth", "Equal weight breadth", "RSP", "SPY", 63, 0.08),
    ("breadth", "Small-cap breadth", "IWM", "SPY", 63, 0.10),
    ("credit", "Credit appetite", "HYG", "LQD", 63, 0.06),
    ("volatility", "Volatility pressure", "VIXY", None, 21, 0.25),
    ("dollar_liquidity", "Dollar pressure", "UUP", None, 63, 0.08),
    ("duration_rates", "Duration pressure", "TLT", "IEF", 63, 0.10),
    ("commodities", "Broad commodities", "DBC", "SPY", 63, 0.12),
    ("commodities", "Oil pressure", "USO", "SPY", 63, 0.18),
    ("concentration", "Concentration pressure", "QQQ", "RSP", 63, 0.12),
    ("positioning", "Speculative risk appetite", "ARKK", "SPY", 63, 0.18),
)
DEFAULT_DRIVER_ROTATION_CONFIRMATION_THEME_MAP = {
    "ai_beta": "ai_leadership",
    "broad_market": "trend",
    "concentration": "concentration",
    "credit": "credit",
    "defensive": "duration_rates",
    "growth_inflation": "commodities",
    "liquidity": "dollar_liquidity",
    "market_risk": "trend",
    "style_rotation": "breadth",
    "volatility": "volatility",
}
DEFAULT_DRIVER_ROTATION_MACRO_CATEGORY_MAP = {
    "commodities": "commodities",
    "consumer": "trend",
    "consumer_credit": "credit",
    "corporate_yields": "credit",
    "credit_spreads": "credit",
    "dollar_fx": "dollar_liquidity",
    "financial_conditions": "dollar_liquidity",
    "inflation_realized": "commodities",
    "liquidity": "dollar_liquidity",
    "monetary_policy": "duration_rates",
    "rates": "duration_rates",
    "sentiment": "positioning",
    "wages": "commodities",
}
DEFAULT_DRIVER_ROTATION_NARRATIVE_SIGNAL_MAP = {
    "ai_capex_inflation_pass_through": "ai_capex",
    "ai_supplier_hyperscaler_divergence": "ai_leadership",
    "concentration_vs_broadening": "concentration",
    "easy_bubble_vs_hard_risk_off": "regime_instability",
    "hyperscaler_capex_fcf_pressure": "ai_capex",
    "international_chip_concentration": "ai_leadership",
    "ipo_equity_supply_pressure": "equity_supply",
    "oil_inflation_shock": "commodities",
    "paid_or_unavailable_data_watchlist": "unsupported_watchlist",
    "policy_put_uncertainty": "duration_rates",
    "positive_catalyst_absorption": "ai_leadership",
    "private_credit_liquidity": "private_credit",
    "sector_valuation_policy_proxy": "positioning",
    "speculative_leverage_proxy": "positioning",
}
DEFAULT_DRIVER_ROTATION_NEWS_CATEGORY_MAP = {
    "ai_infrastructure": "ai_capex",
    "ai_unit_economics": "ai_capex",
    "earnings_revision": "trend",
    "energy_supply": "commodities",
    "fiscal_policy": "duration_rates",
    "macro_release": "duration_rates",
    "market_plumbing": "volatility",
    "monetary_policy": "duration_rates",
    "oil": "commodities",
    "private_credit": "private_credit",
    "retail_sentiment": "positioning",
}

# Signal-family evidence defaults. These support marginal-contribution tests:
# does a signal family improve growth, drawdown, re-entry, or churn after the
# strategy's normal execution-cost assumptions?
DEFAULT_SIGNAL_EVIDENCE_MIN_PAIRED_TESTS = 3
DEFAULT_SIGNAL_EVIDENCE_PROMISING_SCORE = 0.55
DEFAULT_SIGNAL_EVIDENCE_PROVEN_SCORE = 0.65
DEFAULT_SIGNAL_EVIDENCE_SIGNAL_FAMILY_KEYWORDS = {
    "reentry_timing": (
        "reentry",
        "re-entry",
        "dip",
        "buy the dip",
        "washout",
        "repair",
        "deescalation",
    ),
    "concentration_dispersion": (
        "concentration",
        "dispersion",
        "qqq/rsp",
        "equal weight",
        "cap weight",
    ),
    "breadth": ("breadth", "rsp", "small cap", "small-cap", "iwm", "equal_weight"),
    "earnings_revision": ("earnings", "revision", "margin", "fcf", "free cash flow", "profits"),
    "ai_value_chain": (
        "ai",
        "semis",
        "semiconductor",
        "capex",
        "hyperscaler",
        "supplier",
        "chip",
        "infrastructure",
    ),
    "credit": ("credit", "hyg", "lqd", "private_credit", "loan", "spread"),
    "volatility": ("vol", "vix", "instability", "left-tail", "left_tail", "drawdown"),
    "trend_momentum": ("momentum", "trend", "dual_momentum", "moving_average"),
    "sector_rotation": ("sector", "rotation", "cyclical", "xle", "xlk", "xlf", "xli"),
    "macro_policy": (
        "macro",
        "fed",
        "policy",
        "rates",
        "duration",
        "dollar",
        "liquidity",
        "inflation",
    ),
    "decision_sanity": ("decision_sanity", "sanity", "confirmation_cap", "event-only cap"),
    "ml_models": ("future_state_model", "strategy_drawdown_model", "sk_", "bayesian", "ml"),
}
DEFAULT_SIGNAL_EVIDENCE_DATA_STATUS = {
    "reentry_timing": "implemented_backtested",
    "concentration_dispersion": "proxy_backtested_needs_depth",
    "breadth": "proxy_backtested_needs_constituents",
    "earnings_revision": "thin_proxy_needs_better_data",
    "ai_value_chain": "proxy_backtested_needs_company_fundamentals",
    "credit": "implemented_backtested",
    "volatility": "implemented_backtested",
    "trend_momentum": "implemented_backtested",
    "sector_rotation": "implemented_backtested",
    "macro_policy": "proxy_backtested_needs_release_lags",
    "decision_sanity": "paired_ablation_available",
    "ml_models": "implemented_backtested_mixed",
}
DEFAULT_SIGNAL_EVIDENCE_METRIC_WEIGHTS = {
    "cagr_win_rate": 0.25,
    "drawdown_win_rate": 0.25,
    "reentry_win_rate": 0.18,
    "churn_win_rate": 0.14,
    "promotion_win_rate": 0.12,
    "calmar_win_rate": 0.06,
}

# Factor attribution defaults. These are transparent ETF proxy factors used to
# explain strategy behavior. They are not a proprietary risk model and should be
# read as directional decomposition: broad beta, AI/growth beta, rates, credit,
# commodities, volatility, and residual strategy behavior.
DEFAULT_FACTOR_ATTRIBUTION_MIN_OBSERVATIONS = 60
DEFAULT_FACTOR_ATTRIBUTION_RECENT_LOOKBACK_DAYS = 63
DEFAULT_FACTOR_ATTRIBUTION_BETA_DRIFT_THRESHOLD = 0.35
DEFAULT_FACTOR_ATTRIBUTION_R2_DROP_THRESHOLD = 0.20
DEFAULT_FACTOR_ATTRIBUTION_RESIDUAL_VOL_RATIO_THRESHOLD = 1.50
DEFAULT_FACTOR_ATTRIBUTION_FACTOR_SPECS = (
    ("market_beta", "SPY", "Market beta", "Broad U.S. equity beta."),
    ("qqq_growth_beta", "QQQ", "QQQ / growth beta", "Nasdaq and mega-cap growth exposure."),
    ("ai_semis_beta", "SMH", "AI / semis beta", "Semiconductor and AI-capex sensitivity."),
    ("equal_weight_breadth_beta", "RSP", "Breadth beta", "Equal-weight U.S. equity exposure."),
    ("sector_rotation_beta", "XLI", "Sector / cyclicals beta", "Industrial cyclicality proxy."),
    ("rates_duration_beta", "TLT", "Rates / duration beta", "Long-duration Treasury exposure."),
    ("credit_beta", "HYG", "Credit beta", "High-yield credit risk appetite."),
    ("commodity_beta", "DBC", "Commodity beta", "Broad commodity and inflation pressure."),
    ("volatility_beta", "VIXY", "Volatility beta", "Equity-volatility stress exposure."),
)


# Scenario horizons used by the current-state and dashboard scenario layers.
DEFAULT_SCENARIO_HORIZONS = ("1w", "1m", "3m", "6m")
DEFAULT_SCENARIO_EXPLANATION_TOP_SCENARIOS = 8
DEFAULT_SCENARIO_HORIZON_AUDIT_TOP_N = 10
DEFAULT_SCENARIO_HORIZON_FLAT_SPREAD_THRESHOLD = 0.03
DEFAULT_SCENARIO_HORIZON_MODEST_SPREAD_THRESHOLD = 0.08


# Dashboard performance-window defaults.
DEFAULT_PERFORMANCE_WINDOWS = (
    "30 days",
    "90 days",
    "6 months",
    "1 year",
    "3 years",
    "5 years",
    "YTD",
    "Full history",
    "Custom",
)
DEFAULT_PERFORMANCE_WINDOW = "90 days"
DEFAULT_EXPERIMENT_CACHE_TTL_SECONDS = 300


# Portfolio risk-engine defaults.
DEFAULT_RISK_DEFENSIVE_TICKER = "BIL"
DEFAULT_RISK_FACTOR_LOOKBACK_DAYS = 126
DEFAULT_RISK_COVARIANCE_LOOKBACK_DAYS = 126
DEFAULT_RISK_CORRELATION_SHORT_LOOKBACK_DAYS = 63
DEFAULT_RISK_CORRELATION_LONG_LOOKBACK_DAYS = 252
DEFAULT_RISK_TAIL_LOOKBACK_DAYS = 756
DEFAULT_RISK_EXPECTED_SHORTFALL_LEVELS = (0.95, 0.99)
DEFAULT_RISK_MAX_SINGLE_ASSET_WEIGHT = 0.55
DEFAULT_RISK_MAX_CONCENTRATION_HHI = 0.42
DEFAULT_RISK_BASE_MAX_EQUITY_BETA = 1.05
DEFAULT_RISK_BASE_MAX_AI_BETA = 0.85
DEFAULT_RISK_BASE_MAX_EXPECTED_SHORTFALL_95 = 0.035
DEFAULT_RISK_BASE_MAX_STRESS_LOSS = 0.18
DEFAULT_RISK_BASE_MAX_SCENARIO_WEIGHTED_STRESS_LOSS = 0.08
DEFAULT_RISK_BASE_MIN_DEFENSIVE_WEIGHT = 0.0
DEFAULT_RISK_MAX_TURNOVER = 0.35
DEFAULT_RISK_CORRELATION_SHIFT_THRESHOLD = 0.15
DEFAULT_RISK_MIN_RISK_ASSET_MULTIPLIER = 0.20

DEFAULT_RISK_FACTOR_PROXIES = (
    ("market_beta", "SPY", "Broad US equity beta."),
    ("nasdaq_growth_beta", "QQQ", "Mega-cap growth and Nasdaq beta."),
    ("equal_weight_breadth_beta", "RSP", "Equal-weight breadth beta."),
    ("small_cap_beta", "IWM", "Small-cap cyclicality."),
    ("value_beta", "VTV", "Value and old-economy style beta."),
    ("growth_beta", "VUG", "Growth style beta."),
    ("quality_beta", "QUAL", "Quality factor proxy."),
    ("low_vol_beta", "USMV", "Low-volatility defensive equity proxy."),
    ("ai_semiconductor_beta", "SMH", "Semiconductor and AI capex beta."),
    ("software_beta", "IGV", "Software and long-duration growth beta."),
    ("credit_beta", "HYG", "High-yield credit risk appetite."),
    ("duration_beta", "TLT", "Long-duration Treasury beta."),
    ("gold_beta", "GLD", "Gold and safe-haven beta."),
    ("oil_beta", "USO", "Oil/geopolitical inflation beta."),
    ("commodity_beta", "DBC", "Broad commodity beta."),
    ("dollar_beta", "UUP", "Dollar and liquidity-pressure beta."),
)

DEFAULT_RISK_DEFENSIVE_TICKERS = ("BIL", "SGOV", "USFR", "SHY", "VGSH", "BSV")
DEFAULT_RISK_BROAD_EQUITY_TICKERS = (
    "SPY",
    "VOO",
    "IVV",
    "SPLG",
    "VTI",
    "VT",
    "DIA",
    "RSP",
    "MDY",
    "IWB",
    "MGC",
)
DEFAULT_RISK_HIGH_BETA_TICKERS = (
    "QQQ",
    "QQQM",
    "IWM",
    "SPHB",
    "MTUM",
    "ARKK",
    "BOTZ",
    "ROBO",
    "TAN",
    "LIT",
    "SMH",
    "SOXX",
    "IGV",
    "SKYY",
    "CLOU",
)
DEFAULT_RISK_SECTOR_TICKERS = (
    "XLK",
    "XLF",
    "XLY",
    "XLP",
    "XLE",
    "XLV",
    "XLI",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
    "KRE",
    "XOP",
    "XBI",
    "TAN",
    "BOTZ",
    "SMH",
    "SOXX",
)
DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS = (
    "USMV",
    "SPLV",
    "QUAL",
    "SCHD",
    "VIG",
    "COWZ",
    "MOAT",
    "VTV",
)
DEFAULT_STRATEGY_FAMILY_HIGH_BETA_TICKERS = (
    "SPHB",
    "ARKK",
    "IBIT",
    "FBTC",
    "XBI",
    "TAN",
    "BOTZ",
    "TSLA",
)
DEFAULT_STRATEGY_FAMILY_TBILL_TICKERS = ("BIL", "BILS", "SGOV", "SHV", "TBIL", "USFR")
DEFAULT_RISK_AI_BETA_TICKERS = (
    "QQQ",
    "QQQM",
    "SMH",
    "SOXX",
    "XLK",
    "IGV",
    "NVDA",
    "AMD",
    "MSFT",
    "GOOGL",
    "GOOG",
    "META",
    "AMZN",
    "AVGO",
    "ORCL",
    "TSM",
    "PLTR",
    "MU",
    "ASML",
    "ARM",
    "SMCI",
    "DELL",
    "ANET",
    "MRVL",
    "CRWD",
    "SNOW",
    "DDOG",
    "NET",
    "APP",
    "TSLA",
)
DEFAULT_RISK_DURATION_TICKERS = (
    "AGG",
    "BND",
    "VGIT",
    "VGLT",
    "IEF",
    "IEI",
    "TLT",
    "EDV",
    "TIP",
    "VTIP",
    "LQD",
    "VCIT",
)
DEFAULT_RISK_CREDIT_TICKERS = (
    "HYG",
    "JNK",
    "LQD",
    "VCIT",
    "VCSH",
    "BKLN",
    "SRLN",
    "JAAA",
    "JBBB",
    "EMB",
    "MUB",
)
DEFAULT_RISK_PRIVATE_CREDIT_TICKERS = (
    "BIZD",
    "SRLN",
    "BKLN",
    "JAAA",
    "JBBB",
    "ARCC",
    "MAIN",
    "BXSL",
    "OBDC",
    "FSK",
)
DEFAULT_RISK_COMMODITY_TICKERS = (
    "GLD",
    "GLDM",
    "IAU",
    "SLV",
    "CPER",
    "USO",
    "BNO",
    "DBC",
    "DBA",
    "UNG",
)
DEFAULT_RISK_ENERGY_TICKERS = ("XLE", "XOP", "XES", "OIH", "USO", "BNO", "UNG")
DEFAULT_RISK_GOLD_TICKERS = ("GLD", "GLDM", "IAU")
DEFAULT_RISK_DOLLAR_TICKERS = ("UUP", "FXE", "FXY", "FXF")
DEFAULT_RISK_VOLATILITY_TICKERS = ("VIXY", "SVXY")
DEFAULT_RISK_INTERNATIONAL_TICKERS = (
    "EFA",
    "EEM",
    "VEA",
    "VWO",
    "VGK",
    "EWJ",
    "MCHI",
    "INDA",
    "EWU",
    "EWC",
    "EWA",
    "EWZ",
    "EWW",
)

# Operating exposure and tactical-matrix defaults. These are presentation and
# monitoring defaults, not live execution permissions. They provide a compact
# way to compare the current book, reference sleeves, and monitored strategies.
DEFAULT_GLOBAL_RISK_SLEEVES_REFERENCE_WEIGHTS = {
    "VT": 0.60,
    "USFR": 0.40,
    "GLDM": 0.0,
    "FBTC": 0.0,
}
DEFAULT_OPERATING_SLEEVE_MAX_EXPOSURES = {
    "stocks": 0.60,
    "defensive": 1.00,
    "gold": 0.30,
    "crypto": 0.10,
    "credit": 0.30,
    "other": 1.00,
}
DEFAULT_OPERATING_SLEEVE_TICKERS = {
    "defensive": ("BIL", "BILS", "SGOV", "SHV", "TBIL", "USFR", "SHY", "VGSH", "BSV"),
    "gold": DEFAULT_RISK_GOLD_TICKERS,
    "crypto": ("IBIT", "FBTC", "BITB", "ETHE", "BTC", "ETH"),
    "credit": DEFAULT_RISK_CREDIT_TICKERS + DEFAULT_RISK_PRIVATE_CREDIT_TICKERS,
    "stocks": (
        DEFAULT_RISK_BROAD_EQUITY_TICKERS
        + DEFAULT_RISK_HIGH_BETA_TICKERS
        + DEFAULT_RISK_SECTOR_TICKERS
        + DEFAULT_RISK_DEFENSIVE_FACTOR_TICKERS
        + DEFAULT_RISK_AI_BETA_TICKERS
        + DEFAULT_RISK_INTERNATIONAL_TICKERS
    ),
}
DEFAULT_BETA_ADJUSTED_DELTA_BENCHMARK = "SPY"
DEFAULT_BETA_ADJUSTED_DELTA_LOOKBACK_DAYS = 252
DEFAULT_TACTICAL_MATRIX_LOOKBACK_DAYS = 63
DEFAULT_TACTICAL_MATRIX_TREND_DAYS = 126
DEFAULT_TACTICAL_MATRIX_TICKERS = (
    "SPY",
    "QQQ",
    "MTUM",
    "QUAL",
    "SPHB",
    "IWM",
    "IWF",
    "IWD",
    "XLK",
    "XLF",
    "XLI",
    "XLE",
    "XLC",
    "XLB",
    "XLP",
    "ACWX",
    "EEM",
    "AGG",
    "BIL",
    "USFR",
    "HYG",
    "LQD",
    "BKLN",
    "BIZD",
    "GLD",
    "GLDM",
    "DBA",
    "DBC",
    "FBTC",
)


@dataclass(frozen=True)
class RiskStressTestDefinition:
    name: str
    description: str
    group_shocks: tuple[tuple[str, float], ...]
    default_shock: float


DEFAULT_RISK_STRESS_TESTS: tuple[RiskStressTestDefinition, ...] = (
    RiskStressTestDefinition(
        name="equity_crash",
        description="Fast equity crash with credit stress and defensive bid.",
        group_shocks=(
            ("defensive", 0.0),
            ("broad_equity", -0.20),
            ("high_beta", -0.28),
            ("ai_beta", -0.32),
            ("credit", -0.10),
            ("private_credit", -0.12),
            ("duration", 0.06),
            ("gold", 0.04),
            ("commodity", -0.08),
            ("dollar", 0.04),
            ("volatility", 0.25),
            ("international", -0.18),
        ),
        default_shock=-0.08,
    ),
    RiskStressTestDefinition(
        name="rates_up_shock",
        description="Inflation/rates shock that hits duration and long-duration growth.",
        group_shocks=(
            ("defensive", 0.0),
            ("broad_equity", -0.08),
            ("high_beta", -0.13),
            ("ai_beta", -0.16),
            ("credit", -0.06),
            ("private_credit", -0.08),
            ("duration", -0.18),
            ("gold", -0.07),
            ("commodity", 0.03),
            ("energy", 0.06),
            ("dollar", 0.06),
            ("international", -0.10),
        ),
        default_shock=-0.04,
    ),
    RiskStressTestDefinition(
        name="credit_event",
        description="Credit spread shock and private-credit repricing.",
        group_shocks=(
            ("defensive", 0.0),
            ("broad_equity", -0.12),
            ("high_beta", -0.18),
            ("ai_beta", -0.18),
            ("credit", -0.16),
            ("private_credit", -0.22),
            ("duration", 0.05),
            ("gold", 0.03),
            ("dollar", 0.04),
            ("international", -0.13),
        ),
        default_shock=-0.07,
    ),
    RiskStressTestDefinition(
        name="ai_capex_unwind",
        description="AI capex and unit-economics repricing led by semis/software/mega-cap growth.",
        group_shocks=(
            ("defensive", 0.0),
            ("broad_equity", -0.10),
            ("high_beta", -0.18),
            ("ai_beta", -0.35),
            ("credit", -0.05),
            ("duration", 0.02),
            ("gold", 0.03),
            ("dollar", 0.03),
            ("international", -0.08),
        ),
        default_shock=-0.04,
    ),
    RiskStressTestDefinition(
        name="oil_geopolitical_shock",
        description="Oil/geopolitical inflation shock with pressure on risk assets.",
        group_shocks=(
            ("defensive", 0.0),
            ("broad_equity", -0.07),
            ("high_beta", -0.10),
            ("ai_beta", -0.12),
            ("credit", -0.05),
            ("duration", -0.04),
            ("commodity", 0.12),
            ("energy", 0.18),
            ("gold", 0.06),
            ("dollar", 0.04),
            ("international", -0.08),
        ),
        default_shock=-0.03,
    ),
    RiskStressTestDefinition(
        name="dollar_liquidity_squeeze",
        description="Dollar/liquidity squeeze with broad de-risking.",
        group_shocks=(
            ("defensive", 0.0),
            ("broad_equity", -0.10),
            ("high_beta", -0.16),
            ("ai_beta", -0.18),
            ("credit", -0.08),
            ("private_credit", -0.10),
            ("duration", 0.02),
            ("gold", -0.03),
            ("dollar", 0.10),
            ("international", -0.14),
        ),
        default_shock=-0.06,
    ),
    RiskStressTestDefinition(
        name="risk_on_relief",
        description="Upside stress: relief rally that mainly checks under-exposure opportunity cost.",
        group_shocks=(
            ("defensive", 0.0),
            ("broad_equity", 0.08),
            ("high_beta", 0.12),
            ("ai_beta", 0.15),
            ("credit", 0.04),
            ("private_credit", 0.04),
            ("duration", -0.04),
            ("gold", -0.03),
            ("commodity", 0.02),
            ("international", 0.08),
        ),
        default_shock=0.03,
    ),
)


@dataclass(frozen=True)
class RegimeDefinition:
    name: str
    start: str
    end: str
    regime_type: str
    description: str


DEFAULT_REGIMES: tuple[RegimeDefinition, ...] = (
    RegimeDefinition(
        name="global_financial_crisis",
        start="2007-10-01",
        end="2009-03-31",
        regime_type="left_tail",
        description="Credit-led equity crash and policy response.",
    ),
    RegimeDefinition(
        name="euro_debt_us_downgrade",
        start="2011-07-01",
        end="2011-12-31",
        regime_type="transition",
        description="Sovereign-risk shock, downgrade stress, and sharp risk reversals.",
    ),
    RegimeDefinition(
        name="china_commodity_usd_squeeze",
        start="2015-07-01",
        end="2016-02-29",
        regime_type="transition",
        description="Commodity collapse, China devaluation concern, and dollar pressure.",
    ),
    RegimeDefinition(
        name="q4_2018_hike_liquidity_shock",
        start="2018-10-01",
        end="2018-12-31",
        regime_type="left_tail",
        description="Fed/liquidity shock with fast equity drawdown.",
    ),
    RegimeDefinition(
        name="covid_crash",
        start="2020-02-19",
        end="2020-03-31",
        regime_type="left_tail",
        description="Pandemic crash and volatility/liquidity event.",
    ),
    RegimeDefinition(
        name="covid_liquidity_rebound",
        start="2020-04-01",
        end="2020-12-31",
        regime_type="rebound",
        description="Policy-driven recovery and growth leadership.",
    ),
    RegimeDefinition(
        name="inflation_rates_bear",
        start="2022-01-01",
        end="2022-10-31",
        regime_type="left_tail",
        description="Inflation/rate shock with equity and duration drawdowns.",
    ),
    RegimeDefinition(
        name="ai_narrow_leadership",
        start="2023-03-01",
        end="2023-12-31",
        regime_type="narrow_upside",
        description="Mega-cap and AI-led rally with narrow participation.",
    ),
    RegimeDefinition(
        name="late_cycle_ai_capex",
        start="2024-01-01",
        end="2026-12-31",
        regime_type="current_analog",
        description="AI capex cycle, concentration risk, and late-cycle macro tension.",
    ),
)
