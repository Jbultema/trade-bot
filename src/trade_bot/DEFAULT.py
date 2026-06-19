from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DEFAULT_CONFIG_PATH = Path("configs/baseline.yaml")
DEFAULT_EVENTS_PATH = Path("configs/events.yaml")
DEFAULT_MACRO_PATH = Path("configs/macro_fred.yaml")
DEFAULT_NEWS_PATH = Path("configs/news_sources.yaml")
DEFAULT_REPORT_PATH = Path("reports/baseline_report.html")
DEFAULT_EXPERIMENTS_DIR = Path("reports/experiments")
DEFAULT_RESET_EXPERIMENTS_DIR = Path("data/experiments_reset_v2")
DEFAULT_JOURNAL_PATH = Path("data/trading_journal.sqlite")
DEFAULT_RUN_STORE_DB_PATH = Path("data/run_store/trade_bot.duckdb")
DEFAULT_RUN_STORE_ARTIFACT_DIR = Path("data/run_store/snapshots")
DEFAULT_RUN_STORE_JOB_LOG_DIR = Path("data/run_store/jobs")
DEFAULT_SNAPSHOT_CACHE_TTL_SECONDS = 15
DEFAULT_MONITORING_TOP_N = 5
DEFAULT_EXPERIMENT_REGISTRY_LIMIT = 500

DEFAULT_TICKET_PRICE_BAND_PCT = 0.0075
DEFAULT_TICKET_SIZE_BAND_PCT = 0.20
DEFAULT_TICKET_MIN_TRADE_NOTIONAL = 25.0
DEFAULT_TICKET_WHOLE_SHARES = True
DEFAULT_FORWARD_TEST_ACCOUNT = "default_paper_account"
DEFAULT_FORWARD_TEST_STRATEGY = "scenario_adjusted_trade_decision"
DEFAULT_BOOK_ALIGNMENT_MIN_TRADE_WEIGHT = 0.02

DEFAULT_DATA_CACHE_DIR = "data/cache"
DEFAULT_DATA_ADJUSTED = True

DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_REBALANCE = "W-FRI"
DEFAULT_SIGNAL_LAG_DAYS = 1

DEFAULT_ROLLING_WINDOW_YEARS = (1, 3, 5)
DEFAULT_ROLLING_STEP_MONTHS = 1
DEFAULT_WINDOW_MIN_OBSERVATION_RATIO = 0.80
DEFAULT_CALENDAR_YEAR_MIN_OBSERVATIONS = 60
DEFAULT_REGIME_MIN_OBSERVATIONS = 20
DEFAULT_WALK_FORWARD_TRAIN_YEARS = 5
DEFAULT_WALK_FORWARD_TEST_YEARS = 1
DEFAULT_WALK_FORWARD_STEP_MONTHS = 6

DEFAULT_VOL_TARGET_ANNUALIZED_VOLATILITY = 0.12
DEFAULT_VOL_TARGET_LOOKBACK_DAYS = 63
DEFAULT_VOL_TARGET_MAX_LEVERAGE = 1.0

DEFAULT_DRAWDOWN_EQUITY_LOOKBACK_DAYS = 252
DEFAULT_DRAWDOWN_MAX_DRAWDOWN = -0.10
DEFAULT_DRAWDOWN_RISK_MULTIPLIER = 0.50

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

DEFAULT_MACRO_SIGNAL_LOOKBACK_DAYS = 1260
DEFAULT_SIGNAL_INCLUSION_LOOKBACK_DAYS = 756
DEFAULT_SIGNAL_INCLUSION_DEFENSIVE_TICKER = "BIL"
DEFAULT_SIGNAL_INCLUSION_PUBLICATION_LAG_DAYS = 21
DEFAULT_SIGNAL_INCLUSION_MIN_OBSERVATIONS = 252
DEFAULT_SIGNAL_INCLUSION_PRESSURE_THRESHOLD = 0.65
DEFAULT_SIGNAL_INCLUSION_RISK_MULTIPLIER = 0.50

DEFAULT_SCENARIO_STRESS_MULTIPLIER = 0.35
DEFAULT_SCENARIO_TRANSITION_MULTIPLIER = 0.65
DEFAULT_SCENARIO_FRAGILE_UPSIDE_MULTIPLIER = 0.80
DEFAULT_SCENARIO_RISK_ON_MULTIPLIER = 1.0
DEFAULT_SCENARIO_MIN_MULTIPLIER = 0.20
DEFAULT_SCENARIO_MAX_MULTIPLIER = 1.0
DEFAULT_SCENARIO_SIZING_LOOKBACK_DAYS = 63

DEFAULT_NEWS_CACHE_FILE = "news_items.json"
DEFAULT_NEWS_SOURCE_TYPE = "rss"
DEFAULT_NEWS_SOURCE_PRIORITY = 3
DEFAULT_NEWS_SOURCE_ENABLED = True
DEFAULT_NEWS_MAX_AGE_MINUTES = 120
DEFAULT_NEWS_LOOKBACK_DAYS = 7
DEFAULT_NEWS_ACTIVATION_THRESHOLD = 0.80
DEFAULT_NEWS_MAX_ITEMS_PER_SOURCE = 25

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

DEFAULT_SCENARIO_HORIZONS = ("1w", "1m", "3m", "6m")

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
DEFAULT_RISK_COMMODITY_TICKERS = ("GLD", "IAU", "SLV", "CPER", "USO", "BNO", "DBC", "DBA", "UNG")
DEFAULT_RISK_ENERGY_TICKERS = ("XLE", "XOP", "XES", "OIH", "USO", "BNO", "UNG")
DEFAULT_RISK_GOLD_TICKERS = ("GLD", "IAU")
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
