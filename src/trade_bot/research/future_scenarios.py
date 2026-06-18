from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_bot.DEFAULT import DEFAULT_SCENARIO_HORIZONS


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    scenario: str
    family: str
    risk_bucket: str
    severity: str
    base_score: float
    risk_tilt: float
    driver_weights: dict[str, float]
    horizon_bias: dict[str, float]
    expected_bot_posture: str
    preferred_exposure: str
    avoid_exposure: str
    confirmation: str
    invalidation: str
    off_ramp: str


def build_scenario_driver_state(
    confirmation_matrix: pd.DataFrame,
    market_health: pd.DataFrame,
    vams: pd.DataFrame,
    risk_score: float,
) -> pd.DataFrame:
    rows = [
        _driver(
            "market_trend",
            _mean_scores(
                [
                    _vams_score(vams, "SPY"),
                    _vams_score(vams, "QQQ"),
                    _vams_score(vams, "RSP"),
                    _signal_score(confirmation_matrix, "SPY Trend"),
                    _signal_score(confirmation_matrix, "QQQ Trend"),
                ]
            ),
            "Broad equity trend and index confirmation.",
            "SPY, QQQ, RSP",
        ),
        _driver(
            "breadth",
            _mean_scores(
                [
                    _signal_score(confirmation_matrix, "Equal Weight vs Cap Weight"),
                    _signal_score(confirmation_matrix, "Small Caps vs Mega Caps"),
                    _vams_score(vams, "RSP"),
                    _vams_score(vams, "IWM"),
                ]
            ),
            "Participation beyond mega-cap indexes.",
            "RSP/SPY, IWM/MGC, market-health breadth",
        ),
        _driver(
            "credit",
            _mean_scores(
                [
                    _signal_score(confirmation_matrix, "High Yield vs IG Credit"),
                    _vams_score(vams, "HYG"),
                    -0.5 * _vams_score(vams, "LQD"),
                ]
            ),
            "Risk appetite in credit markets.",
            "HYG/LQD, HYG trend, credit ETF drawdowns",
        ),
        _driver(
            "ai_leadership",
            _mean_scores(
                [
                    _signal_score(confirmation_matrix, "Semis vs Broad Market"),
                    _signal_score(confirmation_matrix, "QQQ Trend"),
                    _vams_score(vams, "SMH"),
                    _vams_score(vams, "QQQ"),
                    _vams_score(vams, "NVDA"),
                ]
            ),
            "AI and mega-cap growth leadership.",
            "SMH/SPY, QQQ/RSP, NVDA, AVGO, MSFT",
        ),
        _driver(
            "concentration_pressure",
            _concentration_pressure(confirmation_matrix, vams),
            "Narrow leadership risk from mega-cap and AI concentration.",
            "QQQ/RSP, SMH/SPY, RSP/SPY",
        ),
        _driver(
            "volatility_liquidity",
            _mean_scores(
                [
                    _signal_score(confirmation_matrix, "Volatility ETF Pressure"),
                    _signal_score(confirmation_matrix, "Dollar Pressure"),
                    -0.5 * _vams_score(vams, "UUP"),
                    -0.5 * _vams_score(vams, "VIXY"),
                ]
            ),
            "Volatility and dollar/liquidity pressure.",
            "VIXY, UUP, HYG/LQD",
        ),
        _driver(
            "energy_inflation_relief",
            _energy_inflation_relief(vams),
            "Oil, commodity, and inflation-pressure relief.",
            "USO, XLE, DBC, CPER/GLD",
        ),
        _driver(
            "defensive_pressure",
            _mean_scores(
                [
                    _vams_score(vams, "GLD"),
                    _vams_score(vams, "TLT"),
                    _vams_score(vams, "BIL"),
                    _vams_score(vams, "VIXY"),
                ]
            ),
            "Safe-haven and defensive-asset bid.",
            "GLD, TLT, BIL, VIXY",
        ),
        _driver(
            "duration_support",
            _mean_scores(
                [
                    _vams_score(vams, "TLT"),
                    _vams_score(vams, "IEF"),
                    _vams_score(vams, "VGIT"),
                    -0.5 * _vams_score(vams, "UUP"),
                ]
            ),
            "Duration bid and rate-sensitive support.",
            "TLT, IEF, VGIT, UUP",
        ),
        _driver(
            "drawdown_resilience",
            _drawdown_resilience(market_health, risk_score),
            "Distance from recent drawdowns and current market stress.",
            "SPY, QQQ, HYG, VIXY drawdowns",
        ),
        _driver(
            "style_rotation",
            _mean_scores(
                [
                    _signal_score(confirmation_matrix, "Value vs Growth"),
                    _vams_score(vams, "VTV"),
                    -0.5 * _vams_score(vams, "VUG"),
                    _vams_score(vams, "XLE"),
                    _vams_score(vams, "XLI"),
                ]
            ),
            "Value/cyclical rotation pressure versus growth leadership.",
            "VTV/VUG, XLE, XLI, XLF",
        ),
    ]
    frame = pd.DataFrame(rows)
    frame["score"] = frame["score"].clip(-1.0, 1.0)
    frame["state"] = frame["score"].map(_driver_state)
    return frame.sort_values("score")


def build_scenario_lattice(
    confirmation_matrix: pd.DataFrame,
    market_health: pd.DataFrame,
    vams: pd.DataFrame,
    risk_score: float,
    risk_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    driver_state = build_scenario_driver_state(
        confirmation_matrix,
        market_health,
        vams,
        risk_score,
    )
    driver_scores = dict(zip(driver_state["driver"], driver_state["score"], strict=False))
    rows: list[dict[str, object]] = []
    for horizon in DEFAULT_SCENARIO_HORIZONS:
        scored = []
        for template in _scenario_templates():
            score = _scenario_score(template, driver_scores, risk_score, horizon)
            scored.append((template, score))
        probabilities = _softmax([score for _, score in scored])
        for rank_seed, ((template, score), probability) in enumerate(
            zip(scored, probabilities, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "horizon": horizon,
                    "scenario_id": template.scenario_id,
                    "scenario": template.scenario,
                    "family": template.family,
                    "risk_bucket": template.risk_bucket,
                    "severity": template.severity,
                    "probability": probability,
                    "score": score,
                    "rank_seed": rank_seed,
                    "risk_status": risk_status,
                    "expected_bot_posture": template.expected_bot_posture,
                    "preferred_exposure": template.preferred_exposure,
                    "avoid_exposure": template.avoid_exposure,
                    "confirmation": template.confirmation,
                    "invalidation": template.invalidation,
                    "off_ramp": template.off_ramp,
                }
            )

    lattice = pd.DataFrame(rows)
    lattice["rank"] = lattice.groupby("horizon")["probability"].rank(
        ascending=False,
        method="first",
    )
    lattice["rank"] = lattice["rank"].astype(int)
    return (
        lattice.sort_values(["horizon", "rank"]).reset_index(drop=True),
        driver_state.reset_index(drop=True),
    )


def build_scenario_rollup(scenario_lattice: pd.DataFrame, risk_status: str) -> pd.DataFrame:
    if scenario_lattice.empty:
        return pd.DataFrame()

    one_month = scenario_lattice[scenario_lattice["horizon"] == "1m"]
    grouped = (
        one_month.groupby("risk_bucket", as_index=False)
        .agg(
            probability=("probability", "sum"),
            top_scenario=("scenario", lambda values: values.iloc[0]),
            expected_bot_posture=("expected_bot_posture", lambda values: values.iloc[0]),
            watch_items=("confirmation", lambda values: values.iloc[0]),
        )
        .sort_values("probability", ascending=False)
    )
    grouped["risk_status"] = risk_status
    return grouped


def _scenario_score(
    template: ScenarioTemplate,
    driver_scores: dict[str, float],
    risk_score: float,
    horizon: str,
) -> float:
    score = template.base_score + template.horizon_bias.get(horizon, 0.0)
    score += template.risk_tilt * (risk_score - 0.5)
    for driver, weight in template.driver_weights.items():
        score += weight * driver_scores.get(driver, 0.0)
    return float(score)


def _driver(driver: str, score: float, evidence: str, watch_proxies: str) -> dict[str, object]:
    return {
        "driver": driver,
        "score": float(max(-1.0, min(1.0, score))),
        "evidence": evidence,
        "watch_proxies": watch_proxies,
    }


def _signal_score(confirmation_matrix: pd.DataFrame, name: str) -> float:
    if confirmation_matrix.empty or "name" not in confirmation_matrix:
        return 0.0
    rows = confirmation_matrix[confirmation_matrix["name"] == name]
    if rows.empty:
        return 0.0
    return float(rows["score"].mean())


def _vams_score(vams: pd.DataFrame, ticker: str) -> float:
    if ticker not in vams.index or "vams_score" not in vams:
        return 0.0
    value = vams.loc[ticker, "vams_score"]
    if pd.isna(value):
        return 0.0
    return float(max(-1.0, min(1.0, float(value) / 1.5)))


def _mean_scores(values: list[float]) -> float:
    clean = [value for value in values if pd.notna(value)]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def _concentration_pressure(confirmation_matrix: pd.DataFrame, vams: pd.DataFrame) -> float:
    qqq_rsp = _signal_score(confirmation_matrix, "Nasdaq vs Equal Weight")
    smh_spy = _signal_score(confirmation_matrix, "Semis vs Broad Market")
    breadth = _signal_score(confirmation_matrix, "Equal Weight vs Cap Weight")
    ai = _mean_scores([qqq_rsp, smh_spy, _vams_score(vams, "QQQ"), _vams_score(vams, "SMH")])
    return float(max(-1.0, min(1.0, ai - max(0.0, breadth) * 0.75)))


def _energy_inflation_relief(vams: pd.DataFrame) -> float:
    energy_pressure = _mean_scores(
        [
            _vams_score(vams, "USO"),
            _vams_score(vams, "DBC"),
            _vams_score(vams, "XLE"),
            _vams_score(vams, "UUP"),
        ]
    )
    return float(max(-1.0, min(1.0, -energy_pressure)))


def _drawdown_resilience(market_health: pd.DataFrame, risk_score: float) -> float:
    if market_health.empty or "drawdown" not in market_health:
        return float(1.0 - risk_score * 2.0)
    focus = [ticker for ticker in ("SPY", "QQQ", "HYG", "RSP") if ticker in market_health.index]
    if not focus:
        return float(1.0 - risk_score * 2.0)
    average_drawdown = float(market_health.loc[focus, "drawdown"].mean())
    drawdown_component = 1.0 + average_drawdown / 0.20
    risk_component = 1.0 - risk_score * 2.0
    return float(max(-1.0, min(1.0, 0.65 * drawdown_component + 0.35 * risk_component)))


def _driver_state(score: float) -> str:
    if score >= 0.35:
        return "supportive"
    if score <= -0.35:
        return "adverse"
    return "mixed"


def _softmax(scores: list[float], temperature: float = 0.70) -> np.ndarray:
    values = np.array(scores, dtype=float) / temperature
    values = np.clip(values - values.max(), -50.0, 50.0)
    weights = np.exp(values)
    total = weights.sum()
    if total <= 0:
        return np.repeat(1.0 / len(scores), len(scores))
    return weights / total


def _scenario_templates() -> tuple[ScenarioTemplate, ...]:
    return (
        ScenarioTemplate(
            scenario_id="broad_risk_on",
            scenario="Broad risk-on broadening",
            family="risk_on",
            risk_bucket="risk_on",
            severity="constructive",
            base_score=0.15,
            risk_tilt=-0.45,
            driver_weights={
                "market_trend": 0.28,
                "breadth": 0.35,
                "credit": 0.22,
                "volatility_liquidity": 0.15,
                "drawdown_resilience": 0.12,
                "concentration_pressure": -0.12,
            },
            horizon_bias={"1w": -0.05, "1m": 0.05, "3m": 0.10, "6m": 0.05},
            expected_bot_posture="Maintain risk and allow rotation into breadth leaders.",
            preferred_exposure="SPY, RSP, IWM, sector momentum, selected cyclicals.",
            avoid_exposure="Over-concentrated mega-cap-only exposure if breadth confirms.",
            confirmation="RSP/SPY and IWM/MGC improve while HYG/LQD stays firm.",
            invalidation="Breadth rolls over while QQQ and SMH remain the only leaders.",
            off_ramp="Cut broadening trades if credit weakens and RSP/SPY fails for two weeks.",
        ),
        ScenarioTemplate(
            scenario_id="narrow_ai_melt_up",
            scenario="Narrow AI-led melt-up",
            family="ai_concentration",
            risk_bucket="risk_on_fragile",
            severity="fragile_upside",
            base_score=0.08,
            risk_tilt=-0.20,
            driver_weights={
                "ai_leadership": 0.45,
                "market_trend": 0.18,
                "credit": 0.12,
                "concentration_pressure": 0.30,
                "breadth": -0.12,
                "volatility_liquidity": 0.10,
            },
            horizon_bias={"1w": 0.15, "1m": 0.12, "3m": 0.00, "6m": -0.08},
            expected_bot_posture="Participate, but size below broad-risk-on because exits can be sharp.",
            preferred_exposure="QQQ, SMH, high-quality AI-beta names only when trend confirms.",
            avoid_exposure="Chasing low-quality AI beta after vertical moves.",
            confirmation="SMH/SPY and QQQ/RSP lead while VIXY and credit remain calm.",
            invalidation="QQQ/RSP breaks down or semis lose leadership against SPY.",
            off_ramp="Reduce AI beta quickly if SMH/SPY and HYG/LQD both weaken.",
        ),
        ScenarioTemplate(
            scenario_id="ai_capex_unwind",
            scenario="AI capex/concentration unwind",
            family="ai_concentration",
            risk_bucket="risk_off",
            severity="left_tail",
            base_score=-0.05,
            risk_tilt=0.35,
            driver_weights={
                "ai_leadership": -0.20,
                "concentration_pressure": 0.30,
                "breadth": -0.20,
                "credit": -0.22,
                "volatility_liquidity": -0.18,
                "drawdown_resilience": -0.20,
            },
            horizon_bias={"1w": -0.05, "1m": 0.05, "3m": 0.12, "6m": 0.18},
            expected_bot_posture="Reduce AI beta and require re-entry confirmation.",
            preferred_exposure="BIL/T-bills, defensive ETFs, possibly GLD when confirmed.",
            avoid_exposure="QQQ/SMH concentration and high-duration growth.",
            confirmation="QQQ/RSP, SMH/SPY, and HYG/LQD all deteriorate together.",
            invalidation="Semis recover leadership and breadth stops falling.",
            off_ramp="Exit marginal AI exposure if QQQ trend breaks and RSP does not offset.",
        ),
        ScenarioTemplate(
            scenario_id="factor_rotation_chop",
            scenario="Choppy factor rotation",
            family="rotation",
            risk_bucket="transition",
            severity="moderate",
            base_score=0.18,
            risk_tilt=0.05,
            driver_weights={
                "market_trend": -0.05,
                "breadth": 0.08,
                "style_rotation": 0.20,
                "volatility_liquidity": -0.08,
                "concentration_pressure": 0.12,
                "credit": 0.06,
            },
            horizon_bias={"1w": 0.12, "1m": 0.15, "3m": 0.03, "6m": -0.05},
            expected_bot_posture="Keep allocations smaller and demand persistence before rotating.",
            preferred_exposure="Sector momentum, value/quality, smaller position sizes.",
            avoid_exposure="Fast over-trading on one-day leadership flips.",
            confirmation="Leadership changes without broad credit deterioration.",
            invalidation="Breadth and credit both improve enough for broad risk-on.",
            off_ramp="Move defensive if chop becomes volatility expansion plus credit weakness.",
        ),
        ScenarioTemplate(
            scenario_id="rates_down_soft_landing",
            scenario="Rates-down soft landing",
            family="rates_liquidity",
            risk_bucket="risk_on",
            severity="constructive",
            base_score=0.05,
            risk_tilt=-0.15,
            driver_weights={
                "duration_support": 0.28,
                "credit": 0.20,
                "market_trend": 0.20,
                "energy_inflation_relief": 0.18,
                "volatility_liquidity": 0.16,
                "drawdown_resilience": 0.10,
            },
            horizon_bias={"1w": -0.03, "1m": 0.04, "3m": 0.12, "6m": 0.10},
            expected_bot_posture="Hold risk while allowing duration-sensitive growth to recover.",
            preferred_exposure="SPY, QQQ, quality growth, moderate TLT/IEF confirmation.",
            avoid_exposure="Energy/inflation winners if oil pressure is fading.",
            confirmation="TLT/IEF firms without HYG/LQD deterioration.",
            invalidation="Long duration rallies only because equities and credit are breaking.",
            off_ramp="Cut growth if rates-down becomes recessionary credit stress.",
        ),
        ScenarioTemplate(
            scenario_id="rates_up_liquidity_squeeze",
            scenario="Rates-up liquidity squeeze",
            family="rates_liquidity",
            risk_bucket="risk_off",
            severity="left_tail",
            base_score=-0.02,
            risk_tilt=0.28,
            driver_weights={
                "duration_support": -0.30,
                "volatility_liquidity": -0.25,
                "credit": -0.20,
                "market_trend": -0.15,
                "drawdown_resilience": -0.15,
                "energy_inflation_relief": -0.10,
            },
            horizon_bias={"1w": 0.03, "1m": 0.08, "3m": 0.10, "6m": 0.05},
            expected_bot_posture="Cut duration-sensitive beta and favor cash/T-bills.",
            preferred_exposure="BIL, SGOV, short-duration fixed income.",
            avoid_exposure="Long-duration growth, weak balance-sheet cyclicals.",
            confirmation="UUP/real-yield proxies rise, TLT weakens, credit spreads widen.",
            invalidation="Rates stabilize and credit regains trend.",
            off_ramp="Keep defensive posture until HYG/LQD and QQQ reclaim trend.",
        ),
        ScenarioTemplate(
            scenario_id="credit_led_risk_off",
            scenario="Credit-led risk-off",
            family="credit",
            risk_bucket="risk_off",
            severity="left_tail",
            base_score=-0.05,
            risk_tilt=0.50,
            driver_weights={
                "credit": -0.40,
                "volatility_liquidity": -0.22,
                "market_trend": -0.20,
                "breadth": -0.15,
                "drawdown_resilience": -0.22,
                "defensive_pressure": 0.14,
            },
            horizon_bias={"1w": 0.05, "1m": 0.12, "3m": 0.16, "6m": 0.10},
            expected_bot_posture="Prioritize drawdown control over upside capture.",
            preferred_exposure="BIL/SGOV, defensive ETFs, minimal equity beta.",
            avoid_exposure="High beta, small caps, cyclicals, levered balance sheets.",
            confirmation="HYG/LQD turns lower before or alongside equities.",
            invalidation="Credit recovers while volatility fades.",
            off_ramp="De-risk immediately if HYG trend breaks and VIXY turns bullish.",
        ),
        ScenarioTemplate(
            scenario_id="oil_inflation_shock",
            scenario="Oil/inflation shock",
            family="inflation_energy",
            risk_bucket="risk_off",
            severity="left_tail",
            base_score=-0.02,
            risk_tilt=0.24,
            driver_weights={
                "energy_inflation_relief": -0.38,
                "duration_support": -0.15,
                "credit": -0.16,
                "volatility_liquidity": -0.14,
                "style_rotation": 0.10,
                "market_trend": -0.08,
            },
            horizon_bias={"1w": 0.08, "1m": 0.12, "3m": 0.08, "6m": 0.02},
            expected_bot_posture="Do not add equity beta until oil pressure and credit stabilize.",
            preferred_exposure="Cash/T-bills, XLE only if trend/risk budget confirms, GLD as monitor.",
            avoid_exposure="Long-duration growth and broad cyclicals while oil spikes.",
            confirmation="USO/DBC rise with UUP/VIXY while equities lose breadth.",
            invalidation="Oil spike fades and credit/breadth recover.",
            off_ramp="Cut risk if oil retraces higher and HYG/LQD breaks lower.",
        ),
        ScenarioTemplate(
            scenario_id="disinflationary_slowdown",
            scenario="Disinflationary slowdown",
            family="growth_slowdown",
            risk_bucket="transition",
            severity="moderate",
            base_score=0.00,
            risk_tilt=0.18,
            driver_weights={
                "energy_inflation_relief": 0.20,
                "duration_support": 0.20,
                "credit": -0.18,
                "breadth": -0.12,
                "market_trend": -0.10,
                "defensive_pressure": 0.18,
            },
            horizon_bias={"1w": -0.02, "1m": 0.02, "3m": 0.10, "6m": 0.12},
            expected_bot_posture="Favor quality/defensive posture until growth confirms.",
            preferred_exposure="BIL, quality, some duration when credit is not breaking.",
            avoid_exposure="Cyclicals and small caps without breadth confirmation.",
            confirmation="Oil fades and bonds firm, but HYG/IWM lag.",
            invalidation="Credit and breadth improve alongside falling inflation pressure.",
            off_ramp="Reduce equities if slowdown shifts from soft to credit-led.",
        ),
        ScenarioTemplate(
            scenario_id="commodity_value_reflation",
            scenario="Commodity/value reflation",
            family="inflation_energy",
            risk_bucket="transition",
            severity="moderate",
            base_score=0.02,
            risk_tilt=0.00,
            driver_weights={
                "style_rotation": 0.30,
                "energy_inflation_relief": -0.20,
                "breadth": 0.18,
                "credit": 0.10,
                "market_trend": 0.08,
                "ai_leadership": -0.08,
            },
            horizon_bias={"1w": 0.02, "1m": 0.07, "3m": 0.11, "6m": 0.08},
            expected_bot_posture="Rotate selectively if breadth/credit support commodity leadership.",
            preferred_exposure="XLE, XLI, XLF, value/quality ETFs when trend confirms.",
            avoid_exposure="Overweight long-duration growth if rates and commodities rise.",
            confirmation="Value/cyclicals lead without credit stress.",
            invalidation="Commodity strength becomes inflation shock with weak credit.",
            off_ramp="Exit reflation trades if oil rises but breadth and credit deteriorate.",
        ),
        ScenarioTemplate(
            scenario_id="policy_whipsaw",
            scenario="Policy/headline whipsaw",
            family="policy_event",
            risk_bucket="transition",
            severity="moderate",
            base_score=0.10,
            risk_tilt=0.18,
            driver_weights={
                "volatility_liquidity": -0.22,
                "market_trend": -0.05,
                "breadth": -0.08,
                "concentration_pressure": 0.16,
                "drawdown_resilience": -0.10,
                "defensive_pressure": 0.12,
            },
            horizon_bias={"1w": 0.18, "1m": 0.12, "3m": -0.02, "6m": -0.08},
            expected_bot_posture="Lower trade size and require multi-day confirmation.",
            preferred_exposure="Existing winners only after confirmation; otherwise BIL/cash.",
            avoid_exposure="Large one-day headline trades without breadth/credit follow-through.",
            confirmation="Large reversals, VIXY pressure, and leadership flips across sessions.",
            invalidation="Volatility fades and the same leaders persist for several weeks.",
            off_ramp="Do not add after a headline spike unless credit and breadth confirm.",
        ),
        ScenarioTemplate(
            scenario_id="defensive_grind",
            scenario="Defensive grind higher",
            family="defensive",
            risk_bucket="transition",
            severity="moderate",
            base_score=0.02,
            risk_tilt=0.12,
            driver_weights={
                "defensive_pressure": 0.24,
                "duration_support": 0.18,
                "credit": 0.04,
                "market_trend": 0.02,
                "breadth": -0.05,
                "volatility_liquidity": 0.04,
            },
            horizon_bias={"1w": 0.02, "1m": 0.05, "3m": 0.06, "6m": 0.02},
            expected_bot_posture="Stay invested only through lower-beta or defensive exposures.",
            preferred_exposure="Low-volatility, staples/utilities, GLD/TLT when confirmed.",
            avoid_exposure="High-beta and weak breadth trades.",
            confirmation="Defensives lead while volatility stays contained.",
            invalidation="High beta and breadth recover decisively.",
            off_ramp="Move to cash if defensive grind turns into credit-led risk-off.",
        ),
        ScenarioTemplate(
            scenario_id="crash_policy_put",
            scenario="Fast drawdown then policy-put rebound",
            family="reflexive_policy",
            risk_bucket="risk_off_then_relief",
            severity="left_tail_then_upside",
            base_score=-0.08,
            risk_tilt=0.35,
            driver_weights={
                "drawdown_resilience": -0.28,
                "volatility_liquidity": -0.24,
                "credit": -0.18,
                "defensive_pressure": 0.18,
                "duration_support": 0.06,
                "market_trend": -0.08,
            },
            horizon_bias={"1w": 0.00, "1m": 0.08, "3m": 0.09, "6m": 0.03},
            expected_bot_posture="Survive first; re-risk only after reversal confirmation.",
            preferred_exposure="BIL during drawdown; staged re-entry after breadth/credit confirmation.",
            avoid_exposure="Trying to bottom-tick the first shock.",
            confirmation="Volatility spike and drawdown followed by credit stabilization.",
            invalidation="No policy/liquidity response and credit continues deteriorating.",
            off_ramp="Keep position size capped until VIXY falls and HYG/LQD turns up.",
        ),
    )
