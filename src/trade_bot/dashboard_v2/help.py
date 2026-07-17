from __future__ import annotations

import re

from trade_bot.dashboard.metric_explainers import metric_help as _legacy_metric_help

_GENERIC_METRIC_HELP = (
    "Directional read for this view. Compare it with neighboring metrics, the chart below, "
    "and the relevant drilldown before treating it as actionable."
)
_GENERIC_SECTION_HELP = (
    "This section groups related diagnostics. Start with the headline cards, then use the "
    "tables and charts to inspect the evidence behind the read."
)
_GENERIC_CHART_HELP = (
    "Hover points for exact values. Use nearby controls for candidate, horizon, cohort, or "
    "date-range changes. Look for direction, persistence, and breaks rather than one noisy point."
)


_METRIC_HELP: dict[str, str] = {
    "market date": "The market date represented by the loaded snapshot. Use it to confirm you are not reading stale state.",
    "risk": "Current traffic-light risk state. GREEN is permissive, YELLOW is constrained, and RED means risk controls are forcing strong defense.",
    "risk state": "Current risk label plus score. Use it as the headline posture read, then inspect scenario and portfolio-risk drivers.",
    "risk score": "Composite risk pressure from 0 to 1. Rough guide: below 0.30 is calmer, 0.30-0.60 is cautionary, above 0.60 is defensive.",
    "risk budget": "Fraction of normal risk budget available after scenario and risk-engine clamps. Near 1.0 is full budget; below 0.5 is materially defensive.",
    "risk multiplier": "Portfolio-risk sizing clamp. 1.0 means unconstrained by this engine; lower values mean tail risk, beta, or scenario stress reduced sizing.",
    "1m risk-off": "One-month probability mass assigned to risk-off scenarios. Read above 25% as cautionary and above 40% as a major defensive pressure.",
    "1m scenario mix": "Top one-month scenario branches and their probabilities. Use this to see whether the current risk read is driven by one narrow story or a broad scenario tilt.",
    "current event pressure": "News/event pressure currently affecting sizing context. Event pressure can justify caution, but should usually wait for market confirmation before driving large allocations by itself.",
    "target posture": "Final target allocation after strategy signals, scenario probabilities, and risk constraints. Compare this to the current book before creating tickets.",
    "portfolio risk engine": "Sizing guardrail layer that checks expected shortfall, stress loss, beta, AI beta, concentration, and correlation. It can reduce risk even when the strategy signal is constructive.",
    "macro inclusion": "Whether a macro category has allocation authority or is only contextual. Watch/rejected categories should inform caution without directly sizing trades.",
    "decision sanity": "Bias and guardrail check for the proposed action. Use this to catch cases where narrative pressure is stronger than tradable confirmation.",
    "posture calibration": "Historical and recent evidence for whether the current defensive or risk-on posture has usually helped. This is a sniff test, not the native strategy engine.",
    "open tickets": "Number of execution tickets still open. High counts mean the paper/live book may not match the target until tickets are resolved.",
    "active windows": "Active monitoring windows. Separate start dates are intentional so hindsight-sensitive starts can be compared.",
    "start cohorts": "Distinct monitoring start dates. Use cohorts to separate early-year what-if starts from recent live starts.",
    "valued today": "Monitoring windows with a valuation for the latest date. If this is below active windows, refresh valuation jobs before judging performance.",
    "ahead": "Monitoring rows currently ahead of their benchmark. This is a status count, not proof of long-term superiority.",
    "lagging": "Monitoring rows currently behind their benchmark. Use this to identify experiments needing review or pruning.",
    "champions": "Strategies marked as the active reference for a mode/account. There should usually be only one champion per operating sleeve.",
    "challengers": "Strategies being monitored against champions or benchmarks. Challengers help validate alternatives before real allocation changes.",
    "price series": "Number of market price columns loaded. More series improves exploration but does not mean each ticker is tradable.",
    "macro series": "Number of raw macro columns loaded for visual inspection and signal construction.",
    "macro signals": "Trade-bot-derived macro indicators available in the current snapshot.",
    "pressure groups": "Macro or event categories currently applying pressure. Use this to distinguish broad pressure from a single noisy driver.",
    "active drivers": "Drivers currently above activation thresholds. Higher counts mean macro pressure or support is broader than one isolated signal.",
    "emerging drivers": "Drivers whose current activation is rising versus recent history. Use this to spot fresh pressure or new support.",
    "fading drivers": "Drivers whose current activation has weakened versus recent history. Use this to avoid overreacting to stale narratives.",
    "news items": "Current news/event records in the snapshot triage layer. These are context inputs, not automatic allocation orders.",
    "operating context": "Cross-source narrative rows backed by direct or proxy data. These can contextualize decisions but still need model evidence before sizing authority.",
    "research-only": "Thin-proxy or unsupported narrative rows. Use them for hypotheses and watchlists, not allocation decisions.",
    "validation runs": "Persisted simulation validation runs. More runs give better evidence on whether simulation calibration is stable.",
    "metric rows": "Persisted validation metric rows across horizons and variants. More rows mean richer comparisons by horizon and engine variant.",
    "latest strategy": "Strategy represented by the latest validation run. Confirm it matches the candidate you are evaluating.",
    "latest horizons": "Horizons populated in the latest validation run. Missing horizons require rerunning validation before conclusions.",
    "coverage": "Share of realized outcomes inside the simulated interval. Good calibration is close to the target interval width, not necessarily high.",
    "median miss": "Typical absolute miss of the simulated median versus realized outcome. Lower is better; high values mean p50 is directional only.",
    "candidates": "Candidate count in the research universe. Large menus raise overfitting risk, so use robustness and PBO diagnostics.",
    "displayed": "Number of candidate rows currently shown after filters. This is a UI subset, not the full research universe.",
    "champion cagr": "Best displayed compound annual growth rate. Treat it as a starting point and check drawdown, robustness, and leakage tests.",
    "best utility": "Highest composite utility after return/drawdown preferences. Utility is a ranking aid, not a guarantee.",
    "paper-ready": "Count of candidates promoted far enough for paper monitoring. Zero means research evidence has not cleared readiness gates.",
    "validation rows": "Rows of validation or QC evidence available for the selected research view.",
    "readiness": "Workflow status for the selected candidate. Snapshot-only candidates may have metrics before full rebuildable artifacts exist.",
    "overfit": "Overfitting or PBO read when available. Higher concern means the result may be more selection artifact than durable edge.",
    "path phase": "Path-aware Cycle Tracker phase after transition and duration rules. This is the preferred cycle read when available.",
    "dominant phase": "Highest-probability cycle phase after the model scores current evidence. Treat it as weak if the probability is close to competing phases.",
    "evidence phase": "Raw evidence-only phase before path constraints. Use this to see what the signals wanted before sequence rules were applied.",
    "path probability": "Confidence assigned to the path-aware phase. Low values mean the cycle state is mixed or unstable.",
    "phase probability": "Probability assigned to a cycle phase. Below roughly 25% is usually a branch to monitor; above 50% is a stronger state read if reliability is acceptable.",
    "duration": "How long the current path phase has persisted. Duration matters because phase transitions are not equally likely at all ages.",
    "duration state": "Whether the current phase duration is short, normal, or extended versus history.",
    "transition read": "Most likely next transition under the phase model. Use it as a scenario watchlist, not a deterministic forecast.",
    "path fit rate": "Historical share where the path-aware operational phase was followed by the expected forward behavior. Higher is more trustworthy.",
    "forward fit rate": "Historical share where the selected phase had useful forward outcome alignment. Use low values as a trust warning.",
    "origins": "Historical origins included in the current validation or playback view. Small origin counts make conclusions fragile.",
    "playback fit": "How often the historical crisis playback phase read matched the labeled crisis stage. Around 50% is mixed; materially higher is more useful.",
    "phase odds": "Probability assigned to the selected phase. Low odds mean the phase is only a scenario branch, not the dominant read.",
    "top ticker": "Highest-ranked ticker for the selected scenario/phase/horizon. Confirm durability before assuming it is a live candidate.",
    "top role": "Suggested role for the top ticker, such as watch, starter reentry, or defensive. Roles are research guidance, not trade orders.",
    "risk level": "Portfolio-risk status after guardrails. Risk-reduced means the engine is clamping size because one or more constraints are active.",
    "es 95": "Expected shortfall at the 95% tail. It estimates average loss in bad tail outcomes; lower is safer.",
    "max stress loss": "Worst modeled loss across configured stress tests. Use it as a scenario-loss guardrail, not a prediction.",
    "equity beta": "Estimated sensitivity to broad equity markets. Around 1 behaves like the market; below 0.5 is more defensive.",
    "ai beta": "Estimated sensitivity to AI/growth leadership proxies. Higher values mean more exposure to the current AI/tech leadership theme.",
    "beta adjusted s&p delta": "S&P-like exposure after adjusting holdings by beta. It translates different assets into broad-market risk equivalents.",
    "defensive % of max": "How much of the allowed defensive sleeve is currently used. High values mean the strategy is leaning heavily into cash/BIL-like exposure.",
    "instability": "Regime instability state. Elevated or unstable means market internals are shifting enough to monitor, but this is watch-only unless other controls confirm.",
}

_SECTION_HELP: dict[str, str] = {
    "trade decision": "Daily operating answer: what the system wants the book to do now and which constraints are driving it.",
    "raw evidence": "Underlying evidence rows behind the decision. Use this when the headline seems surprising.",
    "trading alerts": "Strategy and portfolio alerts that may require review before execution.",
    "scenario outlook": "Current scenario probabilities feeding risk budget and future-state simulation.",
    "champion / challenger readout": "Status table for monitored strategies by start cohort. Use it to decide whether live or paper experiments are proving out.",
    "macro overview": "Fast macro landing page: driver rotation, narrative pressure, and macro history before raw tables.",
    "macro visual explorer": "Interactive raw market and macro visual surface. Use it to inspect the inputs before trusting narrative conclusions.",
    "signal drivers": "Driver-rotation and macro-signal detail. Use this to separate broad, fresh pressure from a single noisy item.",
    "news & events": "Structured news and event-risk records. Use this to inspect what current narratives are pressuring the model.",
    "macro signal tables": "Tabular macro and news signals. Use this for precise driver names, values, and categories.",
    "risk operating read": "Fast risk landing page. It combines risk guardrails, operating exposure, scenario pressure, and instability.",
    "portfolio risk detail": "Guardrail detail for constraints, scenario stress, betas, tail loss, and correlation.",
    "operating exposure": "Current target posture translated into sleeves and beta-adjusted market exposure.",
    "regime instability": "Watch-only transition-risk diagnostic. Use this to separate a fresh instability jump from persistent instability.",
    "future-state scenario lattice": "Scenario probability view by horizon. Use this to see which future branches are shaping risk budget.",
    "scenario drivers": "Input drivers pushing the scenario lattice. Use this to see why risk-off, transition, or risk-on branches moved.",
    "confirmation and health": "Market confirmation, breadth, credit, volatility, and health checks behind the current posture.",
    "vol-adjusted momentum": "Ticker momentum adjusted for recent volatility. Use bullish/neutral/bearish labels as confirmation context.",
    "top candidate summary": "Fast research landing page. Use it to find promising candidates before deep inspection.",
    "candidate deep dive": "Candidate-specific research, robustness, and operating diagnostics. This is the main belief-maintenance view.",
    "candidate detail tabs": "Detailed candidate artifacts: performance, allocation behavior, decisions, attribution, mechanics, and robustness.",
    "validation artifacts": "Simulation and research validation outputs. Use these to evaluate whether a candidate or model is trustworthy.",
    "pbo summary": "Probability-of-backtest-overfitting read. Use this to judge whether a candidate may be a selection artifact from too many trials.",
    "pbo selections": "Selected folds and ranks behind the PBO read. Inspect this when a high-CAGR candidate looks too clean.",
    "leadership summary": "Tech, AI, sector, and leadership-dependence diagnostics. Use it to see whether returns came from broad robustness or a narrow market regime.",
    "router comparison": "Walk-forward strategy-router evidence. Use this to see whether state-aware selection helped out of sample compared with fixed candidates.",
    "scenario / phase frontier": "Conditional winner map by scenario, phase, and horizon. Use it to ask what may work if a future state dominates.",
    "category summary": "Grouped macro or research-category scores. Use this to find which areas are driving the read before drilling into individual rows.",
    "signal detail": "Individual macro signal rows and values. This is where you verify the raw inputs behind the summary.",
    "current news/event triage": "Current news and event items converted into structured pressure tags. These are evidence context, not automatic trade instructions.",
    "0m nowcast phase probabilities": "Current, same-date Cycle Tracker phase probabilities. Use this as the present-state map before looking at forward horizons.",
    "current-phase conditional candidates": "Candidates historically associated with the current nowcast phase. Treat these as research hypotheses unless robustness and sample size are acceptable.",
    "path-aware cycle tracker": "Sequential phase model with memory and duration rules. Use this instead of raw phase affinity for cycle-state calls.",
    "path-aware transition model": "Cycle Tracker nowcast and next-phase probabilities after applying allowed transitions and duration rules.",
    "path reliability": "Backtest evidence for the path-aware operational read after transition, duration, memory, and drawdown constraints.",
    "phase reliability": "Raw evidence-label reliability before path-aware sequence rules. Use it to diagnose the classifier, not override the path state.",
    "historical phase reliability": "Trust check for raw phase evidence labels before path constraints. Thin samples and low fit rates should stay research-only.",
    "historical crisis playback": "Replay of Cycle Tracker probabilities through named stress windows. Use this to judge whether the model behaves sensibly in crises.",
    "scenario / phase winner frontier": "Ticker and role ranking for a selected phase and horizon. Treat odd winners as hypotheses requiring durability checks.",
    "simulation lab": "Forward-looking planning workbench. Use it for future-state paths, validation history, and strategy simulation diagnostics.",
    "validation summary": "Fast read of persisted simulation validation quality. Use it before relying on simulated ranges.",
    "per-horizon metrics": "Simulation calibration split by horizon. A model can be acceptable at 3m and weak at 1m.",
    "strategy simulations": "Candidate-specific future-state simulation surface. Use it to inspect path ranges and scenario-conditioned outcomes.",
    "full workbench": "Loads the complete slower workbench for detailed controls and detail-depth diagnostics.",
}

_CHART_HELP: dict[str, str] = {
    "risk score, risk-off odds, and trade budget": "Shows whether risk pressure is persistent or a one-day jump. Rising risk score/risk-off odds and falling budget support defensive sizing.",
    "cumulative return by monitoring window": "Compares monitored windows since their start dates. Look for sustained excess versus benchmark, not a single valuation point.",
    "selected market proxies": "Indexes selected price series to a common start. Use this to compare relative moves across tickers and asset classes.",
    "selected macro signals": "Shows selected macro signals as z-scores or levels. In z-score mode, values beyond +/-2 are unusually stretched.",
    "driver rotation: historical relevance vs current activation": "Compares which drivers usually matter with which are active now. Upper-right drivers deserve most attention.",
    "driver activation heatmap": "Shows current and recent driver activation. Use it to see whether a pressure is fresh, persistent, or fading.",
    "scenario driver scores over time": "Historical snapshot trend for scenario-driver scores. Direction and persistence matter more than one point.",
    "driver rotation activation over time": "Historical snapshot trend for driver activation. Use this to spot fresh rotations versus old pressure.",
    "sizing clamp and tail risk": "Risk multiplier and tail-loss constraints over time. Sudden drops show the risk engine clamping exposure.",
    "beta and correlation pressure": "Equity beta, AI beta, and correlation shift over time. Rising beta pressure can make the book more sensitive to growth selloffs.",
    "scenario / phase frontier": "Ranks candidate winners for a selected conditional future. Higher bars are stronger historical conditional scores, but require robustness review.",
    "phase probabilities": "Stacked probabilities across cycle phases. Dominant and persistent phase bands matter more than noisy daily wiggles.",
    "path reliability": "Shows whether path-aware operational phases historically behaved as expected. Prefer enough origins and fit clearly above mixed/noise levels.",
    "phase reliability": "Compares raw phase-label reliability by horizon. This is a classifier audit before sequence constraints.",
    "historical phase reliability": "Shows whether raw evidence phases historically behaved as expected. Use it to understand evidence quality, not as the operational cycle read.",
    "historical crisis playback": "Replays phase probabilities through known stress windows. Sensible models should lift unwind/liquidation during declines and recovery after bottoming.",
    "scenario / phase winner frontier chart": "Conditional ticker ranking. Extreme winners may reflect short histories or one regime; inspect sample size and durability.",
}


def metric_help(label: str) -> str:
    local = _METRIC_HELP.get(_normalise(label))
    if local:
        return local
    legacy = _legacy_metric_help(label)
    if legacy:
        return legacy
    return _GENERIC_METRIC_HELP


def section_help(title: str) -> str:
    return _SECTION_HELP.get(_normalise(title), _GENERIC_SECTION_HELP)


def chart_help(title: str | None) -> str:
    if not title:
        return _GENERIC_CHART_HELP
    return _CHART_HELP.get(_normalise(title), _GENERIC_CHART_HELP)


def _normalise(value: str) -> str:
    cleaned = re.sub(r"[_/]+", " ", str(value)).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
