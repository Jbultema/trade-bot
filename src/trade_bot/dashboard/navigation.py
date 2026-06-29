from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import streamlit as st

from trade_bot.DEFAULTS import DEFAULT_DASHBOARD_SECTIONS


@dataclass(frozen=True)
class DashboardSectionGuide:
    name: str
    role: str
    primary_question: str
    use_when: str
    first_read: str
    next_step: str
    tone: str = "neutral"


_SECTION_GUIDES: dict[str, DashboardSectionGuide] = {
    "Command Center": DashboardSectionGuide(
        name="Command Center",
        role="Current target posture and action table",
        primary_question="What is the system asking me to do now?",
        use_when="Start here after the daily brief if you need ticker-level target weights.",
        first_read="Trade decision, target weights, and material deltas.",
        next_step="Move to Forward Test only after the recommendation is worth paper/live execution.",
        tone="critical",
    ),
    "Risk & Scenarios": DashboardSectionGuide(
        name="Risk & Scenarios",
        role="Sizing guardrails, stress, and scenario bridge",
        primary_question="Why is exposure being capped, expanded, or left alone?",
        use_when="Use this when the action seems too defensive, too aggressive, or scenario-driven.",
        first_read="Risk constraints, expected shortfall, stress loss, scenario probabilities.",
        next_step="Use the constraints to decide whether the trade plan is a signal or a risk-control action.",
        tone="warning",
    ),
    "Research Lab": DashboardSectionGuide(
        name="Research Lab",
        role="Strategy evidence and experiment drilldown",
        primary_question="Which approaches are worth trusting, monitoring, or pruning?",
        use_when="Use this before promoting a strategy into champion/challenger monitoring.",
        first_read="Outcome Frontier, Candidate Details, Signal Evidence, Factor Attribution.",
        next_step="Promote only candidates with acceptable growth utility, robustness, and operability.",
        tone="success",
    ),
    "Monitoring": DashboardSectionGuide(
        name="Monitoring",
        role="Champion/challenger forward paper evidence",
        primary_question="Are monitored strategies behaving like their backtests?",
        use_when="Use this after daily valuation to compare paper windows against references.",
        first_read="Champion/challenger table, forward status, warehouse health.",
        next_step="Keep collecting evidence, demote stale windows, and review implementation drift.",
        tone="success",
    ),
    "News & Macro": DashboardSectionGuide(
        name="News & Macro",
        role="Context, narrative, and macro source review",
        primary_question="What is active context versus a proven trade driver?",
        use_when="Use this when external commentary or news seems important but may not be validated.",
        first_read="Driver Rotation, event triage, macro pressure, narrative diagnostics.",
        next_step="Treat unproven items as watch context until signal evidence supports trading impact.",
        tone="warning",
    ),
    "Performance": DashboardSectionGuide(
        name="Performance",
        role="Backtest windows and recent behavior",
        primary_question="Did the current system work in the selected historical window?",
        use_when="Use this to rebalance the evidence around custom dates or recent regimes.",
        first_read="Windowed growth, drawdown, custom start/end, benchmark comparison.",
        next_step="If recent behavior diverges from long-run behavior, inspect Research Lab and Monitoring.",
        tone="neutral",
    ),
    "Forward Test": DashboardSectionGuide(
        name="Forward Test",
        role="Recommendation locking and execution journal",
        primary_question="What was recommended, what did I do, and at what price?",
        use_when="Use this only when you decide to paper-trade or live-trade a recommendation.",
        first_read="Locked tickets, execution log, current paper/live book alignment.",
        next_step="Record exact executions so monitoring and taxable lots can reconcile later.",
        tone="critical",
    ),
}


def dashboard_section_names() -> tuple[str, ...]:
    return tuple(name for name in DEFAULT_DASHBOARD_SECTIONS if name in _SECTION_GUIDES)


def section_guide(section: str) -> DashboardSectionGuide:
    return _SECTION_GUIDES.get(section, _SECTION_GUIDES["Command Center"])


def render_dashboard_workbench_selector() -> str:
    st.markdown(
        """
        <div class="dashboard-section-header">
            <p class="dashboard-section-kicker">Dashboard Drilldown</p>
            <div class="dashboard-primary-nav-label">Insight Workbench</div>
            <p class="dashboard-nav-caption">
                Pick one focused section. The card below explains what that section answers before
                the dense tables and charts render.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_section = st.pills(
        "Dashboard section",
        dashboard_section_names(),
        selection_mode="single",
        default="Command Center",
        label_visibility="collapsed",
        key="dashboard_section",
        width="stretch",
    )
    return selected_section or "Command Center"


def render_selected_section_guide(selected_section: str) -> None:
    guide = section_guide(selected_section)
    st.markdown(
        _section_guide_html(guide),
        unsafe_allow_html=True,
    )
    with st.expander("Section map", expanded=False):
        st.caption(
            "Use this map to route questions to the right workbench without scanning the whole app."
        )
        st.dataframe(_section_map_frame(_SECTION_GUIDES.values()), use_container_width=True)


def _section_guide_html(guide: DashboardSectionGuide) -> str:
    return f"""
    <div class="workbench-guide workbench-guide-{guide.tone}">
        <div>
            <p class="workbench-guide-kicker">Active Workbench</p>
            <h2 class="workbench-guide-title">{guide.name}</h2>
            <p class="workbench-guide-role">{guide.role}</p>
        </div>
        <div class="workbench-guide-main">
            <p class="workbench-guide-question">{guide.primary_question}</p>
            <div class="workbench-guide-grid">
                <div>
                    <span>Use when</span>
                    <p>{guide.use_when}</p>
                </div>
                <div>
                    <span>Read first</span>
                    <p>{guide.first_read}</p>
                </div>
                <div>
                    <span>Next step</span>
                    <p>{guide.next_step}</p>
                </div>
            </div>
        </div>
    </div>
    """


def _section_map_frame(guides: Iterable[DashboardSectionGuide]) -> list[dict[str, str]]:
    return [
        {
            "section": guide.name,
            "answers": guide.primary_question,
            "read_first": guide.first_read,
            "use_when": guide.use_when,
        }
        for guide in guides
    ]
