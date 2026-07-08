from __future__ import annotations

import streamlit as st


def _install_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(90deg, rgba(15, 118, 110, 0.035) 1px, transparent 1px),
                linear-gradient(180deg, rgba(245, 158, 11, 0.040) 1px, transparent 1px),
                linear-gradient(
                    180deg,
                    color-mix(in srgb, #eef4f2 54%, var(--background-color, #ffffff)) 0%,
                    color-mix(in srgb, var(--secondary-background-color, #f8fafc) 74%, var(--background-color, #ffffff)) 320px,
                    var(--background-color, #ffffff) 760px
                );
            background-size: 56px 56px, 56px 56px, auto;
        }
        .block-container {
            max-width: 1540px;
            padding-top: 2.45rem;
            padding-bottom: 3.25rem;
        }
        h1 {
            font-size: 2.45rem;
            line-height: 1.05;
            letter-spacing: 0;
            margin-bottom: 0.18rem;
            color: var(--text-color, #111827);
        }
        h2, h3 {
            letter-spacing: 0;
        }
        h2 {
            margin-top: 1.55rem;
        }
        .brand-masthead {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 18px;
            margin: 0 0 20px;
            padding: 26px 24px 23px;
            border: 1px solid rgba(15, 118, 110, 0.35);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(21, 26, 33, 0.98) 0%, rgba(21, 26, 33, 0.94) 52%, rgba(15, 118, 110, 0.90) 100%),
                repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0 1px, transparent 1px 12px);
            box-shadow: var(--tb-shadow);
            color: #f8fafc;
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 0;
        }
        .brand-mark {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            width: 54px;
            height: 54px;
            border-radius: 8px;
            border: 1px solid rgba(167, 243, 208, 0.55);
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.92), rgba(20, 184, 166, 0.46)),
                linear-gradient(180deg, rgba(255, 255, 255, 0.12), transparent);
            overflow: hidden;
        }
        .brand-mark::before {
            content: "";
            position: absolute;
            inset: 10px;
            border-left: 2px solid rgba(248, 250, 252, 0.22);
            border-bottom: 2px solid rgba(248, 250, 252, 0.22);
        }
        .brand-mark-text {
            position: relative;
            z-index: 2;
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 860;
            line-height: 1;
        }
        .brand-mark-line {
            position: absolute;
            left: 13px;
            right: 11px;
            bottom: 17px;
            height: 16px;
            border-top: 4px solid #f59e0b;
            border-right: 4px solid #22c55e;
            transform: skewY(-28deg);
            opacity: 0.88;
        }
        .brand-copy {
            min-width: 0;
        }
        .brand-eyebrow {
            margin: 0 0 7px;
            color: #a7f3d0;
            font-size: 0.74rem;
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.2;
            text-transform: uppercase;
        }
        .brand-title {
            margin: 0;
            color: #f8fafc !important;
            font-size: 1.9rem;
            line-height: 1.05;
            font-weight: 820;
            letter-spacing: 0;
        }
        .brand-subtitle {
            max-width: 840px;
            margin: 8px 0 0;
            color: rgba(248, 250, 252, 0.86);
            font-size: 0.98rem;
            line-height: 1.42;
        }
        .brand-proof-row {
            align-self: flex-start;
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 8px;
            max-width: 390px;
            padding-top: 10px;
        }
        .brand-proof {
            display: inline-flex;
            align-items: center;
            min-height: 30px;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid rgba(248, 250, 252, 0.24);
            background: rgba(248, 250, 252, 0.10);
            color: #f8fafc;
            font-size: 0.76rem;
            font-weight: 760;
            white-space: nowrap;
        }
        .freshness-strip {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            margin: -8px 0 14px;
            padding: 8px 11px;
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-accent);
            border-radius: 8px;
            background: color-mix(in srgb, var(--tb-card-bg) 86%, transparent);
            color: var(--tb-card-text);
            box-shadow: var(--tb-shadow-soft);
        }
        .freshness-kicker {
            color: var(--tb-accent);
            font-size: 0.72rem;
            font-weight: 820;
            text-transform: uppercase;
        }
        .freshness-main {
            font-size: 0.86rem;
            font-weight: 720;
        }
        .freshness-chip {
            border: 1px solid var(--tb-card-border);
            border-radius: 999px;
            padding: 2px 8px;
            color: var(--tb-card-muted);
            font-size: 0.76rem;
            font-weight: 680;
            white-space: nowrap;
        }
        .freshness-detail {
            color: var(--tb-card-muted);
            font-size: 0.76rem;
        }
        .dashboard-section-header {
            margin: 38px 0 16px;
            padding: 26px 30px 24px;
            border: 2px solid color-mix(in srgb, var(--tb-accent) 44%, var(--tb-card-border));
            border-left: 12px solid var(--tb-accent);
            border-radius: 8px;
            background:
                linear-gradient(90deg, color-mix(in srgb, var(--tb-accent) 18%, var(--tb-card-bg)), var(--tb-card-bg) 62%),
                linear-gradient(180deg, color-mix(in srgb, var(--tb-accent) 8%, transparent), transparent);
            box-shadow: var(--tb-shadow);
        }
        .dashboard-section-kicker {
            margin: 0 0 9px;
            color: var(--tb-accent);
            font-size: 0.98rem;
            font-weight: 900;
            letter-spacing: 0;
            line-height: 1.2;
            text-transform: uppercase;
        }
        .dashboard-primary-nav-label {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0;
            color: var(--tb-card-text);
            font-size: 2.28rem;
            font-weight: 920;
            letter-spacing: 0;
            line-height: 1.05;
        }
        .dashboard-primary-nav-label::before {
            content: "";
            width: 14px;
            height: 42px;
            border-radius: 3px;
            background: linear-gradient(180deg, var(--tb-accent), #f59e0b);
            flex: 0 0 auto;
        }
        .dashboard-nav-caption {
            margin: 10px 0 0 26px;
            color: var(--tb-card-muted);
            font-size: 1.14rem;
            line-height: 1.35;
        }
        div[data-testid="stPills"] {
            margin: 0 0 0.55rem;
            padding: 10px 0 0.7rem;
            border-bottom: 0;
        }
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] {
            margin: 0 0 1.1rem;
            padding: 14px 14px 16px;
            border: 1px solid color-mix(in srgb, var(--tb-card-border) 82%, var(--tb-accent));
            border-radius: 10px;
            background:
                linear-gradient(180deg, color-mix(in srgb, var(--tb-card-bg) 92%, var(--tb-accent)), var(--tb-card-bg)),
                color-mix(in srgb, var(--tb-card-border) 12%, transparent);
        }
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] [data-baseweb="tab-list"],
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] div[role="radiogroup"] {
            gap: 14px;
        }
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] button,
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] [role="button"] {
            min-height: 66px;
            padding: 15px 26px !important;
            border: 2px solid color-mix(in srgb, var(--tb-card-border) 74%, var(--tb-card-text)) !important;
            background: color-mix(in srgb, var(--tb-card-bg) 94%, var(--tb-card-border)) !important;
            color: var(--tb-card-text) !important;
            font-size: 1.18rem !important;
            font-weight: 900 !important;
            line-height: 1.1 !important;
            box-shadow:
                0 1px 0 color-mix(in srgb, white 64%, transparent) inset,
                0 8px 18px color-mix(in srgb, var(--tb-card-border) 44%, transparent);
        }
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] button:hover,
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] [role="button"]:hover {
            border-color: color-mix(in srgb, var(--tb-accent) 54%, var(--tb-card-border)) !important;
            background: color-mix(in srgb, var(--tb-accent) 10%, var(--tb-card-bg)) !important;
            box-shadow:
                0 1px 0 color-mix(in srgb, white 68%, transparent) inset,
                0 11px 24px color-mix(in srgb, var(--tb-accent) 16%, transparent);
        }
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] button[aria-selected="true"],
        .st-key-dashboard_main_station_nav div[data-testid="stPills"] [role="button"][aria-selected="true"] {
            border: 3px solid var(--tb-danger) !important;
            background:
                linear-gradient(180deg, color-mix(in srgb, var(--tb-danger) 14%, var(--tb-card-bg)), color-mix(in srgb, var(--tb-danger) 6%, var(--tb-card-bg))) !important;
            color: var(--tb-danger) !important;
            box-shadow:
                0 0 0 4px color-mix(in srgb, var(--tb-danger) 10%, transparent),
                0 12px 28px color-mix(in srgb, var(--tb-danger) 22%, transparent);
        }
        .dashboard-workbench-divider {
            height: 1px;
            margin: 0.85rem 0 1.35rem;
            background: color-mix(in srgb, var(--tb-card-border) 78%, transparent);
        }
        .workbench-guide {
            display: grid;
            grid-template-columns: minmax(220px, 0.34fr) minmax(0, 1fr);
            gap: 18px;
            margin: 0.2rem 0 0.7rem;
            padding: 18px 20px;
            border: 1px solid var(--tb-card-border);
            border-left: 7px solid var(--tb-card-border);
            border-radius: 8px;
            background:
                linear-gradient(90deg, color-mix(in srgb, var(--tb-card-border) 16%, var(--tb-card-bg)), var(--tb-card-bg) 62%),
                var(--tb-card-bg);
            color: var(--tb-card-text);
            box-shadow: var(--tb-shadow-soft);
        }
        .workbench-guide-critical {
            border-left-color: var(--tb-critical-border);
        }
        .workbench-guide-warning {
            border-left-color: var(--tb-warning-border);
        }
        .workbench-guide-success {
            border-left-color: var(--tb-success-border);
        }
        .workbench-guide-neutral {
            border-left-color: var(--tb-accent);
        }
        .workbench-guide-kicker {
            margin: 0 0 5px;
            color: var(--tb-accent);
            font-size: 0.75rem;
            font-weight: 850;
            line-height: 1.15;
            text-transform: uppercase;
        }
        .workbench-guide-title {
            margin: 0 !important;
            color: var(--tb-card-text) !important;
            font-size: 1.5rem;
            line-height: 1.12;
        }
        .workbench-guide-role {
            margin: 7px 0 0;
            color: var(--tb-card-muted);
            font-size: 0.94rem;
            line-height: 1.35;
        }
        .workbench-guide-main {
            min-width: 0;
        }
        .workbench-guide-question {
            margin: 0 0 10px;
            color: var(--tb-card-text);
            font-size: 1.08rem;
            font-weight: 780;
            line-height: 1.35;
        }
        .workbench-guide-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .workbench-guide-grid div {
            min-height: 86px;
            padding: 10px 12px;
            border: 1px solid var(--tb-card-border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--tb-panel-bg) 76%, transparent);
        }
        .workbench-guide-grid span {
            display: block;
            margin: 0 0 5px;
            color: var(--tb-card-muted);
            font-size: 0.72rem;
            font-weight: 820;
            line-height: 1.2;
            text-transform: uppercase;
        }
        .workbench-guide-grid p {
            margin: 0;
            color: var(--tb-card-text);
            font-size: 0.89rem;
            line-height: 1.38;
        }
        .runtime-notice {
            margin: 8px 0 14px;
            padding: 12px 14px;
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
            box-shadow: var(--tb-shadow-soft);
        }
        .runtime-notice-warning {
            border-left-color: var(--tb-warning-border);
            background: var(--tb-warning-bg);
        }
        .runtime-notice-critical {
            border-left-color: var(--tb-critical-border);
            background: var(--tb-critical-bg);
        }
        .runtime-notice-success {
            border-left-color: var(--tb-success-border);
            background: var(--tb-success-bg);
        }
        .runtime-notice-neutral {
            border-left-color: var(--tb-accent);
            background: color-mix(in srgb, var(--tb-accent) 7%, var(--tb-card-bg));
        }
        .runtime-notice-kicker {
            display: block;
            margin: 0 0 4px;
            color: var(--tb-card-muted);
            font-size: 0.72rem;
            font-weight: 850;
            line-height: 1.15;
            text-transform: uppercase;
        }
        .runtime-notice strong {
            display: block;
            margin: 0 0 4px;
            font-size: 1.02rem;
            line-height: 1.25;
        }
        .runtime-notice p {
            margin: 0;
            color: var(--tb-card-text);
            font-size: 0.92rem;
            line-height: 1.42;
        }
        .metric-info-rail {
            margin: 8px 0 12px;
            padding: 16px 16px 14px;
            border: 1px solid var(--tb-card-border);
            border-left: 6px solid var(--tb-accent);
            border-radius: 8px;
            background:
                linear-gradient(180deg, color-mix(in srgb, var(--tb-accent) 9%, var(--tb-card-bg)), var(--tb-card-bg)),
                var(--tb-card-bg);
            box-shadow: var(--tb-shadow-soft);
        }
        .metric-info-kicker {
            margin: 0 0 5px;
            color: var(--tb-accent);
            font-size: 0.72rem;
            font-weight: 860;
            line-height: 1.15;
            text-transform: uppercase;
        }
        .metric-info-title {
            color: var(--tb-card-text);
            font-size: 1.2rem;
            font-weight: 840;
            line-height: 1.18;
        }
        .metric-info-copy {
            margin: 7px 0 0;
            color: var(--tb-card-muted);
            font-size: 0.85rem;
            line-height: 1.38;
        }
        .metric-info-card {
            margin: 10px 0 12px;
            padding: 14px 14px 12px;
            border: 1px solid var(--tb-card-border);
            border-radius: 8px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
            box-shadow: var(--tb-shadow-soft);
        }
        .metric-info-card-label {
            margin: 0 0 5px;
            color: var(--tb-card-muted);
            font-size: 0.70rem;
            font-weight: 820;
            line-height: 1.15;
            text-transform: uppercase;
        }
        .metric-info-card h3 {
            margin: 0 0 8px !important;
            color: var(--tb-card-text) !important;
            font-size: 1.05rem;
            line-height: 1.22;
        }
        .metric-info-card p {
            margin: 0 0 10px;
            color: var(--tb-card-text);
            font-size: 0.86rem;
            line-height: 1.42;
        }
        .st-key-quick_reference_rail {
            position: fixed;
            top: 4.15rem;
            right: 1.05rem;
            z-index: 999;
            width: 330px;
            max-height: calc(100vh - 5.15rem);
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0 0.18rem 0.8rem;
            scrollbar-width: thin;
        }
        .st-key-quick_reference_rail div[data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }
        .metric-info-card-section {
            margin-top: 10px;
            padding-top: 9px;
            border-top: 1px solid var(--tb-card-border);
        }
        .metric-info-card-section span {
            display: block;
            margin: 0 0 4px;
            color: var(--tb-card-muted);
            font-size: 0.70rem;
            font-weight: 820;
            line-height: 1.15;
            text-transform: uppercase;
        }
        .metric-info-card-section p {
            margin: 0;
        }
        .simulation-validation-verdict {
            margin: 12px 0 16px;
            padding: 18px 20px 16px;
            border: 1px solid var(--tb-card-border);
            border-left: 8px solid var(--tb-card-border);
            border-radius: 8px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
            box-shadow: var(--tb-shadow-soft);
        }
        .simulation-validation-verdict h4 {
            margin: 3px 0 8px !important;
            color: var(--tb-card-text) !important;
            font-size: 1.34rem;
            line-height: 1.18;
            letter-spacing: 0;
        }
        .simulation-validation-verdict p {
            max-width: 1120px;
            margin: 0 0 11px;
            color: var(--tb-card-text);
            font-size: 0.95rem;
            line-height: 1.42;
        }
        .simulation-validation-verdict .simulation-validation-detail {
            color: var(--tb-card-muted);
        }
        .simulation-validation-verdict-kicker,
        .simulation-validation-card-label {
            color: var(--tb-card-muted);
            font-size: 0.74rem;
            font-weight: 840;
            line-height: 1.15;
            text-transform: uppercase;
        }
        .simulation-validation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 14px;
            margin: 0 0 18px;
        }
        .simulation-validation-card {
            min-height: 156px;
            padding: 16px 16px 14px;
            border: 1px solid var(--tb-card-border);
            border-left: 6px solid var(--tb-card-border);
            border-radius: 8px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
            box-shadow: var(--tb-shadow-soft);
        }
        .simulation-validation-card strong {
            display: block;
            margin: 8px 0 7px;
            color: var(--tb-card-text);
            font-size: 1.38rem;
            font-weight: 780;
            line-height: 1.12;
        }
        .simulation-validation-card p {
            min-height: 42px;
            margin: 0 0 10px;
            color: var(--tb-card-muted);
            font-size: 0.82rem;
            line-height: 1.32;
        }
        .simulation-validation-pill {
            display: inline-flex;
            align-items: center;
            min-height: 24px;
            padding: 3px 9px;
            border: 1px solid var(--tb-card-border);
            border-radius: 999px;
            color: var(--tb-card-text);
            background: color-mix(in srgb, var(--tb-card-bg) 82%, transparent);
            font-size: 0.72rem;
            font-weight: 780;
            line-height: 1.1;
        }
        .simulation-validation-good {
            border-left-color: var(--tb-success-border);
            background: var(--tb-success-bg);
        }
        .simulation-validation-warn {
            border-left-color: var(--tb-warning-border);
            background: var(--tb-warning-bg);
        }
        .simulation-validation-bad {
            border-left-color: var(--tb-critical-border);
            background: var(--tb-critical-bg);
        }
        .simulation-validation-neutral {
            border-left-color: var(--tb-card-border);
            background: var(--tb-card-bg);
        }
        div[data-testid="stPills"] button,
        div[data-testid="stPills"] [role="button"] {
            min-height: 46px;
            border-radius: 8px !important;
            border: 1px solid var(--tb-card-border) !important;
            background: var(--tb-card-bg) !important;
            color: var(--tb-card-text) !important;
            font-size: 1.03rem !important;
            font-weight: 760 !important;
            padding: 8px 13px !important;
            box-shadow: var(--tb-shadow-soft);
        }
        div[data-testid="stPills"] button:hover,
        div[data-testid="stPills"] [role="button"]:hover {
            border-color: color-mix(in srgb, var(--tb-accent) 58%, var(--tb-card-border)) !important;
            background: color-mix(in srgb, var(--tb-accent) 8%, var(--tb-card-bg)) !important;
        }
        div[data-testid="stPills"] button[aria-selected="true"],
        div[data-testid="stPills"] [role="button"][aria-selected="true"] {
            border-color: var(--tb-accent) !important;
            background: color-mix(in srgb, var(--tb-accent) 16%, var(--tb-card-bg)) !important;
            color: var(--tb-card-text) !important;
        }
        :root {
            --tb-accent: #0f766e;
            --tb-accent-strong: #115e59;
            --tb-accent-blue: #2563eb;
            --tb-card-bg: var(--secondary-background-color, #ffffff);
            --tb-card-border: rgba(125, 139, 155, 0.35);
            --tb-card-text: var(--text-color, #111827);
            --tb-card-muted: #4b5563;
            --tb-card-muted: color-mix(in srgb, var(--text-color, #111827) 68%, transparent);
            --tb-panel-bg: color-mix(in srgb, var(--secondary-background-color, #ffffff) 90%, var(--background-color, #ffffff));
            --tb-critical-bg: #fff5f5;
            --tb-critical-bg: color-mix(in srgb, #ef4444 12%, var(--background-color, #ffffff));
            --tb-critical-border: #c53030;
            --tb-warning-bg: #fffaf0;
            --tb-warning-bg: color-mix(in srgb, #f59e0b 12%, var(--background-color, #ffffff));
            --tb-warning-border: #b7791f;
            --tb-success-bg: #f7fbf8;
            --tb-success-bg: color-mix(in srgb, #22c55e 10%, var(--background-color, #ffffff));
            --tb-success-border: #2f855a;
            --tb-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            --tb-shadow-soft: 0 6px 18px rgba(15, 23, 42, 0.06);
            --tb-sidebar-bg: #f8fafc;
            --tb-sidebar-panel: #ffffff;
            --tb-sidebar-text: #111827;
            --tb-sidebar-muted: #64748b;
        }
        @media (prefers-color-scheme: dark) {
            .stApp {
                background:
                    linear-gradient(90deg, rgba(20, 184, 166, 0.035) 1px, transparent 1px),
                    linear-gradient(180deg, rgba(245, 158, 11, 0.030) 1px, transparent 1px),
                    linear-gradient(180deg, #101318 0%, #111827 420px, #0f1115 100%);
                background-size: 56px 56px, 56px 56px, auto;
            }
            :root {
                --tb-card-bg: #171b22;
                --tb-card-border: #2d3440;
                --tb-card-text: #f8fafc;
                --tb-card-muted: #cbd5e1;
                --tb-panel-bg: #141922;
                --tb-critical-bg: #2a1518;
                --tb-critical-border: #ef4444;
                --tb-warning-bg: #251d10;
                --tb-warning-border: #f59e0b;
                --tb-success-bg: #102018;
                --tb-success-border: #22c55e;
                --tb-shadow: 0 14px 30px rgba(0, 0, 0, 0.30);
                --tb-shadow-soft: 0 8px 22px rgba(0, 0, 0, 0.22);
                --tb-sidebar-bg: #0f1117;
                --tb-sidebar-panel: #171b22;
                --tb-sidebar-text: #f8fafc;
                --tb-sidebar-muted: #cbd5e1;
            }
        }
        a {
            color: var(--tb-accent);
            text-decoration-thickness: 1px;
            text-underline-offset: 3px;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--tb-card-border);
            background: var(--tb-sidebar-bg) !important;
            color: var(--tb-sidebar-text) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            padding: 1.05rem 0.85rem 1.4rem;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: var(--tb-sidebar-text);
        }
        .sidebar-header {
            padding: 0.7rem 0.75rem 0.85rem;
            margin: 0 0 0.85rem;
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-accent);
            border-radius: 8px;
            background: var(--tb-sidebar-panel);
        }
        .sidebar-kicker {
            color: var(--tb-accent) !important;
            font-size: 0.70rem;
            font-weight: 850;
            line-height: 1.15;
            text-transform: uppercase;
        }
        .sidebar-title {
            margin-top: 0.18rem;
            color: var(--tb-sidebar-text) !important;
            font-size: 1.05rem;
            font-weight: 780;
            line-height: 1.2;
        }
        section[data-testid="stSidebar"] div[data-testid="stTextInput"] {
            margin-bottom: 0.48rem;
        }
        section[data-testid="stSidebar"] div[data-baseweb="input"] {
            background: var(--tb-sidebar-panel) !important;
            border: 1px solid var(--tb-card-border) !important;
            min-height: 38px;
        }
        section[data-testid="stSidebar"] input {
            color: var(--tb-sidebar-text) !important;
            background: transparent !important;
            font-size: 0.82rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: var(--tb-sidebar-panel);
            border-color: var(--tb-card-border);
            margin-bottom: 0.75rem;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label,
        section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label {
            background: transparent !important;
            color: var(--tb-sidebar-text) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p,
        section[data-testid="stSidebar"] small {
            color: var(--tb-sidebar-muted) !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: var(--tb-sidebar-panel) !important;
            color: var(--tb-sidebar-text) !important;
        }
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        textarea {
            border-radius: 8px !important;
        }
        .stButton > button,
        button[kind="secondary"],
        button[kind="primary"] {
            border-radius: 8px !important;
            border: 1px solid var(--tb-card-border) !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }
        .stButton > button:hover {
            border-color: color-mix(in srgb, var(--tb-accent) 60%, var(--tb-card-border)) !important;
            background: color-mix(in srgb, var(--tb-accent) 8%, var(--tb-card-bg)) !important;
        }
        button[kind="primary"] {
            background: var(--tb-accent) !important;
            border-color: var(--tb-accent) !important;
            color: #ffffff !important;
        }
        div[data-testid="stTabs"] {
            margin-top: 0.25rem;
        }
        div[data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
            color: var(--tb-card-text);
            font-weight: 650;
            min-height: 38px;
            padding: 7px 10px;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 8px;
            padding-top: 2px;
            border-bottom: 1px solid var(--tb-card-border);
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--tb-accent);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--tb-card-border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--tb-card-bg) 82%, var(--background-color, #ffffff));
            overflow: hidden;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border: 1px solid var(--tb-card-border);
            border-radius: 8px;
            overflow: hidden;
            background: var(--tb-card-bg);
        }
        div[data-testid="stPlotlyChart"],
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stDeckGlJsonChart"] {
            border: 1px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 10px;
            background: var(--tb-card-bg);
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--tb-card-border);
        }
        div[role="radiogroup"] {
            gap: 0.35rem;
        }
        div[role="radiogroup"] label {
            border-radius: 8px;
        }
        div[role="radiogroup"] label:hover {
            color: var(--tb-card-text);
        }
        hr {
            margin: 1.4rem 0;
            border-color: var(--tb-card-border);
        }
        .market-brief {
            border: 1px solid var(--tb-card-border);
            border-left-width: 8px;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 8px 0 14px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
        }
        .market-brief-critical {
            border-left-color: var(--tb-critical-border);
            background: var(--tb-critical-bg);
        }
        .market-brief-warning {
            border-left-color: var(--tb-warning-border);
            background: var(--tb-warning-bg);
        }
        .market-brief-success {
            border-left-color: var(--tb-success-border);
            background: var(--tb-success-bg);
        }
        .market-brief-label {
            margin: 0;
            font-size: 0.78rem;
            text-transform: uppercase;
            color: var(--tb-card-muted);
            font-weight: 750;
        }
        .market-brief-title {
            margin: 4px 0 7px;
            font-size: 1.42rem;
            line-height: 1.24;
            font-weight: 770;
            color: var(--tb-card-text);
        }
        .market-brief-body {
            max-width: 1180px;
        }
        .market-brief-delta-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 14px;
            max-width: 1180px;
        }
        .brief-delta-card {
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            background: var(--tb-card-bg);
            padding: 12px 14px;
        }
        .brief-delta-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .brief-delta-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .brief-delta-card-success {
            border-left-color: var(--tb-success-border);
        }
        .brief-delta-card-neutral {
            border-left-color: var(--tb-card-border);
        }
        .brief-delta-label {
            margin: 0 0 5px;
            color: var(--tb-card-muted);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .brief-delta-answer {
            margin: 0 0 6px;
            color: var(--tb-card-text);
            font-size: 1.02rem;
            font-weight: 780;
            line-height: 1.25;
        }
        .brief-delta-detail {
            margin: 0;
            color: var(--tb-card-text);
            font-size: 0.91rem;
            line-height: 1.42;
        }
        .market-brief-copy {
            margin: 0 0 12px;
            color: var(--tb-card-text);
            line-height: 1.55;
            font-size: 0.98rem;
        }
        .market-brief-copy:last-child {
            margin-bottom: 12px;
        }
        .market-brief-next {
            margin: 0;
            color: var(--tb-card-text);
            font-weight: 680;
        }
        .market-brief-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 18px;
        }
        .market-brief-readouts {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 6px 0 18px;
        }
        .market-brief-readout {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 4px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 10px 12px;
            min-height: 68px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 4px;
        }
        .market-brief-readout-critical {
            border-left-color: var(--tb-critical-border);
        }
        .market-brief-readout-warning {
            border-left-color: var(--tb-warning-border);
        }
        .market-brief-readout-success {
            border-left-color: var(--tb-success-border);
        }
        .market-brief-readout-neutral {
            border-left-color: var(--tb-card-border);
        }
        .brief-readout-label {
            color: var(--tb-card-muted);
            font-size: 0.72rem;
            font-weight: 750;
            text-transform: uppercase;
            line-height: 1.2;
        }
        .brief-readout-answer {
            color: var(--tb-card-text);
            font-size: 0.94rem;
            font-weight: 740;
            line-height: 1.25;
        }
        .market-brief-card {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 13px 15px;
            min-height: 132px;
        }
        .market-brief-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .market-brief-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .market-brief-card-success {
            border-left-color: var(--tb-success-border);
        }
        .market-brief-card-neutral {
            border-left-color: var(--tb-card-border);
        }
        .brief-card-label {
            margin: 0 0 5px;
            color: var(--tb-card-muted);
            font-size: 0.74rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .brief-card-answer {
            margin: 0 0 7px;
            color: var(--tb-card-text);
            font-size: 1.0rem;
            line-height: 1.28;
            font-weight: 750;
        }
        .brief-card-detail {
            margin: 0;
            color: var(--tb-card-text);
            line-height: 1.42;
            font-size: 0.90rem;
        }
        .action-banner {
            border: 1px solid var(--tb-card-border);
            border-left-width: 8px;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 8px 0 18px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
        }
        .action-do_nothing {
            border-left-color: var(--tb-success-border);
            background: var(--tb-success-bg);
        }
        .action-small_actions {
            border-left-color: var(--tb-warning-border);
            background: var(--tb-warning-bg);
        }
        .action-critical_actions {
            border-left-color: var(--tb-critical-border);
            background: var(--tb-critical-bg);
        }
        .headline-label {
            margin: 0;
            font-size: 0.78rem;
            text-transform: uppercase;
            color: var(--tb-card-muted);
            font-weight: 700;
        }
        .headline-title {
            margin: 4px 0 6px;
            font-size: 1.45rem;
            line-height: 1.25;
            font-weight: 750;
            color: var(--tb-card-text);
        }
        .headline-copy {
            margin: 0 0 10px;
            color: var(--tb-card-text);
            line-height: 1.45;
        }
        .headline-next {
            margin: 0;
            color: var(--tb-card-text);
            font-weight: 650;
        }
        div[data-testid="stMetric"] {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 12px 14px;
            color: var(--tb-card-text);
            min-height: 94px;
            min-width: 0;
            overflow: hidden;
            overflow-wrap: anywhere;
            box-shadow: var(--tb-shadow-soft);
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            color: var(--tb-card-text) !important;
            min-width: 0;
            max-width: 100%;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] {
            color: var(--tb-card-muted) !important;
            line-height: 1.25 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            white-space: normal !important;
            overflow-wrap: anywhere;
            word-break: normal;
            hyphens: auto;
            font-size: clamp(1.18rem, 1.55vw, 1.95rem) !important;
            line-height: 1.12 !important;
            text-wrap: balance;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] *,
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] * {
            min-width: 0 !important;
            max-width: 100% !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
            word-break: normal !important;
        }
        .brand-masthead,
        .freshness-strip,
        .dashboard-section-header,
        .workbench-guide,
        .action-banner,
        .market-brief-card,
        .market-brief-readout,
        .brief-delta-card,
        .brief-card,
        .operating-card,
        .launch-guidance-card,
        .metric-info-card {
            min-width: 0;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .brand-masthead *,
        .freshness-strip *,
        .dashboard-section-header *,
        .workbench-guide *,
        .action-banner *,
        .market-brief-card *,
        .market-brief-readout *,
        .brief-delta-card *,
        .brief-card *,
        .operating-card *,
        .launch-guidance-card *,
        .metric-info-card * {
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: normal;
            text-overflow: clip;
        }
        .brief-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 16px;
        }
        .brief-card {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 155px;
        }
        .brief-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .brief-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .brief-card-success {
            border-left-color: var(--tb-success-border);
        }
        .brief-label {
            margin: 0 0 6px;
            color: var(--tb-card-muted);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .brief-answer {
            margin: 0 0 8px;
            color: var(--tb-card-text);
            font-size: 1.05rem;
            line-height: 1.3;
            font-weight: 750;
        }
        .brief-detail {
            margin: 0;
            color: var(--tb-card-text);
            line-height: 1.42;
            font-size: 0.92rem;
        }
        .operating-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 16px;
        }
        .operating-card {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 15px 17px;
            min-height: 145px;
        }
        .operating-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .operating-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .operating-card-success {
            border-left-color: var(--tb-success-border);
        }
        .operating-label {
            margin: 0 0 6px;
            color: var(--tb-card-muted);
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .operating-answer {
            margin: 0 0 8px;
            color: var(--tb-card-text);
            font-size: 1.08rem;
            line-height: 1.28;
            font-weight: 760;
        }
        .operating-detail {
            margin: 0;
            color: var(--tb-card-text);
            line-height: 1.45;
            font-size: 0.93rem;
        }
        .launch-summary-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 14px;
        }
        .launch-guidance-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 16px;
        }
        .launch-guidance-card {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 126px;
            min-width: 0;
            overflow: hidden;
            overflow-wrap: anywhere;
        }
        .launch-guidance-critical {
            border-left-color: var(--tb-critical-border);
            background: var(--tb-critical-bg);
        }
        .launch-guidance-warning {
            border-left-color: var(--tb-warning-border);
            background: var(--tb-warning-bg);
        }
        .launch-guidance-success {
            border-left-color: var(--tb-success-border);
            background: var(--tb-success-bg);
        }
        .launch-guidance-neutral {
            border-left-color: var(--tb-card-border);
        }
        .launch-card-label {
            margin: 0 0 6px;
            color: var(--tb-card-muted);
            font-size: 0.74rem;
            font-weight: 760;
            text-transform: uppercase;
        }
        .launch-card-answer {
            margin: 0 0 8px;
            color: var(--tb-card-text);
            font-size: clamp(0.98rem, 1.2vw, 1.2rem);
            line-height: 1.25;
            font-weight: 780;
            overflow-wrap: anywhere;
        }
        .launch-card-detail {
            margin: 0;
            color: var(--tb-card-text);
            line-height: 1.42;
            font-size: 0.90rem;
            overflow-wrap: anywhere;
        }
        @media (max-width: 1100px) {
            .brand-masthead {
                align-items: flex-start;
                flex-direction: column;
            }
            .brand-proof-row {
                justify-content: flex-start;
                max-width: none;
                padding-top: 0;
            }
            .workbench-guide {
                grid-template-columns: 1fr;
            }
            .workbench-guide-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .st-key-quick_reference_rail {
                position: static;
                width: auto;
                max-height: none;
                overflow: visible;
                padding: 0;
            }
            .metric-info-rail {
                margin-top: 1rem;
            }
            .simulation-validation-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .block-container {
                padding-right: 1rem !important;
                max-width: 1540px;
            }
            .market-brief-grid,
            .market-brief-readouts,
            .market-brief-delta-grid,
            .launch-summary-grid,
            .brief-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .launch-guidance-grid,
            .operating-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 700px) {
            .block-container {
                padding-top: 2rem;
            }
            .dashboard-section-header {
                margin-top: 26px;
                padding: 19px 16px 17px;
                border-left-width: 9px;
            }
            .dashboard-primary-nav-label {
                font-size: 1.58rem;
            }
            .dashboard-primary-nav-label::before {
                width: 12px;
                height: 30px;
            }
            .dashboard-nav-caption {
                margin-left: 0;
                font-size: 0.95rem;
            }
            .st-key-dashboard_main_station_nav div[data-testid="stPills"] button,
            .st-key-dashboard_main_station_nav div[data-testid="stPills"] [role="button"] {
                min-height: 54px;
                padding: 11px 16px !important;
                font-size: 1.02rem !important;
            }
            .brand-masthead {
                padding: 20px 15px 16px;
            }
            .brand-lockup {
                align-items: flex-start;
            }
            .brand-mark {
                width: 50px;
                height: 50px;
            }
            .brand-title {
                font-size: 1.55rem;
            }
            .market-brief-grid,
            .market-brief-readouts,
            .market-brief-delta-grid,
            .workbench-guide-grid,
            .simulation-validation-grid,
            .launch-summary-grid,
            .launch-guidance-grid,
            .brief-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _install_quick_reference_rail_layout() -> None:
    st.markdown(
        """
        <style>
        @media (min-width: 1101px) {
            .block-container {
                max-width: 1760px;
                padding-right: 370px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
