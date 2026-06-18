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
        .dashboard-section-header {
            margin: 24px 0 8px;
            padding: 16px 18px 14px;
            border: 1px solid var(--tb-card-border);
            border-left: 6px solid var(--tb-accent);
            border-radius: 8px;
            background:
                linear-gradient(90deg, color-mix(in srgb, var(--tb-accent) 10%, var(--tb-card-bg)), var(--tb-card-bg));
            box-shadow: var(--tb-shadow-soft);
        }
        .dashboard-section-kicker {
            margin: 0 0 5px;
            color: var(--tb-accent);
            font-size: 0.76rem;
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.2;
            text-transform: uppercase;
        }
        .dashboard-primary-nav-label {
            margin: 0;
            color: var(--tb-card-text);
            font-size: 1.32rem;
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.18;
        }
        .dashboard-nav-caption {
            margin: 6px 0 0;
            color: var(--tb-card-muted);
            font-size: 0.98rem;
            line-height: 1.38;
        }
        div[data-testid="stPills"] {
            margin: 0 0 1.15rem;
            padding: 10px 0 1.05rem;
            border-bottom: 1px solid var(--tb-card-border);
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
            }
        }
        a {
            color: var(--tb-accent);
            text-decoration-thickness: 1px;
            text-underline-offset: 3px;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--tb-card-border);
            background: color-mix(in srgb, var(--secondary-background-color, #f8fafc) 88%, var(--background-color, #ffffff));
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            padding-top: 1.05rem;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {
            color: var(--tb-card-text);
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
        div[data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
            color: var(--tb-card-text);
            font-weight: 650;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 4px;
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
        .macro-minute {
            border: 1px solid var(--tb-card-border);
            border-left-width: 8px;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 8px 0 14px;
            background: var(--tb-card-bg);
            color: var(--tb-card-text);
        }
        .macro-minute-critical {
            border-left-color: var(--tb-critical-border);
            background: var(--tb-critical-bg);
        }
        .macro-minute-warning {
            border-left-color: var(--tb-warning-border);
            background: var(--tb-warning-bg);
        }
        .macro-minute-success {
            border-left-color: var(--tb-success-border);
            background: var(--tb-success-bg);
        }
        .macro-minute-label {
            margin: 0;
            font-size: 0.78rem;
            text-transform: uppercase;
            color: var(--tb-card-muted);
            font-weight: 750;
        }
        .macro-minute-title {
            margin: 4px 0 7px;
            font-size: 1.42rem;
            line-height: 1.24;
            font-weight: 770;
            color: var(--tb-card-text);
        }
        .macro-minute-body {
            max-width: 1180px;
        }
        .macro-minute-delta-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 14px;
            max-width: 1180px;
        }
        .macro-delta-card {
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            background: var(--tb-card-bg);
            padding: 12px 14px;
        }
        .macro-delta-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .macro-delta-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .macro-delta-card-success {
            border-left-color: var(--tb-success-border);
        }
        .macro-delta-card-neutral {
            border-left-color: var(--tb-card-border);
        }
        .macro-delta-label {
            margin: 0 0 5px;
            color: var(--tb-card-muted);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .macro-delta-answer {
            margin: 0 0 6px;
            color: var(--tb-card-text);
            font-size: 1.02rem;
            font-weight: 780;
            line-height: 1.25;
        }
        .macro-delta-detail {
            margin: 0;
            color: var(--tb-card-text);
            font-size: 0.91rem;
            line-height: 1.42;
        }
        .macro-minute-copy {
            margin: 0 0 12px;
            color: var(--tb-card-text);
            line-height: 1.55;
            font-size: 0.98rem;
        }
        .macro-minute-copy:last-child {
            margin-bottom: 12px;
        }
        .macro-minute-next {
            margin: 0;
            color: var(--tb-card-text);
            font-weight: 680;
        }
        .macro-minute-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 18px;
        }
        .macro-minute-readouts {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 6px 0 18px;
        }
        .macro-minute-readout {
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
        .macro-minute-readout-critical {
            border-left-color: var(--tb-critical-border);
        }
        .macro-minute-readout-warning {
            border-left-color: var(--tb-warning-border);
        }
        .macro-minute-readout-success {
            border-left-color: var(--tb-success-border);
        }
        .macro-minute-readout-neutral {
            border-left-color: var(--tb-card-border);
        }
        .macro-readout-label {
            color: var(--tb-card-muted);
            font-size: 0.72rem;
            font-weight: 750;
            text-transform: uppercase;
            line-height: 1.2;
        }
        .macro-readout-answer {
            color: var(--tb-card-text);
            font-size: 0.94rem;
            font-weight: 740;
            line-height: 1.25;
        }
        .macro-minute-card {
            background: var(--tb-card-bg);
            border: 1px solid var(--tb-card-border);
            border-left: 5px solid var(--tb-card-border);
            border-radius: 8px;
            padding: 13px 15px;
            min-height: 132px;
        }
        .macro-minute-card-critical {
            border-left-color: var(--tb-critical-border);
        }
        .macro-minute-card-warning {
            border-left-color: var(--tb-warning-border);
        }
        .macro-minute-card-success {
            border-left-color: var(--tb-success-border);
        }
        .macro-minute-card-neutral {
            border-left-color: var(--tb-card-border);
        }
        .macro-card-label {
            margin: 0 0 5px;
            color: var(--tb-card-muted);
            font-size: 0.74rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .macro-card-answer {
            margin: 0 0 7px;
            color: var(--tb-card-text);
            font-size: 1.0rem;
            line-height: 1.28;
            font-weight: 750;
        }
        .macro-card-detail {
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
            box-shadow: var(--tb-shadow-soft);
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            color: var(--tb-card-text) !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] {
            color: var(--tb-card-muted) !important;
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
            .macro-minute-grid,
            .macro-minute-readouts,
            .macro-minute-delta-grid,
            .brief-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .operating-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 700px) {
            .block-container {
                padding-top: 2rem;
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
            .macro-minute-grid,
            .macro-minute-readouts,
            .macro-minute-delta-grid,
            .brief-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
