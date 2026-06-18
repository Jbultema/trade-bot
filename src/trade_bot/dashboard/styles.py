from __future__ import annotations

import streamlit as st


def _install_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        h1 {
            font-size: 2rem;
            letter-spacing: 0;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        :root {
            --tb-card-bg: var(--secondary-background-color, #ffffff);
            --tb-card-border: rgba(125, 139, 155, 0.35);
            --tb-card-text: var(--text-color, #111827);
            --tb-card-muted: #4b5563;
            --tb-card-muted: color-mix(in srgb, var(--text-color, #111827) 68%, transparent);
            --tb-critical-bg: #fff5f5;
            --tb-critical-bg: color-mix(in srgb, #ef4444 12%, var(--background-color, #ffffff));
            --tb-critical-border: #c53030;
            --tb-warning-bg: #fffaf0;
            --tb-warning-bg: color-mix(in srgb, #f59e0b 12%, var(--background-color, #ffffff));
            --tb-warning-border: #b7791f;
            --tb-success-bg: #f7fbf8;
            --tb-success-bg: color-mix(in srgb, #22c55e 10%, var(--background-color, #ffffff));
            --tb-success-border: #2f855a;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --tb-card-bg: #171b22;
                --tb-card-border: #2d3440;
                --tb-card-text: #f8fafc;
                --tb-card-muted: #cbd5e1;
                --tb-critical-bg: #2a1518;
                --tb-critical-border: #ef4444;
                --tb-warning-bg: #251d10;
                --tb-warning-border: #f59e0b;
                --tb-success-bg: #102018;
                --tb-success-border: #22c55e;
            }
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
            .macro-minute-grid,
            .macro-minute-readouts,
            .brief-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .operating-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 700px) {
            .macro-minute-grid,
            .macro-minute-readouts,
            .brief-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
