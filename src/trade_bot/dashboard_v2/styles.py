from __future__ import annotations

import streamlit as st


def install_v2_styles() -> None:
    st.markdown(
        """
        <style>
        .v2-masthead {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            padding: 22px 24px;
            margin-bottom: 14px;
            border: 1px solid rgba(15, 118, 110, 0.34);
            border-radius: 8px;
            background: linear-gradient(135deg, #111827 0%, #102522 58%, #0f766e 100%);
            color: #f8fafc;
            box-shadow: var(--tb-shadow, 0 10px 30px rgba(15, 23, 42, 0.12));
        }
        .v2-kicker {
            margin: 0 0 8px;
            color: #a7f3d0;
            font-size: .76rem;
            font-weight: 850;
            text-transform: uppercase;
        }
        .v2-title {
            margin: 0;
            color: #f8fafc !important;
            font-size: 2.1rem;
            line-height: 1.05;
            letter-spacing: 0;
        }
        .v2-subtitle {
            max-width: 920px;
            margin: 8px 0 0;
            color: rgba(248,250,252,.82);
            line-height: 1.42;
        }
        .v2-chip-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 8px;
            align-content: flex-start;
            padding-top: 4px;
        }
        .v2-chip {
            border: 1px solid rgba(248,250,252,.24);
            border-radius: 999px;
            padding: 5px 10px;
            background: rgba(248,250,252,.10);
            color: #f8fafc;
            font-size: .78rem;
            font-weight: 760;
            white-space: nowrap;
        }
        .v2-route-card {
            padding: 15px 16px;
            margin: 12px 0 18px;
            border: 1px solid var(--tb-card-border, #d1d5db);
            border-left: 6px solid var(--tb-accent, #0f766e);
            border-radius: 8px;
            background: color-mix(in srgb, var(--tb-card-bg, #ffffff) 92%, transparent);
        }
        .v2-route-card h2 {
            margin: 0 0 4px;
            font-size: 1.45rem;
            line-height: 1.15;
        }
        .v2-route-card p {
            margin: 4px 0;
            color: var(--tb-card-muted, #6b7280);
        }
        .v2-runtime {
            display: inline-flex;
            width: fit-content;
            margin-top: 8px;
            border-radius: 999px;
            padding: 3px 9px;
            background: rgba(245, 158, 11, .13);
            color: #92400e;
            font-size: .78rem;
            font-weight: 800;
        }
        .v2-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 12px;
            margin: 12px 0;
        }
        .v2-card {
            border: 1px solid var(--tb-card-border, #d1d5db);
            border-radius: 8px;
            padding: 14px 15px;
            background: var(--tb-card-bg, #fff);
            min-height: 112px;
        }
        .v2-card-label {
            margin: 0 0 8px;
            color: var(--tb-card-muted, #6b7280);
            font-size: .84rem;
            font-weight: 740;
        }
        .v2-help-wrap {
            position: relative;
            display: inline-flex;
            align-items: center;
            vertical-align: .08rem;
            outline: none;
        }
        .v2-help-dot {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.05rem;
            height: 1.05rem;
            margin-left: .38rem;
            border: 1px solid var(--tb-card-border, #d1d5db);
            border-radius: 999px;
            color: var(--tb-card-muted, #6b7280);
            font-size: .68rem;
            font-weight: 900;
            line-height: 1;
            cursor: help;
        }
        .v2-help-wrap:hover .v2-help-dot,
        .v2-help-wrap:focus-within .v2-help-dot {
            color: var(--tb-card-text, #111827);
            border-color: #0f766e;
        }
        .v2-help-popover {
            position: absolute;
            left: .35rem;
            top: calc(100% + 8px);
            z-index: 10000;
            width: min(320px, 72vw);
            padding: 10px 12px;
            border: 1px solid var(--tb-card-border, #d1d5db);
            border-radius: 8px;
            background: var(--tb-card-bg, #fff);
            color: var(--tb-card-text, #111827);
            box-shadow: 0 14px 34px rgba(15, 23, 42, .20);
            font-size: .78rem;
            font-weight: 650;
            line-height: 1.35;
            text-transform: none;
            white-space: normal;
            pointer-events: none;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-2px);
            transition: opacity .12s ease, transform .12s ease, visibility .12s ease;
        }
        .v2-help-popover::before {
            content: "";
            position: absolute;
            top: -5px;
            left: 12px;
            width: 8px;
            height: 8px;
            border-left: 1px solid var(--tb-card-border, #d1d5db);
            border-top: 1px solid var(--tb-card-border, #d1d5db);
            background: var(--tb-card-bg, #fff);
            transform: rotate(45deg);
        }
        .v2-help-wrap:hover .v2-help-popover,
        .v2-help-wrap:focus-within .v2-help-popover {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        .v2-card-value {
            margin: 0;
            color: var(--tb-card-text, #111827);
            font-size: 1.6rem;
            font-weight: 840;
            line-height: 1.12;
            overflow-wrap: anywhere;
        }
        .v2-section-heading h3 {
            display: inline-flex;
            align-items: center;
            margin: 22px 0 10px;
            color: var(--tb-card-text, #111827);
            font-size: 1.5rem;
            font-weight: 850;
        }
        .v2-chart-heading {
            display: inline-flex;
            align-items: center;
            margin: 6px 0 2px;
            color: var(--tb-card-text, #111827);
            font-size: 1rem;
            font-weight: 850;
        }
        .v2-callout {
            border: 1px solid rgba(37, 99, 235, .24);
            border-radius: 8px;
            background: rgba(37, 99, 235, .10);
            color: #1d4ed8;
            padding: 12px 14px;
            margin: 12px 0;
        }
        .v2-heavy-callout {
            border-color: rgba(245, 158, 11, .36);
            background: rgba(245, 158, 11, .12);
            color: #92400e;
        }
        .v2-decision-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
            gap: 12px;
            margin: 12px 0 16px;
        }
        .v2-decision-card {
            border: 1px solid var(--tb-card-border, #d1d5db);
            border-radius: 8px;
            padding: 14px 15px;
            background: var(--tb-card-bg, #fff);
            min-height: 156px;
        }
        .v2-decision-answer {
            margin: 0 0 8px;
            color: var(--tb-card-text, #111827);
            font-size: 1.02rem;
            font-weight: 850;
            line-height: 1.25;
        }
        .v2-decision-detail {
            margin: 0;
            color: var(--tb-card-muted, #6b7280);
            font-size: .90rem;
            line-height: 1.42;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
