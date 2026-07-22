from __future__ import annotations

import html
from collections.abc import Iterable
from typing import Any

import streamlit as st
from plotly.graph_objects import Figure

from trade_bot.dashboard_v2.components.tones import normalize_tone
from trade_bot.dashboard_v2.help import chart_help, metric_help, section_help

type CardSpec = (
    tuple[str, object]
    | tuple[str, object, str | None]
    | tuple[str, object, str | None, object | None]
)


def render_card_grid(cards: Iterable[CardSpec]) -> None:
    html_cards = []
    for card in cards:
        label, value, help_text, tone = _normalise_card(card)
        resolved_help = help_text or metric_help(label)
        tone_class = _tone_class(tone)
        html_cards.append(
            f'<div class="v2-card{tone_class}">'
            f'<p class="v2-card-label">{html.escape(label)}{_help_icon(resolved_help)}</p>'
            f'<p class="v2-card-value">{html.escape(str(value))}</p>'
            "</div>"
        )
    st.markdown(
        f"<div class=\"v2-grid\">{''.join(html_cards)}</div>",
        unsafe_allow_html=True,
    )


def render_callout(message: str, *, heavy: bool = False) -> None:
    class_name = "v2-callout v2-heavy-callout" if heavy else "v2-callout"
    st.markdown(
        f'<div class="{class_name}">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, *, help_text: str | None = None) -> None:
    resolved_help = help_text or section_help(title)
    st.markdown(
        '<div class="v2-section-heading">'
        f"<h3>{html.escape(title)}{_help_icon(resolved_help)}</h3>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_chart(
    figure: Figure,
    *,
    title: str | None = None,
    help_text: str | None = None,
    width: str | int = "stretch",
    **kwargs: Any,
) -> None:
    chart_title = title or _figure_title(figure)
    if chart_title:
        st.markdown(
            '<div class="v2-chart-heading">'
            f"<span>{html.escape(chart_title)}</span>{_help_icon(help_text or chart_help(chart_title))}"
            "</div>",
            unsafe_allow_html=True,
        )
        _strip_figure_title(figure)
    st.plotly_chart(
        figure,
        width=width,
        **kwargs,
    )


def help_icon(help_text: str | None) -> str:
    return _help_icon(help_text)


def _normalise_card(card: CardSpec) -> tuple[str, object, str | None, object | None]:
    if len(card) == 2:
        label, value = card
        return str(label), value, None, None
    if len(card) == 3:
        label, value, help_text = card
        return str(label), value, help_text, None
    label, value, help_text, tone = card
    return str(label), value, help_text, tone


def _tone_class(tone: object | None) -> str:
    normalized = normalize_tone(tone)
    if normalized == "neutral":
        return ""
    return f" v2-card-{normalized}"


def _figure_title(figure: Figure) -> str | None:
    title = getattr(getattr(figure, "layout", None), "title", None)
    text = getattr(title, "text", None)
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned or cleaned.lower() in {"none", "nan", "undefined"}:
        return None
    return cleaned


def _strip_figure_title(figure: Figure) -> None:
    figure.update_layout(title_text="")
    title = getattr(getattr(figure, "layout", None), "title", None)
    if title is not None:
        title.text = ""


def _help_icon(help_text: str | None) -> str:
    if not help_text:
        return ""
    escaped_help = html.escape(str(help_text), quote=True)
    return (
        '<span class="v2-help-wrap" tabindex="0">'
        '<span class="v2-help-dot" aria-hidden="true">?</span>'
        f'<span class="v2-help-popover" role="tooltip">{escaped_help}</span>'
        "</span>"
    )
