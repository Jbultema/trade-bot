from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st


def render_card_grid(cards: Iterable[tuple[str, object]]) -> None:
    html_cards = []
    for label, value in cards:
        html_cards.append(
            f"""
            <div class="v2-card">
                <p class="v2-card-label">{html.escape(str(label))}</p>
                <p class="v2-card-value">{html.escape(str(value))}</p>
            </div>
            """
        )
    st.markdown(
        f"<div class=\"v2-grid\">{''.join(html_cards)}</div>",
        unsafe_allow_html=True,
    )


def render_callout(message: str, *, heavy: bool = False) -> None:
    class_name = "v2-callout v2-heavy-callout" if heavy else "v2-callout"
    st.markdown(
        f"<div class=\"{class_name}\">{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )

