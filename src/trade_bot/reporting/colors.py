from __future__ import annotations

from collections.abc import Iterable, Mapping

SERIES_COLOR_SEQUENCE: tuple[str, ...] = (
    "#f97316",
    "#8b5cf6",
    "#ec4899",
    "#0f766e",
    "#dc2626",
    "#f59e0b",
    "#0891b2",
    "#84cc16",
    "#a855f7",
    "#64748b",
    "#2563eb",
    "#10b981",
)

PREFERRED_SERIES_COLORS: dict[str, str] = {
    "buy_hold_spy": "#5b6cff",
    "hold_spy": "#5b6cff",
    "spy": "#5b6cff",
    "buy_hold_qqq": "#00c896",
    "hold_qqq": "#00c896",
    "qqq": "#00c896",
    "buy_hold_bil": "#94a3b8",
    "bil": "#94a3b8",
    "i41_ref_us_60_40": "#f97316",
}

ALLOCATION_EXPOSURE_COLORS: dict[str, str] = {
    "risk_assets": "#0f766e",
    "risk assets": "#0f766e",
    "defensive": "#f59e0b",
    "cash_or_unallocated": "#94a3b8",
    "cash or unallocated": "#94a3b8",
}


def series_color_map(
    series_names: Iterable[object],
    *,
    preferred: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return one stable color per visible series inside a single plot block."""

    ordered_names = list(dict.fromkeys(str(name) for name in series_names if str(name)))
    preferred_colors = {**PREFERRED_SERIES_COLORS, **dict(preferred or {})}
    color_map: dict[str, str] = {}
    for name in ordered_names:
        color = preferred_colors.get(name) or preferred_colors.get(name.lower())
        if color:
            color_map[name] = color

    next_color = 0
    for name in ordered_names:
        if name in color_map:
            continue
        color = SERIES_COLOR_SEQUENCE[next_color % len(SERIES_COLOR_SEQUENCE)]
        next_color += 1
        color_map[name] = color

    return color_map
