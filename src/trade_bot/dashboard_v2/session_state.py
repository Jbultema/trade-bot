from __future__ import annotations

DEFAULT_ROUTE_KEY = "today"


def route_state_key() -> str:
    return "dashboard_v2_route"


def route_view_state_key(route_key: str) -> str:
    return f"dashboard_v2_{route_key}_view"


def heavy_gate_key(route_key: str, gate_name: str) -> str:
    return f"dashboard_v2_{route_key}_{gate_name}_loaded"

