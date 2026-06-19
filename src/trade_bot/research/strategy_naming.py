from __future__ import annotations

import re

TOKEN_LABELS = {
    "ai": "AI",
    "api": "API",
    "beta": "Beta",
    "bil": "T-Bill",
    "capex": "Capex",
    "core": "Core",
    "cross": "Cross",
    "etf": "ETF",
    "fx": "FX",
    "hormuz": "Hormuz",
    "ipo": "IPO",
    "lowvol": "Low Vol",
    "macro": "Macro",
    "mega": "Mega",
    "qqq": "QQQ",
    "riskadj": "Risk-Adjusted",
    "risk": "Risk",
    "spy": "SPY",
    "tbill": "T-Bill",
    "us": "US",
    "momentum_state": "Vol-Adjusted Momentum",
    "vol": "Vol",
}


def strategy_display_name(
    strategy_id: str,
    *,
    family: str | None = None,
    phase: str | None = None,
) -> str:
    clean_id = _strip_iteration_prefix(strategy_id)
    label = _title_tokens(clean_id.split("_"))
    family_label = _title_tokens((family or "").split("_"))
    phase_label = _title_tokens((phase or "").split("_"))
    if family_label and family_label.lower() not in label.lower():
        label = f"{family_label}: {label}"
    if phase_label and phase_label.lower() in {"reference", "baseline"}:
        label = f"{label} Reference"
    return _squash_spaces(label)


def canonical_strategy_id(*, family: str, behavior: str, variant: str, number: int) -> str:
    """Return the reset-era ID format for new experiment batches.

    Example: ``mw_growth_liquidity_01`` is easier to read and group than the
    old iteration-prefixed names while still staying filesystem-safe.
    """
    pieces = [family, behavior, f"{number:02d}", variant]
    return "_".join(_slug(piece) for piece in pieces if piece)


def _strip_iteration_prefix(strategy_id: str) -> str:
    return re.sub(r"^i\d+_", "", str(strategy_id).strip())


def _title_tokens(tokens: list[str]) -> str:
    labels = []
    for token in tokens:
        if not token:
            continue
        labels.append(TOKEN_LABELS.get(token.lower(), token.replace("-", " ").title()))
    return " ".join(labels)


def _slug(value: str) -> str:
    lowered = str(value).strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def _squash_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
