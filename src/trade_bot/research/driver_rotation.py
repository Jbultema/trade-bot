from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trade_bot.DEFAULTS import (
    DEFAULT_DRIVER_ROTATION_ACTIVE_THRESHOLD,
    DEFAULT_DRIVER_ROTATION_CONFIRMATION_THEME_MAP,
    DEFAULT_DRIVER_ROTATION_EMERGING_DELTA_THRESHOLD,
    DEFAULT_DRIVER_ROTATION_FADING_DELTA_THRESHOLD,
    DEFAULT_DRIVER_ROTATION_FALLBACK_RELEVANCE,
    DEFAULT_DRIVER_ROTATION_LONG_LOOKBACK_DAYS,
    DEFAULT_DRIVER_ROTATION_MACRO_CATEGORY_MAP,
    DEFAULT_DRIVER_ROTATION_ML_FAMILY_IMPORTANCE_PATH,
    DEFAULT_DRIVER_ROTATION_NARRATIVE_SIGNAL_MAP,
    DEFAULT_DRIVER_ROTATION_NEWS_CATEGORY_MAP,
    DEFAULT_DRIVER_ROTATION_PRICE_PROXY_SPECS,
    DEFAULT_DRIVER_ROTATION_PROVEN_THRESHOLD,
    DEFAULT_DRIVER_ROTATION_SHORT_LOOKBACK_DAYS,
)

DRIVER_ROTATION_COLUMNS = [
    "driver",
    "driver_label",
    "model_role",
    "primary_rotation_state",
    "normally_important",
    "currently_active",
    "emerging_importance",
    "fading_importance",
    "proven_relevance",
    "current_activation",
    "previous_30d_activation",
    "previous_90d_activation",
    "change_30d",
    "change_90d",
    "data_support",
    "source_count",
    "sources",
    "evidence",
    "interpretation",
]

_FEATURE_FAMILY_DRIVER_MAP = {
    "ai_leadership": "ai_leadership",
    "breadth": "breadth",
    "broad_equity": "trend",
    "commodities": "commodities",
    "credit": "credit",
    "dollar": "dollar_liquidity",
    "drawdown": "drawdown",
    "duration_rates": "duration_rates",
    "volatility": "volatility",
}

_CONTEXT_ONLY_DRIVERS = {
    "concentration",
    "drawdown",
    "positioning",
    "private_credit",
    "regime_instability",
}
_EXPLAINER_ONLY_DRIVERS = {"ai_capex", "equity_supply", "unsupported_watchlist"}


def build_driver_rotation_table(
    prices: pd.DataFrame,
    current_state: Any,
    *,
    narrative_signals: pd.DataFrame | None = None,
    news_triage: pd.DataFrame | None = None,
    family_importance_path: str | Path | None = DEFAULT_DRIVER_ROTATION_ML_FAMILY_IMPORTANCE_PATH,
    short_lookback_days: int = DEFAULT_DRIVER_ROTATION_SHORT_LOOKBACK_DAYS,
    long_lookback_days: int = DEFAULT_DRIVER_ROTATION_LONG_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build the Driver Rotation research table for dashboard interpretation.

    The table deliberately separates historical relevance from current
    activation. It is an insight layer, not a hidden trading rule.
    """

    clean_prices = prices.sort_index().ffill() if not prices.empty else pd.DataFrame()
    relevance = _historical_relevance(family_importance_path)
    current_price_records = _price_activation_records(clean_prices)
    previous_30d = _price_activation_by_driver(_prices_as_of_lookback(clean_prices, short_lookback_days))
    previous_90d = _price_activation_by_driver(_prices_as_of_lookback(clean_prices, long_lookback_days))

    records: list[dict[str, object]] = [*current_price_records]
    records.extend(_confirmation_records(getattr(current_state, "confirmation_matrix", pd.DataFrame())))
    records.extend(_macro_records(getattr(current_state, "macro_category_summary", pd.DataFrame())))
    records.extend(_narrative_records(narrative_signals if narrative_signals is not None else pd.DataFrame()))
    records.extend(_news_records(news_triage if news_triage is not None else pd.DataFrame()))
    records.extend(_regime_instability_records(getattr(current_state, "regime_instability", pd.DataFrame())))

    record_frame = pd.DataFrame(records)
    drivers = sorted(
        {
            *DEFAULT_DRIVER_ROTATION_FALLBACK_RELEVANCE.keys(),
            *relevance.keys(),
            *(
                set(record_frame["driver"].dropna().astype(str))
                if not record_frame.empty and "driver" in record_frame
                else set()
            ),
        }
    )
    rows = [
        _driver_row(
            driver,
            record_frame,
            relevance.get(driver, 0.0),
            previous_30d.get(driver, np.nan),
            previous_90d.get(driver, np.nan),
        )
        for driver in drivers
    ]
    return pd.DataFrame(rows, columns=DRIVER_ROTATION_COLUMNS).sort_values(
        ["currently_active", "emerging_importance", "proven_relevance", "current_activation"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def summarize_driver_rotation(rotation: pd.DataFrame) -> dict[str, str]:
    if rotation.empty:
        return {
            "answer": "No driver-rotation table",
            "detail": "Run the current-state pipeline to compare proven relevance with current activation.",
            "tone": "neutral",
        }

    active = rotation[rotation["currently_active"]]
    emerging = rotation[rotation["emerging_importance"]]
    fading = rotation[rotation["fading_importance"]]
    normal = rotation[rotation["normally_important"]]
    if active.empty:
        return {
            "answer": "No dominant driver firing",
            "detail": (
                f"{len(normal)} historically relevant driver(s) are tracked, but none are "
                "currently above the activation threshold."
            ),
            "tone": "neutral",
        }

    top = active.sort_values(["current_activation", "proven_relevance"], ascending=False).head(3)
    top_drivers = ", ".join(str(value) for value in top["driver_label"])
    return {
        "answer": f"{len(active)} active driver(s)",
        "detail": (
            f"Most active drivers: {top_drivers}. Emerging: {len(emerging)}; "
            f"fading: {len(fading)}; historically important: {len(normal)}."
        ),
        "tone": "warning" if len(emerging) or len(fading) else "neutral",
    }


def _historical_relevance(path: str | Path | None) -> dict[str, float]:
    relevance = {key: float(value) for key, value in DEFAULT_DRIVER_ROTATION_FALLBACK_RELEVANCE.items()}
    if path is None:
        return relevance
    file_path = Path(path)
    if not file_path.exists():
        return relevance
    try:
        frame = pd.read_csv(file_path)
    except (OSError, ValueError):
        return relevance
    if frame.empty or "feature_family" not in frame or "mean_importance" not in frame:
        return relevance

    ml = frame.copy()
    ml["driver"] = ml["feature_family"].astype(str).map(_FEATURE_FAMILY_DRIVER_MAP)
    ml["mean_importance"] = pd.to_numeric(ml["mean_importance"], errors="coerce")
    grouped = ml.dropna(subset=["driver", "mean_importance"]).groupby("driver")[
        "mean_importance"
    ].mean()
    if grouped.empty:
        return relevance
    max_value = float(grouped.max())
    if max_value <= 0:
        return relevance
    for driver, value in grouped.items():
        relevance[str(driver)] = max(relevance.get(str(driver), 0.0), float(value) / max_value)
    return relevance


def _price_activation_records(prices: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for driver, label, numerator, denominator, lookback, threshold in DEFAULT_DRIVER_ROTATION_PRICE_PROXY_SPECS:
        measurement = _proxy_measurement(prices, numerator, denominator, int(lookback), float(threshold))
        if measurement is None:
            continue
        activation, value = measurement
        direction = "up" if value >= 0 else "down"
        records.append(
            {
                "driver": driver,
                "source": f"price:{label}",
                "activation": activation,
                "data_support": "market_price_proxy",
                "evidence": f"{label} {direction} {value:.1%} over {int(lookback)} trading days.",
            }
        )
    return records


def _price_activation_by_driver(prices: pd.DataFrame) -> dict[str, float]:
    activations: dict[str, float] = {}
    for row in _price_activation_records(prices):
        driver = str(row["driver"])
        activations[driver] = max(activations.get(driver, 0.0), float(row["activation"]))
    return activations


def _prices_as_of_lookback(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if prices.empty:
        return prices
    clean = prices.sort_index()
    if not isinstance(clean.index, pd.DatetimeIndex):
        if len(clean) <= lookback_days:
            return clean.iloc[:0]
        return clean.iloc[:-lookback_days]
    cutoff = clean.index.max() - pd.Timedelta(days=lookback_days)
    prior = clean[clean.index <= cutoff]
    if not prior.empty:
        return prior
    if len(clean) <= lookback_days:
        return clean.iloc[:0]
    return clean.iloc[:-lookback_days]


def _proxy_measurement(
    prices: pd.DataFrame,
    numerator: str,
    denominator: str | None,
    lookback: int,
    threshold: float,
) -> tuple[float, float] | None:
    if prices.empty or numerator not in prices:
        return None
    if denominator is not None and denominator not in prices:
        return None
    if denominator is None:
        series = prices[numerator].dropna()
    else:
        series = (prices[numerator] / prices[denominator]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return None
    periods = min(lookback, len(series) - 1)
    start = float(series.iloc[-periods - 1])
    end = float(series.iloc[-1])
    if start == 0 or not np.isfinite(start) or not np.isfinite(end):
        return None
    value = end / start - 1.0
    activation = _clip01(abs(value) / threshold) if threshold > 0 else 0.0
    return activation, value


def _confirmation_records(confirmation_matrix: pd.DataFrame) -> list[dict[str, object]]:
    if confirmation_matrix.empty or "theme" not in confirmation_matrix:
        return []
    rows: list[dict[str, object]] = []
    for _, row in confirmation_matrix.iterrows():
        driver = DEFAULT_DRIVER_ROTATION_CONFIRMATION_THEME_MAP.get(str(row.get("theme", "")))
        if not driver:
            continue
        score = _as_float(row.get("score"), default=0.0)
        rows.append(
            {
                "driver": driver,
                "source": "confirmation_matrix",
                "activation": min(abs(score), 1.0),
                "data_support": "market_confirmation",
                "evidence": (
                    f"{row.get('name', driver)} is {row.get('status', 'n/a')} "
                    f"with score {score:.2f}."
                ),
            }
        )
    return rows


def _macro_records(macro_category_summary: pd.DataFrame) -> list[dict[str, object]]:
    if macro_category_summary.empty or "category" not in macro_category_summary:
        return []
    rows: list[dict[str, object]] = []
    for _, row in macro_category_summary.iterrows():
        driver = DEFAULT_DRIVER_ROTATION_MACRO_CATEGORY_MAP.get(str(row.get("category", "")))
        if not driver:
            continue
        score = _as_float(row.get("mean_risk_score"), default=0.0)
        rows.append(
            {
                "driver": driver,
                "source": f"macro:{row.get('category', driver)}",
                "activation": min(abs(score), 1.0),
                "data_support": "macro_fred",
                "evidence": (
                    f"Macro category {row.get('category', driver)} has risk score {score:.2f} "
                    f"and state {row.get('risk_state', 'n/a')}."
                ),
            }
        )
    return rows


def _narrative_records(narrative_signals: pd.DataFrame) -> list[dict[str, object]]:
    if narrative_signals.empty or "signal_id" not in narrative_signals:
        return []
    rows: list[dict[str, object]] = []
    for _, row in narrative_signals.iterrows():
        signal_id = str(row.get("signal_id", ""))
        driver = DEFAULT_DRIVER_ROTATION_NARRATIVE_SIGNAL_MAP.get(signal_id)
        if not driver:
            continue
        rows.append(
            {
                "driver": driver,
                "source": f"narrative:{signal_id}",
                "activation": _as_float(row.get("score"), default=0.0),
                "data_support": str(row.get("data_support", "thin_proxy")),
                "evidence": f"{row.get('signal_name', signal_id)}: {row.get('evidence', '')}",
            }
        )
    return rows


def _news_records(news_triage: pd.DataFrame) -> list[dict[str, object]]:
    if news_triage.empty or "category" not in news_triage:
        return []
    rows: list[dict[str, object]] = []
    for category, group in news_triage.groupby("category"):
        driver = DEFAULT_DRIVER_ROTATION_NEWS_CATEGORY_MAP.get(str(category))
        if not driver:
            continue
        urgency = pd.to_numeric(group.get("urgency_score", pd.Series(dtype=float)), errors="coerce")
        activation = float(urgency.max()) if not urgency.empty and urgency.notna().any() else 0.0
        active_count = int(
            group.get("activation_status", pd.Series(dtype=object))
            .astype(str)
            .isin(["active", "activated", "warning"])
            .sum()
        )
        if active_count:
            activation = max(activation, DEFAULT_DRIVER_ROTATION_ACTIVE_THRESHOLD)
        rows.append(
            {
                "driver": driver,
                "source": f"news:{category}",
                "activation": min(activation, 1.0),
                "data_support": "news_triage",
                "evidence": f"{active_count} active item(s) in news category {category}.",
            }
        )
    return rows


def _regime_instability_records(regime_instability: pd.DataFrame) -> list[dict[str, object]]:
    if regime_instability.empty:
        return []
    row = regime_instability.iloc[0]
    score = _as_float(row.get("regime_instability_score"), default=0.0)
    return [
        {
            "driver": "regime_instability",
            "source": "regime_instability",
            "activation": min(max(score, 0.0), 1.0),
            "data_support": "market_price_proxy",
            "evidence": (
                f"Instability is {row.get('regime_instability_state', 'n/a')} "
                f"with score {score:.2f}."
            ),
        }
    ]


def _driver_row(
    driver: str,
    records: pd.DataFrame,
    proven_relevance: float,
    previous_30d: float,
    previous_90d: float,
) -> dict[str, object]:
    group = (
        records[records["driver"].astype(str) == driver].copy()
        if not records.empty and "driver" in records
        else pd.DataFrame()
    )
    if group.empty:
        current_activation = 0.0
        source_count = 0
        sources = ""
        evidence = "No current activation source is firing."
        data_support = "not_currently_active"
    else:
        group["activation"] = pd.to_numeric(group["activation"], errors="coerce").fillna(0.0)
        group = group.sort_values("activation", ascending=False)
        current_activation = float(group["activation"].max())
        source_count = int(group["source"].nunique())
        sources = "; ".join(str(value) for value in group["source"].dropna().unique()[:5])
        evidence = " ".join(str(value) for value in group["evidence"].dropna().head(3))
        data_support = _combined_data_support(group["data_support"])

    previous_30d = _nan_if_missing(previous_30d)
    previous_90d = _nan_if_missing(previous_90d)
    change_30d = current_activation - previous_30d if np.isfinite(previous_30d) else np.nan
    change_90d = current_activation - previous_90d if np.isfinite(previous_90d) else np.nan
    normally_important = proven_relevance >= DEFAULT_DRIVER_ROTATION_PROVEN_THRESHOLD
    currently_active = current_activation >= DEFAULT_DRIVER_ROTATION_ACTIVE_THRESHOLD
    emerging = currently_active and (
        not normally_important
        or _max_change(change_30d, change_90d) >= DEFAULT_DRIVER_ROTATION_EMERGING_DELTA_THRESHOLD
    )
    fading = (
        _min_change(change_30d, change_90d) <= DEFAULT_DRIVER_ROTATION_FADING_DELTA_THRESHOLD
        or (
            max(_finite_or_zero(previous_30d), _finite_or_zero(previous_90d))
            >= DEFAULT_DRIVER_ROTATION_ACTIVE_THRESHOLD
            and current_activation < DEFAULT_DRIVER_ROTATION_ACTIVE_THRESHOLD
        )
    )
    model_role = _model_role(driver, proven_relevance, data_support, normally_important)
    primary_state = _primary_rotation_state(normally_important, currently_active, emerging, fading)
    return {
        "driver": driver,
        "driver_label": _driver_label(driver),
        "model_role": model_role,
        "primary_rotation_state": primary_state,
        "normally_important": normally_important,
        "currently_active": currently_active,
        "emerging_importance": emerging,
        "fading_importance": fading,
        "proven_relevance": _clip01(proven_relevance),
        "current_activation": _clip01(current_activation),
        "previous_30d_activation": previous_30d,
        "previous_90d_activation": previous_90d,
        "change_30d": change_30d,
        "change_90d": change_90d,
        "data_support": data_support,
        "source_count": source_count,
        "sources": sources,
        "evidence": evidence,
        "interpretation": _interpretation(primary_state, model_role),
    }


def _combined_data_support(values: pd.Series) -> str:
    supports = {str(value) for value in values.dropna() if str(value)}
    if not supports:
        return "unknown"
    if supports == {"unsupported_watchlist"}:
        return "unsupported_watchlist"
    if "unsupported_watchlist" in supports and len(supports) > 1:
        return "mixed_with_unsupported_watchlist"
    if supports <= {"thin_proxy"}:
        return "thin_proxy"
    if "macro_fred" in supports or "market_confirmation" in supports or "market_price_proxy" in supports:
        return "validated_market_or_macro_proxy"
    if "proxy" in supports or "news_triage" in supports:
        return "proxy_context"
    return "; ".join(sorted(supports))


def _model_role(
    driver: str,
    proven_relevance: float,
    data_support: str,
    normally_important: bool,
) -> str:
    if driver == "unsupported_watchlist" or data_support == "unsupported_watchlist":
        return "unsupported"
    if driver in _EXPLAINER_ONLY_DRIVERS:
        return "explainer_only"
    if driver in _CONTEXT_ONLY_DRIVERS:
        return "validated_context"
    if normally_important and "validated" in data_support:
        return "allocation_driver"
    if proven_relevance >= DEFAULT_DRIVER_ROTATION_PROVEN_THRESHOLD and data_support != "thin_proxy":
        return "allocation_driver"
    if data_support in {"thin_proxy", "proxy_context", "mixed_with_unsupported_watchlist"}:
        return "explainer_only"
    return "validated_context"


def _primary_rotation_state(
    normally_important: bool,
    currently_active: bool,
    emerging: bool,
    fading: bool,
) -> str:
    if emerging:
        return "emerging_importance"
    if normally_important and currently_active:
        return "normally_important_active"
    if currently_active:
        return "currently_active"
    if fading:
        return "fading_importance"
    if normally_important:
        return "normally_important_quiet"
    return "quiet"


def _interpretation(primary_state: str, model_role: str) -> str:
    role_text = {
        "allocation_driver": "can affect allocation through tested model/risk layers",
        "validated_context": "useful context but not a direct sizing rule",
        "explainer_only": "explanatory only until ablation evidence promotes it",
        "unsupported": "unsupported watchlist item; do not trade from this signal",
    }.get(model_role, "context only")
    state_text = {
        "emerging_importance": "Unusually active versus recent proxy history.",
        "normally_important_active": "Historically important and active now.",
        "currently_active": "Currently firing, but not a top historical driver.",
        "fading_importance": "Previously active but fading from the current read.",
        "normally_important_quiet": "Historically important, but quiet today.",
        "quiet": "Not currently prominent.",
    }.get(primary_state, "Current read is mixed.")
    return f"{state_text} Trading role: {role_text}."


def _driver_label(driver: str) -> str:
    labels = {
        "ai_capex": "AI capex pressure",
        "ai_leadership": "AI / tech leadership",
        "breadth": "Market breadth",
        "commodities": "Commodities / inflation",
        "concentration": "Index concentration",
        "credit": "Credit conditions",
        "dollar_liquidity": "Dollar / liquidity",
        "drawdown": "Drawdown repair",
        "duration_rates": "Rates / duration",
        "equity_supply": "IPO / equity supply",
        "positioning": "Positioning / crowding",
        "private_credit": "Private credit",
        "regime_instability": "Regime instability",
        "trend": "Broad trend",
        "unsupported_watchlist": "Unsupported data watchlist",
        "volatility": "Volatility",
    }
    return labels.get(driver, driver.replace("_", " ").title())


def _as_float(value: object, *, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _nan_if_missing(value: object) -> float:
    result = _as_float(value, default=np.nan)
    return result if np.isfinite(result) else np.nan


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _max_change(*values: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return max(finite) if finite else 0.0


def _min_change(*values: float) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return min(finite) if finite else 0.0


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))
