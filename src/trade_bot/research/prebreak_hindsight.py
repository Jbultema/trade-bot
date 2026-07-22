from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trade_bot.research.cycle_tracker import build_cycle_feature_snapshot
from trade_bot.storage.run_store import RunStore


@dataclass(frozen=True)
class BubbleBreakWindow:
    name: str
    break_date: str
    family: str
    description: str


DEFAULT_PREBREAK_LOOKBACK_DAYS = 365
DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS = 31
DEFAULT_PREBREAK_WEEKLY_FREQUENCY = "W-WED"
DEFAULT_PREBREAK_HORIZON_DAYS = 63
DEFAULT_PREBREAK_OUTPUT_DIR = Path("reports/prebreak_hindsight")
DEFAULT_PREBREAK_RUN_STORE_DB_PATH = Path("data/prebreak_hindsight/trade_bot.duckdb")
DEFAULT_PREBREAK_RUN_STORE_ARTIFACT_DIR = Path("data/prebreak_hindsight/snapshots")
DEFAULT_PREBREAK_RUN_STORE_JOB_LOG_DIR = Path("data/prebreak_hindsight/jobs")
DEFAULT_BUBBLE_BREAK_WINDOWS: tuple[BubbleBreakWindow, ...] = (
    BubbleBreakWindow(
        name="gfc_credit_bubble_peak",
        break_date="2007-10-09",
        family="credit_bubble",
        description="S&P 500 peak before the global financial crisis bear market.",
    ),
    BubbleBreakWindow(
        name="volmageddon_late_cycle_break",
        break_date="2018-01-26",
        family="volatility_short_vol",
        description="Equity peak before the February 2018 volatility shock.",
    ),
    BubbleBreakWindow(
        name="q4_2018_liquidity_break",
        break_date="2018-09-20",
        family="liquidity_policy",
        description="Peak before the Q4 2018 Fed/liquidity drawdown.",
    ),
    BubbleBreakWindow(
        name="covid_crash_peak",
        break_date="2020-02-19",
        family="exogenous_shock",
        description="Peak before the COVID liquidity and volatility crash.",
    ),
    BubbleBreakWindow(
        name="ark_growth_bubble_peak",
        break_date="2021-02-12",
        family="speculative_growth",
        description="Peak in high-growth/speculative technology proxies before 2021-2022 unwind.",
    ),
    BubbleBreakWindow(
        name="inflation_rates_growth_peak",
        break_date="2021-11-19",
        family="inflation_rates",
        description="Nasdaq/growth peak before the 2022 inflation and rates bear market.",
    ),
    BubbleBreakWindow(
        name="yen_carry_ai_vol_break",
        break_date="2024-07-16",
        family="ai_carry_volatility",
        description="AI/carry-trade stress window before the August 2024 volatility shock.",
    ),
    BubbleBreakWindow(
        name="tariff_liquidity_growth_break",
        break_date="2025-02-19",
        family="policy_liquidity",
        description="Growth/AI stress window before the 2025 tariff and liquidity drawdown.",
    ),
)

ACTION_DEFENSIVE_STATUSES = {"orange", "red"}
ACTION_REDUCE_TOKENS = ("REDUCE", "DE_RISK", "DEFENSIVE")
STAGED_RISK_TARGETS: dict[str, float] = {
    "long_lead_context": 1.00,
    "early_watch": 0.75,
    "warning": 0.60,
    "confirmed_prebreak": 0.35,
    "break_unwind": 0.20,
    "postbreak_followthrough": 0.20,
}
STAGED_RISK_ORDER: dict[str, int] = {
    "long_lead_context": 0,
    "early_watch": 1,
    "warning": 2,
    "confirmed_prebreak": 3,
    "break_unwind": 4,
    "postbreak_followthrough": 5,
}
LATE_TRIGGER_DAYS: tuple[int, ...] = (15, 21, 30, 45)


@dataclass(frozen=True)
class PrebreakHindsightResult:
    snapshot_signals: pd.DataFrame
    signal_rankings: pd.DataFrame
    action_timing: pd.DataFrame
    staged_risk_behavior: pd.DataFrame
    late_trigger_mesh: pd.DataFrame
    hard_defense_attribution: pd.DataFrame
    policy_variant_results: pd.DataFrame
    current_signal_readout: pd.DataFrame
    summary: str


def build_prebreak_snapshot_plan(
    available_dates: pd.DatetimeIndex | list[object] | tuple[object, ...],
    *,
    windows: tuple[BubbleBreakWindow, ...] = DEFAULT_BUBBLE_BREAK_WINDOWS,
    lookback_days: int = DEFAULT_PREBREAK_LOOKBACK_DAYS,
    postbreak_days: int = DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS,
    weekly_frequency: str = DEFAULT_PREBREAK_WEEKLY_FREQUENCY,
    include_break_date: bool = True,
) -> pd.DataFrame:
    if lookback_days <= 0:
        msg = "lookback_days must be positive"
        raise ValueError(msg)
    if postbreak_days < 0:
        msg = "postbreak_days cannot be negative"
        raise ValueError(msg)
    dates = _normalize_dates(available_dates)
    if dates.empty:
        msg = "No available market dates were supplied."
        raise ValueError(msg)

    rows: list[dict[str, object]] = []
    for window in windows:
        break_date = pd.Timestamp(window.break_date).normalize()
        start_date = break_date - pd.Timedelta(days=lookback_days)
        end_date = break_date + pd.Timedelta(days=postbreak_days)
        mask = (dates >= start_date) & (dates <= end_date)
        if not include_break_date:
            mask = mask & (dates < break_date)
        event_dates = dates[mask]
        if event_dates.empty:
            continue
        weekly_dates = _latest_date_per_week(event_dates, weekly_frequency=weekly_frequency)
        if include_break_date:
            break_sessions = event_dates[event_dates <= break_date]
            if not break_sessions.empty:
                weekly_dates = pd.DatetimeIndex(
                    sorted(set(weekly_dates) | {break_sessions.max()})
                )
        for market_date in weekly_dates:
            rows.append(
                {
                    "event_name": window.name,
                    "family": window.family,
                    "description": window.description,
                    "break_date": str(break_date.date()),
                    "window_start_date": str(start_date.date()),
                    "window_end_date": str(end_date.date()),
                    "market_date": str(market_date.date()),
                    "days_to_break": int((break_date - market_date).days),
                    "postbreak_snapshot": bool(market_date > break_date),
                    "weekly_frequency": weekly_frequency,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "event_name",
                "family",
                "description",
                "break_date",
                "window_start_date",
                "window_end_date",
                "market_date",
                "days_to_break",
                "postbreak_snapshot",
                "weekly_frequency",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["break_date", "market_date", "event_name"])
        .reset_index(drop=True)
    )


def deduplicate_prebreak_snapshot_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Collapse overlapping event windows to one persisted snapshot per market date."""
    if plan.empty or "market_date" not in plan:
        return plan.copy()
    rows: list[dict[str, object]] = []
    for market_date, group in plan.groupby("market_date", sort=True, observed=True):
        ordered = group.sort_values(["break_date", "event_name"]).copy()
        canonical = ordered.iloc[-1].to_dict()
        canonical["market_date"] = str(market_date)
        canonical["event_name"] = " | ".join(dict.fromkeys(ordered["event_name"].astype(str)))
        canonical["family"] = " | ".join(dict.fromkeys(ordered["family"].astype(str)))
        canonical["break_date"] = " | ".join(dict.fromkeys(ordered["break_date"].astype(str)))
        canonical["days_to_break"] = " | ".join(dict.fromkeys(ordered["days_to_break"].astype(str)))
        canonical["postbreak_snapshot"] = bool(ordered["postbreak_snapshot"].any())
        canonical["event_count"] = int(len(ordered))
        rows.append(canonical)
    return pd.DataFrame(rows).sort_values("market_date").reset_index(drop=True)


def analyze_prebreak_hindsight(
    run_store: RunStore,
    *,
    reference_run_store: RunStore | None = None,
    reference_prices: pd.DataFrame | None = None,
    include_reference_snapshots: bool = True,
    include_current_snapshot: bool = True,
    windows: tuple[BubbleBreakWindow, ...] = DEFAULT_BUBBLE_BREAK_WINDOWS,
    lookback_days: int = DEFAULT_PREBREAK_LOOKBACK_DAYS,
    postbreak_days: int = DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS,
    horizon_days: int = DEFAULT_PREBREAK_HORIZON_DAYS,
    severe_drawdown_threshold: float = -0.10,
    major_drawdown_threshold: float = -0.15,
    snapshot_limit: int = 100_000,
) -> PrebreakHindsightResult:
    source_store = reference_run_store or run_store
    if reference_prices is None:
        snapshot_payload = source_store.load_latest_snapshot(require_matching_config=False)
        if snapshot_payload is None:
            msg = "No snapshots are available to provide reference prices."
            raise ValueError(msg)
        reference_prices = getattr(snapshot_payload[0], "prices", pd.DataFrame())
    reference_prices = _clean_prices(reference_prices)
    if reference_prices.empty:
        msg = "Reference prices are empty."
        raise ValueError(msg)

    signal_rows = _snapshot_signal_rows(
        run_store,
        reference_prices=reference_prices,
        windows=windows,
        lookback_days=lookback_days,
        postbreak_days=postbreak_days,
        horizon_days=horizon_days,
        severe_drawdown_threshold=severe_drawdown_threshold,
        major_drawdown_threshold=major_drawdown_threshold,
        snapshot_limit=snapshot_limit,
        source_label="prebreak_experiment",
    )
    if include_reference_snapshots and reference_run_store is not None:
        signal_rows.extend(
            _snapshot_signal_rows(
                reference_run_store,
                reference_prices=reference_prices,
                windows=windows,
                lookback_days=lookback_days,
                postbreak_days=postbreak_days,
                horizon_days=horizon_days,
                severe_drawdown_threshold=severe_drawdown_threshold,
                major_drawdown_threshold=major_drawdown_threshold,
                snapshot_limit=snapshot_limit,
                source_label="reference_control",
            )
        )
    snapshot_signals = pd.DataFrame(signal_rows)
    if snapshot_signals.empty:
        return PrebreakHindsightResult(
            snapshot_signals=snapshot_signals,
            signal_rankings=pd.DataFrame(),
            action_timing=pd.DataFrame(),
            staged_risk_behavior=pd.DataFrame(),
            late_trigger_mesh=pd.DataFrame(),
            hard_defense_attribution=pd.DataFrame(),
            policy_variant_results=pd.DataFrame(),
            current_signal_readout=pd.DataFrame(),
            summary="No usable snapshots were available for pre-break hindsight analysis.",
        )
    snapshot_signals = _dedupe_snapshot_signals(snapshot_signals)
    signal_rankings = rank_predictive_signals(snapshot_signals)
    action_timing = summarize_action_timing(snapshot_signals)
    staged_risk_behavior = summarize_staged_risk_behavior(snapshot_signals)
    late_trigger_mesh = build_late_trigger_mesh(snapshot_signals)
    hard_defense_attribution = summarize_hard_defense_attribution(snapshot_signals)
    policy_variant_results = evaluate_staged_policy_variants(snapshot_signals)
    current_source = snapshot_signals
    if include_current_snapshot and reference_run_store is not None:
        current_payload = reference_run_store.load_latest_snapshot(require_matching_config=False)
        if current_payload is not None:
            current_run, current_manifest = current_payload
            current_row = snapshot_signal_row(
                current_run,
                run_id=current_manifest.run_id,
                created_at_utc=current_manifest.created_at_utc,
                reference_prices=reference_prices,
                windows=windows,
                lookback_days=lookback_days,
                postbreak_days=postbreak_days,
                horizon_days=horizon_days,
                severe_drawdown_threshold=severe_drawdown_threshold,
                major_drawdown_threshold=major_drawdown_threshold,
            )
            if current_row:
                current_row["snapshot_source"] = "current_reference"
                current_source = pd.concat(
                    [snapshot_signals, pd.DataFrame([current_row])],
                    ignore_index=True,
                )
    current_signal_readout = current_best_signal_readout(current_source, signal_rankings)
    summary = build_prebreak_hindsight_summary(
        snapshot_signals,
        signal_rankings,
        action_timing,
        staged_risk_behavior,
        late_trigger_mesh,
        hard_defense_attribution,
        policy_variant_results,
        current_signal_readout,
    )
    return PrebreakHindsightResult(
        snapshot_signals=snapshot_signals,
        signal_rankings=signal_rankings,
        action_timing=action_timing,
        staged_risk_behavior=staged_risk_behavior,
        late_trigger_mesh=late_trigger_mesh,
        hard_defense_attribution=hard_defense_attribution,
        policy_variant_results=policy_variant_results,
        current_signal_readout=current_signal_readout,
        summary=summary,
    )


def _snapshot_signal_rows(
    run_store: RunStore,
    *,
    reference_prices: pd.DataFrame,
    windows: tuple[BubbleBreakWindow, ...],
    lookback_days: int,
    postbreak_days: int,
    horizon_days: int,
    severe_drawdown_threshold: float,
    major_drawdown_threshold: float,
    snapshot_limit: int,
    source_label: str,
) -> list[dict[str, object]]:
    manifests = run_store.list_snapshots(limit=snapshot_limit)
    rows: list[dict[str, object]] = []
    for _, manifest_row in manifests.iterrows():
        run_id = str(manifest_row.get("run_id", ""))
        try:
            run, manifest = run_store.load_snapshot(run_id)
        except (FileNotFoundError, TypeError, EOFError, OSError):
            continue
        signal_row = snapshot_signal_row(
            run,
            run_id=manifest.run_id,
            created_at_utc=manifest.created_at_utc,
            reference_prices=reference_prices,
            windows=windows,
            lookback_days=lookback_days,
            postbreak_days=postbreak_days,
            horizon_days=horizon_days,
            severe_drawdown_threshold=severe_drawdown_threshold,
            major_drawdown_threshold=major_drawdown_threshold,
        )
        if signal_row:
            signal_row["snapshot_source"] = source_label
            rows.append(signal_row)
    return rows


def _dedupe_snapshot_signals(snapshot_signals: pd.DataFrame) -> pd.DataFrame:
    if snapshot_signals.empty:
        return snapshot_signals
    frame = snapshot_signals.copy()
    event_name = (
        frame["event_name"].fillna("").astype(str)
        if "event_name" in frame
        else pd.Series("", index=frame.index)
    )
    source = (
        frame["snapshot_source"].fillna("").astype(str)
        if "snapshot_source" in frame
        else pd.Series("", index=frame.index)
    )
    in_event_window = event_name.ne("")
    frame = frame.loc[
        ~(
            (source.eq("reference_control") & in_event_window)
            | (source.eq("prebreak_experiment") & ~in_event_window)
        )
    ].copy()
    sort_columns = [
        column
        for column in ["snapshot_source", "event_name", "market_date", "created_at_utc", "run_id"]
        if column in frame
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    dedupe_columns = [
        column for column in ["snapshot_source", "event_name", "market_date"] if column in frame
    ]
    if dedupe_columns:
        frame = frame.drop_duplicates(dedupe_columns, keep="last")
    if "run_id" in frame:
        frame = frame.drop_duplicates("run_id", keep="last")
    final_sort_columns = [
        column for column in ["market_date", "snapshot_source", "event_name"] if column in frame
    ]
    if final_sort_columns:
        frame = frame.sort_values(final_sort_columns)
    return frame.reset_index(drop=True)


def snapshot_signal_row(
    run: Any,
    *,
    run_id: str,
    created_at_utc: str,
    reference_prices: pd.DataFrame,
    windows: tuple[BubbleBreakWindow, ...] = DEFAULT_BUBBLE_BREAK_WINDOWS,
    lookback_days: int = DEFAULT_PREBREAK_LOOKBACK_DAYS,
    postbreak_days: int = DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS,
    horizon_days: int = DEFAULT_PREBREAK_HORIZON_DAYS,
    severe_drawdown_threshold: float = -0.10,
    major_drawdown_threshold: float = -0.15,
) -> dict[str, object]:
    current_state = getattr(run, "current_state", None)
    prices = _clean_prices(getattr(run, "prices", pd.DataFrame()))
    if current_state is None or prices.empty:
        return {}
    market_date = str(getattr(current_state, "market_date", ""))
    if not market_date:
        return {}

    row: dict[str, object] = {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "market_date": market_date,
        "risk_status": str(getattr(current_state, "risk_status", "")),
        "risk_score": _safe_float(getattr(current_state, "risk_score", np.nan)),
        "risk_status_score": _risk_status_score(str(getattr(current_state, "risk_status", ""))),
        "risk_timing_state": str(getattr(current_state, "risk_timing_state", "unassessed")),
        "risk_timing_raw_multiplier": _safe_float(
            getattr(current_state, "risk_timing_multiplier", np.nan)
        ),
        "risk_timing_break_count": len(getattr(current_state, "risk_timing_breaks", ())),
        "risk_timing_breaks": ", ".join(getattr(current_state, "risk_timing_breaks", ())),
        "risk_timing_recovery_count": len(
            getattr(current_state, "risk_timing_recoveries", ())
        ),
    }
    trade_summary = _first_row(
        getattr(getattr(run, "trade_decision", None), "summary", pd.DataFrame())
    )
    row["recommended_action"] = str(trade_summary.get("recommended_action", ""))
    row["risk_budget_multiplier"] = _safe_float(trade_summary.get("risk_budget_multiplier", np.nan))
    row["risk_timing_effective_multiplier"] = _safe_float(
        trade_summary.get("risk_timing_multiplier", np.nan)
    )
    row["risk_timing_sizing_authority"] = _safe_float(
        trade_summary.get("risk_timing_sizing_authority", np.nan)
    )
    row["pre_sanity_risk_budget_multiplier"] = _safe_float(
        trade_summary.get("pre_sanity_risk_budget_multiplier", np.nan)
    )
    row["one_month_risk_off_probability"] = _safe_float(
        trade_summary.get("one_month_risk_off_probability", np.nan)
    )
    row["one_month_transition_probability"] = _safe_float(
        trade_summary.get("one_month_transition_probability", np.nan)
    )
    row["one_month_fragile_upside_probability"] = _safe_float(
        trade_summary.get("one_month_fragile_upside_probability", np.nan)
    )
    row["event_pressure"] = _safe_float(trade_summary.get("event_pressure", np.nan))
    row["macro_pressure"] = _safe_float(trade_summary.get("macro_pressure", np.nan))
    row["decision_sanity_break_count"] = _safe_float(
        trade_summary.get("market_confirmation_break_count", np.nan)
    )
    row["scenario_event_macro_multiplier"] = _safe_float(
        trade_summary.get("scenario_event_macro_multiplier", np.nan)
    )
    row["portfolio_risk_multiplier"] = _safe_float(
        trade_summary.get("portfolio_risk_multiplier", np.nan)
    )
    row["decision_sanity_cap_applied"] = bool(
        trade_summary.get("decision_sanity_cap_applied", False)
    )
    row["current_risk_asset_weight"] = _safe_float(
        trade_summary.get("current_risk_asset_weight", np.nan)
    )
    row["target_risk_asset_weight"] = _safe_float(
        trade_summary.get("target_risk_asset_weight", np.nan)
    )
    row["target_defensive_weight"] = _safe_float(
        trade_summary.get("target_defensive_weight", np.nan)
    )
    row["opportunity_pressure"] = _safe_float(trade_summary.get("opportunity_pressure", np.nan))
    row["decision_sanity_status"] = str(trade_summary.get("decision_sanity_status", ""))
    row["posture_calibration_status"] = str(trade_summary.get("posture_calibration_status", ""))
    row["portfolio_expected_shortfall_95"] = _safe_float(
        trade_summary.get("portfolio_expected_shortfall_95", np.nan)
    )
    row["portfolio_max_stress_loss"] = _safe_float(
        trade_summary.get("portfolio_max_stress_loss", np.nan)
    )
    row["portfolio_equity_beta"] = _safe_float(trade_summary.get("portfolio_equity_beta", np.nan))
    row["portfolio_ai_beta"] = _safe_float(trade_summary.get("portfolio_ai_beta", np.nan))
    row["portfolio_constraints"] = str(trade_summary.get("portfolio_constraints", ""))
    _add_trade_attribution_metrics(
        row,
        getattr(getattr(run, "trade_decision", None), "attribution", pd.DataFrame()),
    )
    row["defensive_action_flag"] = _defensive_action_flag(row)
    row["hard_defensive_action_flag"] = _hard_defensive_action_flag(row)
    row["action_severity_score"] = _action_severity_score(row)

    _add_scenario_driver_metrics(row, getattr(current_state, "scenario_drivers", pd.DataFrame()))
    _add_confirmation_metrics(row, getattr(current_state, "confirmation_matrix", pd.DataFrame()))
    _add_market_health_metrics(row, getattr(current_state, "market_health", pd.DataFrame()))
    _add_regime_instability_metrics(
        row,
        getattr(current_state, "regime_instability", pd.DataFrame()),
        getattr(current_state, "regime_instability_components", pd.DataFrame()),
    )
    _add_cycle_metrics(row, prices)
    _add_forward_outcome_metrics(
        row,
        reference_prices=reference_prices,
        market_date=market_date,
        horizon_days=horizon_days,
        severe_drawdown_threshold=severe_drawdown_threshold,
        major_drawdown_threshold=major_drawdown_threshold,
    )
    _add_event_membership(
        row,
        windows=windows,
        lookback_days=lookback_days,
        postbreak_days=postbreak_days,
    )
    _add_staged_risk_metadata(row)
    row["hindsight_action_aligned"] = bool(
        _truthy_label(row.get("forward_break_label_3m")) and row.get("defensive_action_flag", False)
    )
    return row


def rank_predictive_signals(
    snapshot_signals: pd.DataFrame,
    *,
    target_column: str = "break_severity_3m",
    label_column: str = "forward_break_label_3m",
    min_observations: int = 12,
) -> pd.DataFrame:
    if snapshot_signals.empty or target_column not in snapshot_signals:
        return pd.DataFrame()
    metadata_columns = {
        "run_id",
        "created_at_utc",
        "market_date",
        "snapshot_source",
        "event_name",
        "event_family",
        "event_break_date",
        "event_description",
        "days_to_break",
        "postbreak_snapshot",
        "prebreak_stage",
        "prebreak_stage_order",
        "target_staged_risk_budget_multiplier",
        "risk_budget_gap_to_stage_target",
        "over_defensive_gap_to_stage_target",
        "under_defensive_gap_to_stage_target",
        "early_hard_false_alarm_flag",
        "risk_status",
        "recommended_action",
        "hindsight_action_aligned",
    }
    outcome_columns = {
        target_column,
        label_column,
        "forward_major_break_label_3m",
        "forward_spy_return_3m",
        "forward_qqq_return_3m",
        "forward_smh_return_3m",
        "forward_upside_proxy_3m",
        "forward_spy_max_drawdown_3m",
        "forward_qqq_max_drawdown_3m",
        "forward_smh_max_drawdown_3m",
        "forward_min_max_drawdown_3m",
    }
    candidates = [
        column
        for column in snapshot_signals.columns
        if column not in metadata_columns
        and column not in outcome_columns
        and pd.api.types.is_numeric_dtype(snapshot_signals[column])
    ]
    rows: list[dict[str, object]] = []
    target = pd.to_numeric(snapshot_signals[target_column], errors="coerce")
    label = (
        pd.to_numeric(snapshot_signals[label_column], errors="coerce")
        if label_column in snapshot_signals
        else None
    )
    for column in candidates:
        signal = pd.to_numeric(snapshot_signals[column], errors="coerce").astype(float)
        frame = pd.DataFrame({"signal": signal, "target": target}).dropna()
        if len(frame) < min_observations or frame["signal"].nunique() < 2:
            continue
        spearman = float(frame["signal"].corr(frame["target"], method="spearman"))
        if pd.isna(spearman):
            continue
        lower_quantile = frame["signal"].quantile(0.25)
        upper_quantile = frame["signal"].quantile(0.75)
        low_target = frame.loc[frame["signal"] <= lower_quantile, "target"]
        high_target = frame.loc[frame["signal"] >= upper_quantile, "target"]
        high_low_spread = (
            float(high_target.mean() - low_target.mean())
            if not low_target.empty and not high_target.empty
            else np.nan
        )
        auc = np.nan
        if label is not None:
            aligned = pd.DataFrame({"signal": signal, "label": label}).dropna()
            aligned_label = aligned["label"].astype(bool)
            positives = aligned.loc[aligned_label, "signal"]
            negatives = aligned.loc[~aligned_label, "signal"]
            auc = _directional_auc(positives, negatives)
        auc_edge = abs(auc - 0.5) * 2.0 if pd.notna(auc) else 0.0
        spread_score = min(abs(_safe_float(high_low_spread, 0.0)) / 0.12, 1.0)
        predictive_score = 0.45 * abs(spearman) + 0.40 * auc_edge + 0.15 * spread_score
        rows.append(
            {
                "signal": column,
                "observations": len(frame),
                "spearman_to_break_severity": spearman,
                "absolute_spearman": abs(spearman),
                "event_auc": auc,
                "event_auc_edge": auc_edge,
                "high_minus_low_break_severity": high_low_spread,
                "predictive_score": predictive_score,
                "risk_direction": "higher_is_riskier" if spearman >= 0 else "lower_is_riskier",
                "latest_value": _latest_signal_value(snapshot_signals, column),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["predictive_score", "absolute_spearman", "event_auc_edge"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def summarize_action_timing(snapshot_signals: pd.DataFrame) -> pd.DataFrame:
    if snapshot_signals.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    event_rows = snapshot_signals.dropna(subset=["event_name", "days_to_break"])
    if not event_rows.empty:
        for event_name, group in event_rows.groupby("event_name", dropna=True):
            ordered = group.sort_values("days_to_break", ascending=False)
            defensive = ordered[ordered["defensive_action_flag"].astype(bool)]
            hard_defensive = ordered[_bool_flag_series(ordered, "hard_defensive_action_flag")]
            severe = ordered[_bool_label_series(ordered["forward_break_label_3m"])]
            rows.append(
                {
                    "event_name": str(event_name),
                    "break_date": str(ordered["event_break_date"].dropna().iloc[0]),
                    "snapshots": len(ordered),
                    "first_defensive_market_date": (
                        str(defensive.iloc[0]["market_date"]) if not defensive.empty else ""
                    ),
                    "first_defensive_days_before_break": (
                        int(defensive.iloc[0]["days_to_break"]) if not defensive.empty else np.nan
                    ),
                    "first_hard_defensive_market_date": (
                        str(hard_defensive.iloc[0]["market_date"])
                        if not hard_defensive.empty
                        else ""
                    ),
                    "first_hard_defensive_days_before_break": (
                        int(hard_defensive.iloc[0]["days_to_break"])
                        if not hard_defensive.empty
                        else np.nan
                    ),
                    "first_severe_label_market_date": (
                        str(severe.iloc[0]["market_date"]) if not severe.empty else ""
                    ),
                    "first_severe_label_days_before_break": (
                        int(severe.iloc[0]["days_to_break"]) if not severe.empty else np.nan
                    ),
                    "defensive_snapshot_share": float(ordered["defensive_action_flag"].mean()),
                    "hard_defensive_snapshot_share": float(
                        _bool_flag_series(ordered, "hard_defensive_action_flag").mean()
                    ),
                    "severe_label_share": float(
                        pd.to_numeric(ordered["forward_break_label_3m"], errors="coerce").mean()
                    ),
                    "aligned_when_severe_share": _aligned_when_severe_share(ordered),
                    "hard_aligned_when_severe_share": _aligned_when_severe_share(
                        ordered,
                        action_column="hard_defensive_action_flag",
                    ),
                    "median_risk_budget_multiplier": _safe_float(
                        pd.to_numeric(ordered["risk_budget_multiplier"], errors="coerce").median()
                    ),
                }
            )
    severe_rows = snapshot_signals[
        _bool_label_series(snapshot_signals["forward_break_label_3m"])
    ].copy()
    if not severe_rows.empty:
        severe_rows["days_to_break_bucket"] = pd.cut(
            pd.to_numeric(severe_rows["days_to_break"], errors="coerce"),
            bins=[-np.inf, -1, 21, 42, 63, 92, np.inf],
            labels=[
                "postbreak",
                "0-21d",
                "22-42d",
                "43-63d",
                "64-92d",
                "outside_event_window",
            ],
        ).astype(str)
        for bucket, group in severe_rows.groupby("days_to_break_bucket", dropna=False):
            rows.append(
                {
                    "event_name": f"ALL_SEVERE_{bucket}",
                    "break_date": "",
                    "snapshots": len(group),
                    "first_defensive_market_date": "",
                    "first_defensive_days_before_break": np.nan,
                    "first_hard_defensive_market_date": "",
                    "first_hard_defensive_days_before_break": np.nan,
                    "first_severe_label_market_date": "",
                    "first_severe_label_days_before_break": np.nan,
                    "defensive_snapshot_share": float(group["defensive_action_flag"].mean()),
                    "hard_defensive_snapshot_share": float(
                        _bool_flag_series(group, "hard_defensive_action_flag").mean()
                    ),
                    "severe_label_share": 1.0,
                    "aligned_when_severe_share": float(
                        group["hindsight_action_aligned"].astype(bool).mean()
                    ),
                    "hard_aligned_when_severe_share": _aligned_when_severe_share(
                        group,
                        action_column="hard_defensive_action_flag",
                    ),
                    "median_risk_budget_multiplier": _safe_float(
                        pd.to_numeric(group["risk_budget_multiplier"], errors="coerce").median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_staged_risk_behavior(snapshot_signals: pd.DataFrame) -> pd.DataFrame:
    if snapshot_signals.empty or "prebreak_stage" not in snapshot_signals:
        return pd.DataFrame()
    event_rows = snapshot_signals[
        snapshot_signals.get("event_name", pd.Series("", index=snapshot_signals.index))
        .fillna("")
        .astype(str)
        .ne("")
    ].copy()
    if event_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped = event_rows.groupby(["event_name", "prebreak_stage"], dropna=True, sort=False)
    for (event_name, stage), group in grouped:
        stage_text = str(stage)
        if not stage_text:
            continue
        rows.append(_staged_risk_summary_row(str(event_name), stage_text, group))
    for stage, group in event_rows.groupby("prebreak_stage", dropna=True, sort=False):
        stage_text = str(stage)
        if not stage_text:
            continue
        rows.append(_staged_risk_summary_row(f"ALL_EVENTS_{stage_text}", stage_text, group))
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["event_name", "prebreak_stage_order"])
        .reset_index(drop=True)
    )


def build_late_trigger_mesh(
    snapshot_signals: pd.DataFrame,
    *,
    trigger_days: tuple[int, ...] = LATE_TRIGGER_DAYS,
) -> pd.DataFrame:
    if snapshot_signals.empty or "event_name" not in snapshot_signals:
        return pd.DataFrame()
    event_rows = snapshot_signals[
        snapshot_signals["event_name"].fillna("").astype(str).ne("")
    ].copy()
    if event_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for event_name, group in event_rows.groupby("event_name", dropna=True, sort=False):
        group = group.copy()
        days = pd.to_numeric(group["days_to_break"], errors="coerce")
        prebreak = group[days >= 0].copy()
        if prebreak.empty:
            continue
        prebreak_days = pd.to_numeric(prebreak["days_to_break"], errors="coerce")
        hard = _bool_flag_series(prebreak, "hard_defensive_action_flag")
        severe = _bool_label_series(prebreak["forward_break_label_3m"])
        first_hard_days = (
            int(prebreak.loc[hard, "days_to_break"].max()) if bool(hard.any()) else np.nan
        )
        severe_total = int(severe.sum())
        for trigger in trigger_days:
            before_trigger = prebreak_days > trigger
            inside_trigger = prebreak_days <= trigger
            lifted = before_trigger & hard
            severe_before_trigger = before_trigger & severe
            severe_inside_trigger = inside_trigger & severe
            false_alarm = lifted & _false_alarm_mask(prebreak)
            rows.append(
                {
                    "event_name": str(event_name),
                    "trigger_days_before_break": int(trigger),
                    "snapshots": len(prebreak),
                    "pre_trigger_snapshots": int(before_trigger.sum()),
                    "inside_trigger_snapshots": int(inside_trigger.sum()),
                    "actual_first_hard_defensive_days_before_break": first_hard_days,
                    "hard_defense_lead_cut_days": (
                        max(0, int(first_hard_days) - int(trigger))
                        if pd.notna(first_hard_days)
                        else np.nan
                    ),
                    "pre_trigger_hard_defensive_share": _masked_mean(hard, before_trigger),
                    "pre_trigger_false_alarm_share": _masked_mean(false_alarm, before_trigger),
                    "severe_label_coverage_inside_trigger": (
                        float(severe_inside_trigger.sum() / severe_total)
                        if severe_total
                        else np.nan
                    ),
                    "missed_severe_label_share_if_gated": (
                        float(severe_before_trigger.sum() / severe_total)
                        if severe_total
                        else np.nan
                    ),
                    "median_pre_trigger_risk_budget_multiplier": _masked_median(
                        prebreak,
                        "risk_budget_multiplier",
                        before_trigger,
                    ),
                    "mean_candidate_risk_budget_lift": _candidate_lift_mean(
                        prebreak,
                        before_trigger,
                    ),
                    "median_forward_return_when_lifted": _masked_median(
                        _with_forward_upside_proxy(prebreak),
                        "forward_upside_proxy_3m",
                        lifted,
                    ),
                    "median_forward_drawdown_when_lifted": _masked_median(
                        prebreak,
                        "forward_min_max_drawdown_3m",
                        lifted,
                    ),
                    "mesh_read": _late_trigger_read(
                        severe_total=severe_total,
                        missed_severe_share=(
                            float(severe_before_trigger.sum() / severe_total)
                            if severe_total
                            else np.nan
                        ),
                        candidate_lift=_candidate_lift_mean(prebreak, before_trigger),
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["event_name", "trigger_days_before_break"])
        .reset_index(drop=True)
    )


def summarize_hard_defense_attribution(snapshot_signals: pd.DataFrame) -> pd.DataFrame:
    if snapshot_signals.empty or "event_name" not in snapshot_signals:
        return pd.DataFrame()
    event_rows = snapshot_signals[
        snapshot_signals["event_name"].fillna("").astype(str).ne("")
    ].copy()
    if event_rows.empty:
        return pd.DataFrame()
    event_rows["hard_defense_source"] = event_rows.apply(_hard_defense_source, axis=1)
    rows: list[dict[str, object]] = []
    for event_name, group in event_rows.groupby("event_name", dropna=True, sort=False):
        rows.extend(_hard_defense_attribution_rows(str(event_name), group))
    rows.extend(_hard_defense_attribution_rows("ALL_EVENTS", event_rows))
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["event_name", "prebreak_stage_order", "hard_defense_source"])
        .reset_index(drop=True)
    )


def evaluate_staged_policy_variants(snapshot_signals: pd.DataFrame) -> pd.DataFrame:
    if snapshot_signals.empty or "event_name" not in snapshot_signals:
        return pd.DataFrame()
    event_rows = snapshot_signals[
        snapshot_signals["event_name"].fillna("").astype(str).ne("")
    ].copy()
    if event_rows.empty:
        return pd.DataFrame()
    policies = {
        "actual": _policy_actual_budget,
        "stage_floor": _policy_stage_floor_budget,
        "stage_floor_confirm_45d": lambda row: _policy_confirm_gate_budget(row, 45),
        "stage_floor_confirm_30d": lambda row: _policy_confirm_gate_budget(row, 30),
        "stage_floor_confirm_21d": lambda row: _policy_confirm_gate_budget(row, 21),
        "stage_floor_confirm_15d": lambda row: _policy_confirm_gate_budget(row, 15),
        "watch_warning_floor": _policy_watch_warning_floor_budget,
        "portfolio_watch_floor": _policy_portfolio_watch_floor_budget,
        "portfolio_confirm_45d": lambda row: _policy_portfolio_confirm_budget(row, 45),
        "portfolio_confirm_30d": lambda row: _policy_portfolio_confirm_budget(row, 30),
        "portfolio_confirm_30d_moderate": lambda row: _policy_portfolio_confirm_budget(
            row,
            30,
            floors={
                "long_lead_context": 0.75,
                "early_watch": 0.70,
                "warning": 0.55,
            },
        ),
        "portfolio_confirm_30d_conservative": lambda row: _policy_portfolio_confirm_budget(
            row,
            30,
            floors={
                "long_lead_context": 0.65,
                "early_watch": 0.60,
                "warning": 0.50,
            },
        ),
    }
    rows: list[dict[str, object]] = []
    for policy_name, policy in policies.items():
        rows.append(_policy_variant_summary_row("ALL_EVENTS", policy_name, event_rows, policy))
        for event_name, group in event_rows.groupby("event_name", dropna=True, sort=False):
            rows.append(_policy_variant_summary_row(str(event_name), policy_name, group, policy))
    return pd.DataFrame(rows).sort_values(["event_name", "policy_name"]).reset_index(drop=True)


def current_best_signal_readout(
    snapshot_signals: pd.DataFrame,
    signal_rankings: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    if snapshot_signals.empty or signal_rankings.empty:
        return pd.DataFrame()
    latest = snapshot_signals.sort_values("market_date").iloc[-1]
    rows: list[dict[str, object]] = []
    for _, ranking in signal_rankings.head(top_n).iterrows():
        signal = str(ranking["signal"])
        if signal not in snapshot_signals:
            continue
        series = pd.to_numeric(snapshot_signals[signal], errors="coerce").dropna()
        latest_value = _safe_float(latest.get(signal, np.nan))
        percentile = float((series <= latest_value).mean()) if not series.empty else np.nan
        risk_direction = str(ranking.get("risk_direction", "higher_is_riskier"))
        if pd.isna(latest_value):
            current_read = "missing"
        elif risk_direction == "higher_is_riskier":
            current_read = _percentile_read(percentile)
        else:
            current_read = _percentile_read(1.0 - percentile)
        rows.append(
            {
                "signal": signal,
                "market_date": str(latest["market_date"]),
                "latest_value": latest_value,
                "historical_percentile": percentile,
                "risk_direction": risk_direction,
                "current_risk_read": current_read,
                "predictive_score": _safe_float(ranking.get("predictive_score", np.nan)),
                "spearman_to_break_severity": _safe_float(
                    ranking.get("spearman_to_break_severity", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def write_prebreak_hindsight_outputs(
    result: PrebreakHindsightResult,
    output_dir: str | Path = DEFAULT_PREBREAK_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.snapshot_signals.to_csv(output_path / "snapshot_signal_panel.csv", index=False)
    result.signal_rankings.to_csv(output_path / "signal_predictiveness_rank.csv", index=False)
    result.action_timing.to_csv(output_path / "action_timing.csv", index=False)
    result.staged_risk_behavior.to_csv(output_path / "staged_risk_behavior.csv", index=False)
    result.late_trigger_mesh.to_csv(output_path / "late_trigger_mesh.csv", index=False)
    result.hard_defense_attribution.to_csv(
        output_path / "hard_defense_attribution.csv",
        index=False,
    )
    result.policy_variant_results.to_csv(output_path / "policy_variant_results.csv", index=False)
    result.current_signal_readout.to_csv(
        output_path / "current_best_signal_readout.csv", index=False
    )
    (output_path / "summary.md").write_text(result.summary, encoding="utf-8")


def build_prebreak_hindsight_summary(
    snapshot_signals: pd.DataFrame,
    signal_rankings: pd.DataFrame,
    action_timing: pd.DataFrame,
    staged_risk_behavior: pd.DataFrame,
    late_trigger_mesh: pd.DataFrame,
    hard_defense_attribution: pd.DataFrame,
    policy_variant_results: pd.DataFrame,
    current_signal_readout: pd.DataFrame,
) -> str:
    lines = [
        "# Pre-Break Hindsight Signal Analysis",
        "",
        "This is a hindsight research readout. It ranks saved snapshot signals against realized",
        "forward drawdown labels and should be treated as candidate evidence for later",
        "walk-forward testing, not as a live allocation rule.",
        "",
        "## Coverage",
        "",
        f"- snapshots analyzed: {len(snapshot_signals):,}",
        f"- date range: {snapshot_signals['market_date'].min()} to {snapshot_signals['market_date'].max()}",
        f"- event-window post-break snapshots: {_postbreak_snapshot_count(snapshot_signals):,}",
        f"- 3m break-label share: {snapshot_signals['forward_break_label_3m'].mean():.1%}",
        f"- 3m major-break-label share: {snapshot_signals['forward_major_break_label_3m'].mean():.1%}",
        "",
        "## Most Predictive Signals",
        "",
    ]
    if signal_rankings.empty:
        lines.append("No numeric signals had enough observations to rank.")
    else:
        for _, row in signal_rankings.head(12).iterrows():
            lines.append(
                "- "
                f"{row['signal']}: score {_safe_float(row['predictive_score']):.2f}, "
                f"spearman {_safe_float(row['spearman_to_break_severity']):.2f}, "
                f"direction {row['risk_direction']}"
            )
    lines.extend(["", "## Trade-Bot Action Timing", ""])
    if action_timing.empty:
        lines.append("No event-window action timing rows were available.")
    else:
        event_rows = action_timing[
            ~action_timing["event_name"].astype(str).str.startswith("ALL_SEVERE_")
        ]
        for _, row in event_rows.head(10).iterrows():
            defensive_date = str(row.get("first_defensive_market_date", "")) or "none"
            defensive_days = row.get("first_defensive_days_before_break", np.nan)
            hard_date = str(row.get("first_hard_defensive_market_date", "")) or "none"
            hard_days = row.get("first_hard_defensive_days_before_break", np.nan)
            lines.append(
                "- "
                f"{row['event_name']}: first broad defensive snapshot {defensive_date} "
                f"({_format_days_relative_to_break(defensive_days)}), "
                f"first hard defensive snapshot {hard_date} "
                f"({_format_days_relative_to_break(hard_days)}), "
                "hard aligned when severe "
                f"{_safe_float(row.get('hard_aligned_when_severe_share')):.1%}"
            )
    lines.extend(["", "## Staged-Risk Margin Experiment", ""])
    if staged_risk_behavior.empty:
        lines.append("No staged-risk rows were available.")
    else:
        aggregate = staged_risk_behavior[
            staged_risk_behavior["event_name"].astype(str).str.startswith("ALL_EVENTS_")
        ].copy()
        for _, row in aggregate.sort_values("prebreak_stage_order").iterrows():
            lines.append(
                "- "
                f"{str(row['prebreak_stage']).replace('_', ' ')}: "
                f"median actual budget "
                f"{_safe_float(row.get('median_risk_budget_multiplier')):.1%} "
                f"vs target {_safe_float(row.get('target_staged_risk_budget_multiplier')):.1%}; "
                f"hard-defense share "
                f"{_safe_float(row.get('hard_defensive_snapshot_share')):.1%}; "
                f"early hard false-alarm share "
                f"{_safe_float(row.get('early_hard_false_alarm_share')):.1%}"
            )
    lines.extend(["", "## Late Trigger Mesh", ""])
    if late_trigger_mesh.empty:
        lines.append("No late-trigger mesh rows were available.")
    else:
        for trigger, group in late_trigger_mesh.groupby("trigger_days_before_break"):
            lines.append(
                "- "
                f"{int(trigger)}d hard-defense gate: mean candidate risk-budget lift "
                f"{pd.to_numeric(group['mean_candidate_risk_budget_lift'], errors='coerce').mean():.1%}; "
                f"mean missed severe-label share "
                f"{pd.to_numeric(group['missed_severe_label_share_if_gated'], errors='coerce').mean():.1%}; "
                f"mean pre-trigger false-alarm share "
                f"{pd.to_numeric(group['pre_trigger_false_alarm_share'], errors='coerce').mean():.1%}"
            )
    lines.extend(["", "## Hard-Defense Attribution", ""])
    if hard_defense_attribution.empty:
        lines.append("No hard-defense attribution rows were available.")
    else:
        attribution = hard_defense_attribution[
            hard_defense_attribution["event_name"].astype(str).eq("ALL_EVENTS")
            & hard_defense_attribution["prebreak_stage"]
            .astype(str)
            .isin(["long_lead_context", "early_watch"])
            & hard_defense_attribution["hard_defense_source"].astype(str).ne("not_hard_defensive")
        ].copy()
        attribution = attribution.sort_values(
            "source_share_of_stage_hard_defense",
            ascending=False,
        )
        for _, row in attribution.head(8).iterrows():
            lines.append(
                "- "
                f"{str(row['prebreak_stage']).replace('_', ' ')} / "
                f"{str(row['hard_defense_source']).replace('_', ' ')}: "
                f"{_safe_float(row.get('source_share_of_stage_hard_defense')):.1%} "
                "of hard-defense snapshots; "
                f"median budget {_safe_float(row.get('median_risk_budget_multiplier')):.1%}"
            )
    lines.extend(["", "## Staged Policy Variants", ""])
    if policy_variant_results.empty:
        lines.append("No staged policy variant rows were available.")
    else:
        variants = policy_variant_results[
            policy_variant_results["event_name"].astype(str).eq("ALL_EVENTS")
        ].copy()
        variants = variants.sort_values("candidate_score", ascending=False)
        for _, row in variants.head(8).iterrows():
            lines.append(
                "- "
                f"{row['policy_name']}: read {row['policy_read']}; "
                f"median budget "
                f"{_safe_float(row.get('median_policy_risk_budget_multiplier')):.1%}; "
                f"mean false-alarm lift "
                f"{_safe_float(row.get('mean_false_alarm_risk_budget_lift')):.1%}; "
                f"mean severe-label lift "
                f"{_safe_float(row.get('mean_severe_label_risk_budget_lift')):.1%}; "
                f"score {_safe_float(row.get('candidate_score')):.2f}"
            )
    lines.extend(["", "## Current Read On Best Signals", ""])
    if current_signal_readout.empty:
        lines.append("No current signal readout was available.")
    else:
        for _, row in current_signal_readout.head(12).iterrows():
            lines.append(
                "- "
                f"{row['signal']}: { _safe_float(row['latest_value']):.3f}, "
                f"percentile { _safe_float(row['historical_percentile']):.1%}, "
                f"read {row['current_risk_read']}"
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This panel is intentionally hindsight-labeled and should be used to generate",
            "  hypotheses for purged or walk-forward tests.",
            "- Reference-control snapshots are included when available so the ranking is not",
            "  learned only from hand-picked pre-break windows.",
            "- Historical ETF coverage is uneven before newer AI and factor proxies existed.",
            "- The current AI drawdown question should weight AI leadership, concentration,",
            "  breadth, credit, volatility/liquidity, and cycle-leadership fragility signals",
            "  more heavily than generic broad-index trend alone.",
            "",
        ]
    )
    return "\n".join(lines)


def _add_forward_outcome_metrics(
    row: dict[str, object],
    *,
    reference_prices: pd.DataFrame,
    market_date: str,
    horizon_days: int,
    severe_drawdown_threshold: float,
    major_drawdown_threshold: float,
) -> None:
    outcomes: dict[str, tuple[float, float]] = {}
    for ticker in ("SPY", "QQQ", "SMH"):
        outcomes[ticker] = _forward_return_and_drawdown(
            reference_prices,
            ticker=ticker,
            market_date=market_date,
            horizon_days=horizon_days,
        )
        forward_return, forward_drawdown = outcomes[ticker]
        row[f"forward_{ticker.lower()}_return_3m"] = forward_return
        row[f"forward_{ticker.lower()}_max_drawdown_3m"] = forward_drawdown
    drawdowns = [value[1] for value in outcomes.values() if pd.notna(value[1])]
    min_drawdown = min(drawdowns) if drawdowns else np.nan
    row["forward_min_max_drawdown_3m"] = min_drawdown
    if not drawdowns:
        row["break_severity_3m"] = np.nan
        row["forward_break_label_3m"] = np.nan
        row["forward_major_break_label_3m"] = np.nan
        return
    row["break_severity_3m"] = max(0.0, -_safe_float(min_drawdown, 0.0))
    row["forward_break_label_3m"] = bool(
        _safe_float(outcomes.get("QQQ", (np.nan, np.nan))[1], 0.0) <= severe_drawdown_threshold
        or _safe_float(outcomes.get("SPY", (np.nan, np.nan))[1], 0.0) <= severe_drawdown_threshold
        or _safe_float(outcomes.get("SMH", (np.nan, np.nan))[1], 0.0)
        <= min(severe_drawdown_threshold, -0.15)
    )
    row["forward_major_break_label_3m"] = bool(
        _safe_float(outcomes.get("QQQ", (np.nan, np.nan))[1], 0.0) <= major_drawdown_threshold
        or _safe_float(outcomes.get("SPY", (np.nan, np.nan))[1], 0.0) <= major_drawdown_threshold
        or _safe_float(outcomes.get("SMH", (np.nan, np.nan))[1], 0.0)
        <= min(major_drawdown_threshold, -0.20)
    )


def _add_event_membership(
    row: dict[str, object],
    *,
    windows: tuple[BubbleBreakWindow, ...],
    lookback_days: int,
    postbreak_days: int = DEFAULT_POSTBREAK_FOLLOWTHROUGH_DAYS,
) -> None:
    market_date = pd.Timestamp(row["market_date"]).normalize()
    matches: list[BubbleBreakWindow] = []
    for window in windows:
        break_date = pd.Timestamp(window.break_date).normalize()
        start_date = break_date - pd.Timedelta(days=lookback_days)
        end_date = break_date + pd.Timedelta(days=postbreak_days)
        if start_date <= market_date <= end_date:
            matches.append(window)
    if not matches:
        row["event_name"] = ""
        row["event_family"] = ""
        row["event_break_date"] = ""
        row["event_description"] = ""
        row["days_to_break"] = np.nan
        row["postbreak_snapshot"] = False
        return
    match = min(
        matches,
        key=lambda window: abs((pd.Timestamp(window.break_date).normalize() - market_date).days),
    )
    break_date = pd.Timestamp(match.break_date).normalize()
    row["event_name"] = match.name
    row["event_family"] = match.family
    row["event_break_date"] = str(break_date.date())
    row["event_description"] = match.description
    row["days_to_break"] = int((break_date - market_date).days)
    row["postbreak_snapshot"] = bool(market_date > break_date)


def _add_staged_risk_metadata(row: dict[str, object]) -> None:
    stage = _staged_risk_bucket(
        row.get("days_to_break", np.nan),
        postbreak_snapshot=bool(row.get("postbreak_snapshot", False)),
    )
    row["prebreak_stage"] = stage
    row["prebreak_stage_order"] = STAGED_RISK_ORDER.get(stage, np.nan)
    target = STAGED_RISK_TARGETS.get(stage, np.nan)
    actual = _safe_float(row.get("risk_budget_multiplier", np.nan))
    row["target_staged_risk_budget_multiplier"] = target
    row["risk_budget_gap_to_stage_target"] = (
        target - actual if pd.notna(target) and pd.notna(actual) else np.nan
    )
    row["over_defensive_gap_to_stage_target"] = (
        max(0.0, target - actual) if pd.notna(target) and pd.notna(actual) else np.nan
    )
    row["under_defensive_gap_to_stage_target"] = (
        max(0.0, actual - target) if pd.notna(target) and pd.notna(actual) else np.nan
    )
    row["forward_upside_proxy_3m"] = _forward_upside_proxy_from_row(row)
    row["early_hard_false_alarm_flag"] = bool(
        _hard_defensive_action_flag(row)
        and _safe_float(row.get("days_to_break", np.nan)) >= 60
        and not _truthy_label(row.get("forward_break_label_3m"))
        and _safe_float(row.get("forward_upside_proxy_3m", np.nan), 0.0) > 0.0
    )


def _staged_risk_bucket(value: object, *, postbreak_snapshot: bool = False) -> str:
    days = _safe_float(value)
    if pd.isna(days):
        return ""
    if postbreak_snapshot or days < 0:
        return "postbreak_followthrough"
    if days <= 14:
        return "break_unwind"
    if days <= 45:
        return "confirmed_prebreak"
    if days <= 59:
        return "warning"
    if days <= 120:
        return "early_watch"
    return "long_lead_context"


def _staged_risk_summary_row(
    event_name: str,
    stage: str,
    group: pd.DataFrame,
) -> dict[str, object]:
    data = _with_forward_upside_proxy(group)
    days = pd.to_numeric(data["days_to_break"], errors="coerce")
    target = STAGED_RISK_TARGETS.get(stage, np.nan)
    risk_budget = pd.to_numeric(data.get("risk_budget_multiplier"), errors="coerce")
    early_false_alarm = _bool_flag_series(data, "early_hard_false_alarm_flag")
    return {
        "event_name": event_name,
        "event_family": _first_nonempty(data, "event_family"),
        "break_date": _first_nonempty(data, "event_break_date"),
        "prebreak_stage": stage,
        "prebreak_stage_order": STAGED_RISK_ORDER.get(stage, np.nan),
        "snapshots": len(data),
        "min_days_to_break": _safe_float(days.min()),
        "max_days_to_break": _safe_float(days.max()),
        "median_days_to_break": _safe_float(days.median()),
        "target_staged_risk_budget_multiplier": target,
        "median_risk_budget_multiplier": _safe_float(risk_budget.median()),
        "mean_candidate_risk_budget_lift": _candidate_lift_mean(
            data, pd.Series(True, index=data.index)
        ),
        "median_over_defensive_gap_to_stage_target": _safe_float(
            pd.to_numeric(data.get("over_defensive_gap_to_stage_target"), errors="coerce").median()
        ),
        "median_under_defensive_gap_to_stage_target": _safe_float(
            pd.to_numeric(data.get("under_defensive_gap_to_stage_target"), errors="coerce").median()
        ),
        "defensive_snapshot_share": float(_bool_flag_series(data, "defensive_action_flag").mean()),
        "hard_defensive_snapshot_share": float(
            _bool_flag_series(data, "hard_defensive_action_flag").mean()
        ),
        "severe_label_share": float(_bool_label_series(data["forward_break_label_3m"]).mean()),
        "major_label_share": float(
            _bool_label_series(
                data.get("forward_major_break_label_3m", pd.Series(index=data.index))
            ).mean()
        ),
        "positive_forward_upside_share": float(
            (pd.to_numeric(data["forward_upside_proxy_3m"], errors="coerce") > 0).mean()
        ),
        "early_hard_false_alarm_share": float(early_false_alarm.mean()),
        "median_forward_return_when_early_hard_false_alarm": _masked_median(
            data,
            "forward_upside_proxy_3m",
            early_false_alarm,
        ),
        "median_forward_drawdown_when_early_hard_false_alarm": _masked_median(
            data,
            "forward_min_max_drawdown_3m",
            early_false_alarm,
        ),
    }


def _hard_defense_attribution_rows(event_name: str, group: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data = group.copy()
    if "prebreak_stage" not in data:
        return rows
    for (stage, source), source_group in data.groupby(
        ["prebreak_stage", "hard_defense_source"],
        dropna=True,
        sort=False,
    ):
        stage_group = data[data["prebreak_stage"].astype(str).eq(str(stage))]
        hard_group = stage_group[_bool_flag_series(stage_group, "hard_defensive_action_flag")]
        if stage_group.empty:
            continue
        source_hard = source_group[_bool_flag_series(source_group, "hard_defensive_action_flag")]
        rows.append(
            {
                "event_name": event_name,
                "prebreak_stage": str(stage),
                "prebreak_stage_order": STAGED_RISK_ORDER.get(str(stage), np.nan),
                "hard_defense_source": str(source),
                "snapshots": len(source_group),
                "hard_defensive_snapshots": len(source_hard),
                "source_share_of_stage_snapshots": float(len(source_group) / len(stage_group)),
                "source_share_of_stage_hard_defense": (
                    float(len(source_hard) / len(hard_group)) if len(hard_group) else np.nan
                ),
                "median_risk_budget_multiplier": _safe_float(
                    pd.to_numeric(
                        source_group.get("risk_budget_multiplier"),
                        errors="coerce",
                    ).median()
                ),
                "median_current_risk_asset_weight": _safe_float(
                    pd.to_numeric(
                        source_group.get("current_risk_asset_weight"),
                        errors="coerce",
                    ).median()
                ),
                "median_scenario_event_macro_multiplier": _safe_float(
                    pd.to_numeric(
                        source_group.get("scenario_event_macro_multiplier"),
                        errors="coerce",
                    ).median()
                ),
                "median_portfolio_risk_multiplier": _safe_float(
                    pd.to_numeric(
                        source_group.get("portfolio_risk_multiplier"),
                        errors="coerce",
                    ).median()
                ),
                "early_hard_false_alarm_share": float(
                    _bool_flag_series(source_group, "early_hard_false_alarm_flag").mean()
                ),
            }
        )
    return rows


def _hard_defense_source(row: pd.Series) -> str:
    if not bool(row.get("hard_defensive_action_flag", False)):
        return "not_hard_defensive"
    attribution_sources = {
        "quantitative_risk_status": "quantitative_risk_status",
        "scenario_probabilities": "scenario_probabilities",
        "news_event_pressure": "news_event_pressure",
        "macro_quantitative": "macro_quantitative",
        "portfolio_absolute_risk": "portfolio_absolute_risk",
        "decision_sanity": "decision_sanity_guardrail",
    }
    marginal_adds = {
        output: _safe_float(
            row.get(f"attribution_{layer}_defensive_add_pp", np.nan),
            0.0,
        )
        for layer, output in attribution_sources.items()
    }
    base_defensive = _safe_float(
        row.get("attribution_base_market_strategy_defensive_weight", np.nan)
    )
    if pd.notna(base_defensive) and base_defensive >= 0.65:
        return "base_strategy_already_defensive"
    largest_source = max(marginal_adds, key=marginal_adds.get)
    if marginal_adds[largest_source] > 1e-6:
        return largest_source
    current_risk = _safe_float(row.get("current_risk_asset_weight", np.nan))
    target_risk = _safe_float(row.get("target_risk_asset_weight", np.nan))
    risk_budget = _safe_float(row.get("risk_budget_multiplier", np.nan))
    scenario_multiplier = _safe_float(row.get("scenario_event_macro_multiplier", np.nan))
    portfolio_multiplier = _safe_float(row.get("portfolio_risk_multiplier", np.nan))
    risk_status = str(row.get("risk_status", "")).lower()
    action = str(row.get("recommended_action", "")).upper()
    if pd.notna(current_risk) and current_risk <= 0.55:
        return "base_strategy_already_defensive"
    portfolio_constraints = str(row.get("portfolio_constraints", "")).lower()
    if pd.notna(portfolio_multiplier) and portfolio_multiplier <= 0.70:
        if "scenario_weighted_stress" in portfolio_constraints:
            return "portfolio_scenario_weighted_stress"
        if "expected_shortfall" in portfolio_constraints:
            return "portfolio_expected_shortfall"
        if "stress_loss" in portfolio_constraints:
            return "portfolio_stress_loss"
        if "equity_beta" in portfolio_constraints:
            return "portfolio_equity_beta"
        if "ai_beta" in portfolio_constraints:
            return "portfolio_ai_beta"
        if "scenario_min_defensive" in portfolio_constraints:
            return "portfolio_min_defensive"
        return "portfolio_risk_engine"
    if pd.notna(scenario_multiplier) and scenario_multiplier <= 0.70:
        return "scenario_event_macro_overlay"
    if pd.notna(target_risk) and pd.notna(current_risk) and current_risk - target_risk >= 0.20:
        return "overlay_reduced_risk_assets"
    if pd.notna(risk_budget) and risk_budget <= 0.50:
        return "low_final_risk_budget"
    if risk_status in ACTION_DEFENSIVE_STATUSES or any(
        token in action for token in ACTION_REDUCE_TOKENS
    ):
        return "risk_status_or_action"
    return "unknown"


def _add_trade_attribution_metrics(
    row: dict[str, object],
    attribution: pd.DataFrame,
) -> None:
    required = {"layer", "defensive_weight", "marginal_defensive_add_pp"}
    if attribution.empty or not required.issubset(attribution.columns):
        return
    for _, attribution_row in attribution.iterrows():
        layer = _slug(str(attribution_row.get("layer", "")))
        if not layer:
            continue
        row[f"attribution_{layer}_defensive_weight"] = _safe_float(
            attribution_row.get("defensive_weight", np.nan)
        )
        row[f"attribution_{layer}_defensive_add_pp"] = _safe_float(
            attribution_row.get("marginal_defensive_add_pp", np.nan)
        )
        row[f"attribution_{layer}_authority"] = _safe_float(
            attribution_row.get("authority", np.nan)
        )


def _policy_variant_summary_row(
    event_name: str,
    policy_name: str,
    group: pd.DataFrame,
    policy: Any,
) -> dict[str, object]:
    data = _with_forward_upside_proxy(group)
    actual = pd.to_numeric(data["risk_budget_multiplier"], errors="coerce").clip(0.0, 1.0)
    proposed = data.apply(policy, axis=1).astype(float).clip(0.0, 1.0)
    lift = (proposed - actual).clip(lower=0.0)
    severe = _bool_label_series(data["forward_break_label_3m"])
    early = pd.to_numeric(data["days_to_break"], errors="coerce").ge(60)
    false_alarm = _false_alarm_mask(data)
    upside = pd.to_numeric(data["forward_upside_proxy_3m"], errors="coerce")
    drawdown = pd.to_numeric(data["forward_min_max_drawdown_3m"], errors="coerce")
    incremental_return = lift * upside
    incremental_drawdown = lift * drawdown
    candidate_score = _policy_candidate_score(
        incremental_return=incremental_return,
        incremental_drawdown=incremental_drawdown,
        severe_added_lift=lift[severe].mean(),
        false_alarm_lift=lift[false_alarm].mean(),
    )
    return {
        "event_name": event_name,
        "policy_name": policy_name,
        "snapshots": len(data),
        "median_actual_risk_budget_multiplier": _safe_float(actual.median()),
        "median_policy_risk_budget_multiplier": _safe_float(proposed.median()),
        "mean_risk_budget_lift": _safe_float(lift.mean()),
        "mean_early_risk_budget_lift": _safe_float(lift[early].mean()),
        "mean_false_alarm_risk_budget_lift": _safe_float(lift[false_alarm].mean()),
        "mean_severe_label_risk_budget_lift": _safe_float(lift[severe].mean()),
        "median_incremental_return_proxy_3m": _safe_float(incremental_return.median()),
        "mean_incremental_return_proxy_3m": _safe_float(incremental_return.mean()),
        "median_incremental_drawdown_proxy_3m": _safe_float(incremental_drawdown.median()),
        "mean_incremental_drawdown_proxy_3m": _safe_float(incremental_drawdown.mean()),
        "severe_label_share": float(severe.mean()),
        "false_alarm_share": float(false_alarm.mean()),
        "candidate_score": candidate_score,
        "policy_read": _policy_read(candidate_score, lift[severe].mean()),
    }


def _policy_actual_budget(row: pd.Series) -> float:
    return _safe_float(row.get("risk_budget_multiplier", np.nan), 0.0)


def _policy_stage_floor_budget(row: pd.Series) -> float:
    actual = _policy_actual_budget(row)
    target = _safe_float(row.get("target_staged_risk_budget_multiplier", np.nan), actual)
    return max(actual, target)


def _policy_confirm_gate_budget(row: pd.Series, trigger_days: int) -> float:
    days = _safe_float(row.get("days_to_break", np.nan))
    if pd.isna(days) or days <= trigger_days:
        return _policy_stage_floor_budget(row)
    stage = str(row.get("prebreak_stage", ""))
    if stage == "long_lead_context":
        return max(_policy_actual_budget(row), 0.90)
    if stage == "early_watch":
        return max(_policy_actual_budget(row), 0.75)
    return max(_policy_actual_budget(row), 0.60)


def _policy_watch_warning_floor_budget(row: pd.Series) -> float:
    actual = _policy_actual_budget(row)
    stage = str(row.get("prebreak_stage", ""))
    floors = {
        "long_lead_context": 0.85,
        "early_watch": 0.75,
        "warning": 0.60,
    }
    return max(actual, floors.get(stage, actual))


def _policy_portfolio_watch_floor_budget(row: pd.Series) -> float:
    actual = _policy_actual_budget(row)
    if not _portfolio_sourced_hard_defense(row):
        return actual
    stage = str(row.get("prebreak_stage", ""))
    floors = {
        "long_lead_context": 0.85,
        "early_watch": 0.75,
        "warning": 0.60,
    }
    return max(actual, floors.get(stage, actual))


def _policy_portfolio_confirm_budget(
    row: pd.Series,
    trigger_days: int,
    *,
    floors: dict[str, float] | None = None,
) -> float:
    actual = _policy_actual_budget(row)
    if not _portfolio_sourced_hard_defense(row):
        return actual
    break_count = _safe_float(row.get("decision_sanity_break_count", np.nan), 0.0)
    days = _safe_float(row.get("days_to_break", np.nan))
    stage = str(row.get("prebreak_stage", ""))
    if pd.notna(days) and days <= trigger_days:
        return max(actual, STAGED_RISK_TARGETS.get(stage, actual))
    if break_count >= 2:
        return actual
    floor_map = floors or {
        "long_lead_context": 0.85,
        "early_watch": 0.75,
        "warning": 0.60,
    }
    return max(actual, floor_map.get(stage, actual))


def _portfolio_sourced_hard_defense(row: pd.Series) -> bool:
    source = _hard_defense_source(row)
    return source.startswith("portfolio_")


def _policy_candidate_score(
    *,
    incremental_return: pd.Series,
    incremental_drawdown: pd.Series,
    severe_added_lift: float,
    false_alarm_lift: float,
) -> float:
    mean_return = _safe_float(incremental_return.mean(), 0.0)
    mean_drawdown = abs(_safe_float(incremental_drawdown.mean(), 0.0))
    severe_lift = _safe_float(severe_added_lift, 0.0)
    false_alarm = _safe_float(false_alarm_lift, 0.0)
    return float(mean_return + 0.50 * false_alarm - 1.75 * severe_lift - 0.50 * mean_drawdown)


def _policy_read(candidate_score: float, severe_added_lift: float) -> str:
    severe_lift = _safe_float(severe_added_lift, 0.0)
    if severe_lift >= 0.20:
        return "too_much_crash_exposure"
    if candidate_score > 0.04:
        return "promising"
    if candidate_score > 0.00:
        return "mixed"
    return "not_enough_edge"


def _add_scenario_driver_metrics(row: dict[str, object], frame: pd.DataFrame) -> None:
    if frame.empty or not {"driver", "score"}.issubset(frame.columns):
        return
    for _, driver_row in frame.iterrows():
        key = _slug(str(driver_row.get("driver", "")))
        if key:
            row[f"driver_{key}"] = _safe_float(driver_row.get("score", np.nan))


def _add_confirmation_metrics(row: dict[str, object], frame: pd.DataFrame) -> None:
    if frame.empty or not {"name", "score"}.issubset(frame.columns):
        return
    for _, signal_row in frame.iterrows():
        key = _slug(str(signal_row.get("name", "")))
        if key:
            row[f"confirmation_{key}"] = _safe_float(signal_row.get("score", np.nan))


def _add_market_health_metrics(row: dict[str, object], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    for ticker in ("SPY", "QQQ", "RSP", "IWM", "HYG", "LQD", "TLT", "GLD", "SMH", "VIXY", "UUP"):
        if ticker not in frame.index:
            continue
        for column in ("return_1m", "return_3m", "drawdown"):
            if ticker == "VIXY" and column == "drawdown":
                continue
            if column in frame:
                row[f"health_{ticker.lower()}_{column}"] = _safe_float(frame.loc[ticker, column])


def _add_regime_instability_metrics(
    row: dict[str, object],
    summary: pd.DataFrame,
    components: pd.DataFrame,
) -> None:
    if not summary.empty:
        first = summary.iloc[0]
        row["regime_instability_score"] = _safe_float(first.get("regime_instability_score", np.nan))
        row["spy_ytd_large_move_share"] = _safe_float(first.get("spy_ytd_large_move_share", np.nan))
    if components.empty or not {"component", "component_score"}.issubset(components.columns):
        return
    for _, component_row in components.iterrows():
        key = _slug(str(component_row.get("component", "")))
        if key:
            row[f"instability_{key}"] = _safe_float(component_row.get("component_score", np.nan))


def _add_cycle_metrics(row: dict[str, object], prices: pd.DataFrame) -> None:
    try:
        feature = build_cycle_feature_snapshot(prices)
    except (ValueError, KeyError, TypeError):
        return
    row["cycle_dominant_phase_probability"] = _safe_float(
        feature.get("dominant_phase_probability", np.nan)
    )
    probabilities = feature.get("probabilities", {})
    if isinstance(probabilities, dict):
        for phase, value in probabilities.items():
            row[f"cycle_probability_{_slug(str(phase))}"] = _safe_float(value)
    components = feature.get("components", {})
    if isinstance(components, dict):
        for component, value in components.items():
            row[f"cycle_component_{_slug(str(component))}"] = _safe_float(value)


def _forward_return_and_drawdown(
    prices: pd.DataFrame,
    *,
    ticker: str,
    market_date: str,
    horizon_days: int,
) -> tuple[float, float]:
    if ticker not in prices:
        return np.nan, np.nan
    series = prices[ticker].dropna()
    if series.empty:
        return np.nan, np.nan
    market_timestamp = pd.Timestamp(market_date).normalize()
    positions = np.flatnonzero(series.index.normalize() <= market_timestamp)
    if len(positions) == 0:
        return np.nan, np.nan
    origin_pos = int(positions[-1])
    start_pos = origin_pos + 1
    end_pos = min(start_pos + int(horizon_days), len(series) - 1)
    if start_pos >= len(series) or end_pos <= start_pos:
        return np.nan, np.nan
    path = series.iloc[start_pos : end_pos + 1]
    start_price = float(series.iloc[origin_pos])
    if start_price <= 0 or path.empty:
        return np.nan, np.nan
    forward_return = float(path.iloc[-1] / start_price - 1.0)
    relative = path / start_price
    max_drawdown = float((relative / relative.cummax() - 1.0).min())
    return forward_return, max_drawdown


def _defensive_action_flag(row: dict[str, object]) -> bool:
    status = str(row.get("risk_status", "")).lower()
    action = str(row.get("recommended_action", "")).upper()
    multiplier = _safe_float(row.get("risk_budget_multiplier", np.nan))
    return (
        status in ACTION_DEFENSIVE_STATUSES
        or any(token in action for token in ACTION_REDUCE_TOKENS)
        or (pd.notna(multiplier) and multiplier <= 0.65)
    )


def _hard_defensive_action_flag(row: dict[str, object]) -> bool:
    status = str(row.get("risk_status", "")).lower()
    action = str(row.get("recommended_action", "")).upper()
    multiplier = _safe_float(row.get("risk_budget_multiplier", np.nan))
    return (
        status in ACTION_DEFENSIVE_STATUSES
        or action.startswith("REDUCE")
        or "DE_RISK" in action
        or "DEFENSIVE" in action
        or (pd.notna(multiplier) and multiplier <= 0.50)
    )


def _action_severity_score(row: dict[str, object]) -> float:
    status_score = _safe_float(row.get("risk_status_score", np.nan), 0.0)
    multiplier = _safe_float(row.get("risk_budget_multiplier", np.nan))
    multiplier_score = 0.0 if pd.isna(multiplier) else max(0.0, min(1.0, 1.0 - multiplier))
    hard_score = 1.0 if _hard_defensive_action_flag(row) else 0.0
    broad_score = 0.5 if _defensive_action_flag(row) else 0.0
    return max(status_score, multiplier_score, hard_score, broad_score)


def _risk_status_score(status: str) -> float:
    return {"green": 0.0, "yellow": 0.33, "orange": 0.67, "red": 1.0}.get(
        status.lower(),
        np.nan,
    )


def _directional_auc(positives: pd.Series, negatives: pd.Series) -> float:
    positives = pd.to_numeric(positives, errors="coerce").dropna()
    negatives = pd.to_numeric(negatives, errors="coerce").dropna()
    if positives.empty or negatives.empty:
        return np.nan
    comparisons = []
    for value in positives:
        comparisons.append(float((value > negatives).mean() + 0.5 * (value == negatives).mean()))
    return float(np.mean(comparisons)) if comparisons else np.nan


def _aligned_when_severe_share(
    frame: pd.DataFrame,
    *,
    action_column: str = "defensive_action_flag",
) -> float:
    severe = frame[_bool_label_series(frame["forward_break_label_3m"])]
    if severe.empty:
        return np.nan
    return float(_bool_flag_series(severe, action_column).mean())


def _first_nonempty(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    values = values[values.ne("")]
    return str(values.iloc[0]) if not values.empty else ""


def _with_forward_upside_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "forward_upside_proxy_3m" in data:
        data["forward_upside_proxy_3m"] = pd.to_numeric(
            data["forward_upside_proxy_3m"],
            errors="coerce",
        )
        return data
    returns = [
        pd.to_numeric(data[column], errors="coerce")
        for column in [
            "forward_spy_return_3m",
            "forward_qqq_return_3m",
            "forward_smh_return_3m",
        ]
        if column in data
    ]
    if returns:
        data["forward_upside_proxy_3m"] = pd.concat(returns, axis=1).max(axis=1)
    else:
        data["forward_upside_proxy_3m"] = np.nan
    return data


def _forward_upside_proxy_from_row(row: dict[str, object]) -> float:
    values = [
        _safe_float(row.get("forward_spy_return_3m", np.nan)),
        _safe_float(row.get("forward_qqq_return_3m", np.nan)),
        _safe_float(row.get("forward_smh_return_3m", np.nan)),
    ]
    clean = [value for value in values if pd.notna(value)]
    return max(clean) if clean else np.nan


def _false_alarm_mask(frame: pd.DataFrame) -> pd.Series:
    data = _with_forward_upside_proxy(frame)
    days = pd.to_numeric(data.get("days_to_break", pd.Series(index=data.index)), errors="coerce")
    forward_break = _bool_label_series(data["forward_break_label_3m"])
    upside = pd.to_numeric(data["forward_upside_proxy_3m"], errors="coerce")
    return days.ge(60) & ~forward_break & upside.gt(0.0)


def _masked_mean(values: pd.Series, mask: pd.Series) -> float:
    aligned_mask = mask.reindex(values.index).fillna(False).astype(bool)
    selected = values[aligned_mask]
    if selected.empty:
        return np.nan
    return float(pd.to_numeric(selected, errors="coerce").mean())


def _masked_median(frame: pd.DataFrame, column: str, mask: pd.Series) -> float:
    if column not in frame:
        return np.nan
    aligned_mask = mask.reindex(frame.index).fillna(False).astype(bool)
    values = pd.to_numeric(frame.loc[aligned_mask, column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _candidate_lift_mean(frame: pd.DataFrame, mask: pd.Series) -> float:
    if "risk_budget_multiplier" not in frame:
        return np.nan
    aligned_mask = mask.reindex(frame.index).fillna(False).astype(bool)
    actual = pd.to_numeric(frame.loc[aligned_mask, "risk_budget_multiplier"], errors="coerce")
    if "target_staged_risk_budget_multiplier" in frame:
        target = pd.to_numeric(
            frame.loc[aligned_mask, "target_staged_risk_budget_multiplier"],
            errors="coerce",
        )
    else:
        target = frame.loc[aligned_mask, "prebreak_stage"].map(STAGED_RISK_TARGETS)
    lift = (target - actual).clip(lower=0).dropna()
    return float(lift.mean()) if not lift.empty else np.nan


def _late_trigger_read(
    *,
    severe_total: int,
    missed_severe_share: float,
    candidate_lift: float,
) -> str:
    if severe_total <= 0:
        return "no_severe_labels"
    if pd.notna(missed_severe_share) and missed_severe_share > 0.25:
        return "too_late"
    if pd.notna(candidate_lift) and candidate_lift >= 0.10:
        return "promising"
    return "limited_lift"


def _bool_label_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).astype(bool)


def _bool_flag_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _truthy_label(value: object) -> bool:
    numeric = _safe_float(value)
    return bool(pd.notna(numeric) and numeric != 0.0)


def _latest_signal_value(frame: pd.DataFrame, column: str) -> float:
    if "market_date" not in frame or column not in frame:
        return np.nan
    ordered = frame.sort_values("market_date")
    values = pd.to_numeric(ordered[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan


def _postbreak_snapshot_count(snapshot_signals: pd.DataFrame) -> int:
    if "postbreak_snapshot" not in snapshot_signals:
        return 0
    return int(snapshot_signals["postbreak_snapshot"].astype(bool).sum())


def _percentile_read(risk_percentile: float) -> str:
    if pd.isna(risk_percentile):
        return "missing"
    if risk_percentile >= 0.85:
        return "high_risk"
    if risk_percentile >= 0.65:
        return "elevated"
    if risk_percentile >= 0.40:
        return "mixed"
    return "contained"


def _latest_date_per_week(
    dates: pd.DatetimeIndex,
    *,
    weekly_frequency: str,
) -> tuple[pd.Timestamp, ...]:
    if dates.empty:
        return ()
    frame = pd.DataFrame({"market_date": dates.sort_values()})
    frame["weekly_bucket"] = frame["market_date"].dt.to_period(weekly_frequency).astype(str)
    weekly = frame.groupby("weekly_bucket", sort=True)["market_date"].max()
    return tuple(pd.DatetimeIndex(weekly).sort_values())


def _normalize_dates(
    values: pd.DatetimeIndex | list[object] | tuple[object, ...],
) -> pd.DatetimeIndex:
    dates = pd.to_datetime(list(values), errors="coerce")
    dates = pd.DatetimeIndex(dates).dropna()
    if dates.empty:
        return dates
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    return dates.normalize().drop_duplicates().sort_values()


def _clean_prices(prices: Any) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        return pd.DataFrame()
    clean = prices.dropna(how="all").sort_index().ffill()
    clean.index = pd.DatetimeIndex(clean.index).tz_localize(None).normalize()
    return clean


def _first_row(frame: Any) -> dict[str, object]:
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return frame.iloc[0].to_dict()
    return {}


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if np.isinf(numeric):
        return default
    return numeric


def _slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _format_days(value: object) -> str:
    numeric = _safe_float(value)
    if pd.isna(numeric):
        return "n/a"
    return f"{int(numeric)}d"


def _format_days_relative_to_break(value: object) -> str:
    numeric = _safe_float(value)
    if pd.isna(numeric):
        return "n/a"
    days = int(numeric)
    if days > 0:
        return f"{days}d before break"
    if days < 0:
        return f"{abs(days)}d after break"
    return "break date"
