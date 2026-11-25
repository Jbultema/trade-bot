from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import requests

from trade_bot.backtest.engine import BacktestResult, run_backtest
from trade_bot.config import ExecutionConfig
from trade_bot.research.orthogonal_frontier import (
    _control_returns,
    _frame_hash,
    _metric_row,
    _period_slice,
)

COMMON_START: Final = pd.Timestamp("2012-01-01")
PERIODS: Final = {
    "development": (pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-31")),
    "validation": (pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31")),
    "holdout": (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
    "forward": (pd.Timestamp("2023-01-01"), None),
    "common_full": (COMMON_START, None),
}
USER_AGENT: Final = "trade-bot public-signal research contact github.com/Jbultema"

CFTC_TFF_URL: Final = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
CFTC_DISAGG_URL: Final = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
CBOE_URLS: Final = {
    "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "vix3m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "skew": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
}
FINRA_MARGIN_URL: Final = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"

CFTC_CONTRACTS: Final = {
    "SPY": ("tff", ("S&P 500 Consolidated",)),
    "QQQ": ("tff", ("NASDAQ-100 Consolidated",)),
    "IWM": (
        "tff",
        (
            "RUSSELL 2000 MINI INDEX FUTURE",
            "E-MINI RUSSELL 2000 INDEX",
            "RUSSELL E-MINI",
        ),
    ),
    "EFA": (
        "tff",
        ("E-MINI MSCI EAFE", "MSCI EAFE MINI INDEX", "MSCI EAFE  -"),
    ),
    "EEM": (
        "tff",
        (
            "E-MINI MSCI EMERGING MARKETS",
            "MSCI EMERGING MKTS MINI INDEX",
            "MSCI EMERGING MKTS INDEX",
            "MSCI EM INDEX",
        ),
    ),
    "IEF": ("tff", ("10-YEAR U.S. TREASURY NOTES", "UST 10Y NOTE")),
    "TLT": ("tff", ("U.S. TREASURY BONDS", "UST BOND")),
    "UUP": ("tff", ("U.S. DOLLAR INDEX", "USD INDEX")),
    "DBC": ("tff", ("BLOOMBERG COMMODITY INDEX", "BBG COMMODITY")),
    "GLD": ("disagg", ("GOLD - COMMODITY EXCHANGE",)),
    "CPER": ("disagg", ("COPPER-GRADE #1", "COPPER- #1")),
}


@dataclass(frozen=True)
class SignalCandidate:
    name: str
    family: str
    hypothesis: str
    target_weights: pd.DataFrame
    signal_lag_days: int


@dataclass(frozen=True)
class PublicSignalSurvey:
    candidate_metrics: pd.DataFrame
    family_selection: pd.DataFrame
    overlay_selection: pd.DataFrame
    combination_metrics: pd.DataFrame
    source_feasibility: pd.DataFrame
    cost_stress: pd.DataFrame
    report_path: Path


def run_public_signal_frontier_survey(
    *,
    prices_path: str | Path = "data/cache/yahoo_prices.parquet",
    config_path: str | Path = "configs/baseline.yaml",
    raw_dir: str | Path = "research/public_signal_frontiers/raw",
    output_dir: str | Path = "reports/public_signal_frontiers",
    start_year: int = 2010,
    end_year: int = 2026,
    refresh: bool = False,
) -> PublicSignalSurvey:
    prices = _load_prices(Path(prices_path))
    raw = Path(raw_dir)
    output = Path(output_dir)
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    cftc, cftc_inventory = acquire_cftc_panel(
        raw,
        start_year=start_year,
        end_year=end_year,
        refresh=refresh,
    )
    options, options_inventory = acquire_cboe_panel(raw, refresh=refresh)
    margin, margin_inventory = acquire_finra_margin(raw, refresh=refresh)
    source_feasibility = build_source_feasibility(
        raw,
        cftc_inventory=cftc_inventory,
        options_inventory=options_inventory,
        margin_inventory=margin_inventory,
    )

    candidates = [
        *build_cftc_candidates(prices, cftc),
        *build_options_candidates(prices, options),
        *build_margin_candidates(prices, margin),
    ]
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Public-signal candidate names must be unique.")

    results = {
        candidate.name: _run_candidate(candidate, prices, transaction_cost_bps=5.0)
        for candidate in candidates
    }
    candidate_metrics = _candidate_metric_frame(candidates, results)
    controls = _control_returns(
        prices,
        config_path=Path(config_path),
        rotation_returns_path=None,
        swing_returns_path=None,
    )
    control_metrics = _return_metric_frame(controls, label="control")
    family_selection = add_edge_reads(select_family_winners(candidate_metrics), control_metrics)
    overlay_metrics = build_trade_bot_overlay_metrics(
        candidates,
        results,
        prices,
        controls,
    )
    overlay_selection = add_edge_reads(
        select_family_winners(overlay_metrics),
        control_metrics,
        benchmark="trade_bot_primary",
    )
    combination_returns = build_combination_returns(
        candidates,
        results,
        family_selection,
        prices,
        controls,
    )
    combination_metrics = _return_metric_frame(combination_returns, label="combination")
    cost_stress = _cost_stress(candidates, family_selection, prices)

    candidate_metrics.to_csv(output / "candidate_metrics.csv", index=False)
    family_selection.to_csv(output / "family_selection.csv", index=False)
    overlay_metrics.to_csv(output / "trade_bot_overlay_metrics.csv", index=False)
    overlay_selection.to_csv(output / "trade_bot_overlay_selection.csv", index=False)
    combination_metrics.to_csv(output / "combination_metrics.csv", index=False)
    control_metrics.to_csv(output / "control_metrics.csv", index=False)
    source_feasibility.to_csv(output / "source_feasibility.csv", index=False)
    cost_stress.to_csv(output / "cost_stress.csv", index=False)
    cftc.to_csv(output / "cftc_signal_panel.csv", index=False)
    options.to_csv(output / "options_signal_panel.csv", index_label="date")
    margin.to_csv(output / "margin_signal_panel.csv", index_label="available_date")

    report_path = output / "report.md"
    report_path.write_text(
        _render_report(
            family_selection,
            overlay_selection,
            combination_metrics,
            control_metrics,
            source_feasibility,
            cost_stress,
            variants=len(candidates),
            data_end=prices.index.max(),
        ),
        encoding="utf-8",
    )
    manifest = {
        "study": "public_signal_frontiers",
        "generated_from_official_public_sources": True,
        "data_end": str(prices.index.max().date()),
        "common_start": str(COMMON_START.date()),
        "variants_tested": len(candidates),
        "families_tested": sorted({candidate.family for candidate in candidates}),
        "selection_policy": (
            "Family winners use development 2012-2016 and validation 2017-2019 only; "
            "2020-2022 holdout and 2023+ forward rows are read after selection. All "
            "periods remain retrospective until a prospective freeze matures."
        ),
        "cftc_availability_policy": "Tuesday observation, Friday availability, next-session execution",
        "finra_availability_policy": (
            "Third Friday of following month approximation; exact historical release dates unavailable"
        ),
        "price_hash": _frame_hash(prices),
        "cftc_hash": _frame_hash(cftc.set_index(["available_date", "ticker"])),
        "options_hash": _frame_hash(options),
        "margin_hash": _frame_hash(margin),
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return PublicSignalSurvey(
        candidate_metrics=candidate_metrics,
        family_selection=family_selection,
        overlay_selection=overlay_selection,
        combination_metrics=combination_metrics,
        source_feasibility=source_feasibility,
        cost_stress=cost_stress,
        report_path=report_path,
    )


def acquire_cftc_panel(
    raw_dir: Path,
    *,
    start_year: int,
    end_year: int,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {"tff": [], "disagg": []}
    inventory: list[dict[str, object]] = []
    for report_type, template in (("tff", CFTC_TFF_URL), ("disagg", CFTC_DISAGG_URL)):
        for year in range(start_year, end_year + 1):
            url = template.format(year=year)
            path = raw_dir / f"cftc_{report_type}_{year}.zip"
            try:
                payload, status = _cached_get(url, path, refresh=refresh)
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    member = archive.namelist()[0]
                    frame = pd.read_csv(archive.open(member), low_memory=False)
                frames[report_type].append(frame)
                error = ""
            except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile) as exc:
                status = "failed"
                error = str(exc)
                payload = b""
            inventory.append(
                {
                    "source": f"cftc_{report_type}",
                    "year": year,
                    "url": url,
                    "status": status,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest() if payload else "",
                    "error": error,
                }
            )
    panel_rows: list[pd.DataFrame] = []
    for ticker, (report_type, aliases) in CFTC_CONTRACTS.items():
        if not frames[report_type]:
            continue
        combined = pd.concat(frames[report_type], ignore_index=True)
        names = combined["Market_and_Exchange_Names"].astype(str).str.strip()
        alias_rank = pd.Series(np.nan, index=combined.index)
        for rank, alias in enumerate(aliases):
            matches = names.str.startswith(alias, na=False)
            alias_rank.loc[matches & alias_rank.isna()] = rank
        selected = combined.loc[alias_rank.notna()].copy()
        selected["alias_rank"] = alias_rank.loc[selected.index]
        if selected.empty:
            continue
        report_date = pd.to_datetime(selected.get("Report_Date_as_YYYY-MM-DD"), errors="coerce")
        if "Report_Date_as_MM_DD_YYYY" in selected.columns:
            report_date = report_date.fillna(
                pd.to_datetime(selected["Report_Date_as_MM_DD_YYYY"], errors="coerce")
            )
        selected["report_date"] = report_date
        selected["open_interest"] = pd.to_numeric(selected["Open_Interest_All"], errors="coerce")
        if report_type == "tff":
            long_col = "Asset_Mgr_Positions_Long_All"
            short_col = "Asset_Mgr_Positions_Short_All"
        else:
            long_col = "M_Money_Positions_Long_All"
            short_col = "M_Money_Positions_Short_All"
        selected["net_position"] = (
            pd.to_numeric(selected[long_col], errors="coerce")
            - pd.to_numeric(selected[short_col], errors="coerce")
        ) / selected["open_interest"].replace(0.0, np.nan)
        selected["concentration"] = (
            pd.to_numeric(selected["Conc_Gross_LE_4_TDR_Long_All"], errors="coerce")
            + pd.to_numeric(selected["Conc_Gross_LE_4_TDR_Short_All"], errors="coerce")
        ) / 200.0
        selected = selected.sort_values(["report_date", "alias_rank"]).drop_duplicates(
            "report_date", keep="first"
        )
        selected["available_date"] = selected["report_date"] + pd.Timedelta(days=3)
        selected["ticker"] = ticker
        selected["report_type"] = report_type
        selected["contract"] = selected["Market_and_Exchange_Names"].astype(str).str.strip()
        selected["net_change_4w"] = selected["net_position"].diff(4)
        panel_rows.append(
            selected[
                [
                    "ticker",
                    "report_type",
                    "contract",
                    "report_date",
                    "available_date",
                    "open_interest",
                    "net_position",
                    "net_change_4w",
                    "concentration",
                ]
            ]
        )
    panel = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    return panel.sort_values(["available_date", "ticker"]), pd.DataFrame(inventory)


def acquire_cboe_panel(
    raw_dir: Path,
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series: list[pd.Series] = []
    inventory: list[dict[str, object]] = []
    for name, url in CBOE_URLS.items():
        path = raw_dir / f"cboe_{name}.csv"
        try:
            payload, status = _cached_get(url, path, refresh=refresh)
            frame = pd.read_csv(io.BytesIO(payload))
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            value_column = "close" if "close" in frame.columns else name
            close = pd.to_numeric(frame[value_column], errors="coerce")
            series.append(pd.Series(close.to_numpy(), index=frame["date"], name=name))
            error = ""
        except (requests.RequestException, OSError, ValueError, KeyError) as exc:
            payload = b""
            status = "failed"
            error = str(exc)
        inventory.append(
            {
                "source": f"cboe_{name}",
                "year": "",
                "url": url,
                "status": status,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest() if payload else "",
                "error": error,
            }
        )
    panel = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
    panel = panel.loc[~panel.index.isna() & ~panel.index.duplicated(keep="last")]
    return panel, pd.DataFrame(inventory)


def acquire_finra_margin(
    raw_dir: Path,
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = raw_dir / "finra_margin_statistics.xlsx"
    try:
        payload, status = _cached_get(FINRA_MARGIN_URL, path, refresh=refresh)
        frame = pd.read_excel(io.BytesIO(payload))
        frame.columns = [
            "reference_month",
            "debit_balances",
            "cash_free_credit",
            "margin_free_credit",
        ]
        frame["reference_month"] = pd.to_datetime(
            frame["reference_month"].astype(str), format="%Y-%m", errors="coerce"
        )
        frame = frame.dropna(subset=["reference_month"]).sort_values("reference_month")
        for column in ("debit_balances", "cash_free_credit", "margin_free_credit"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["available_date"] = frame["reference_month"].map(finra_approximate_release_date)
        frame["credit_total"] = frame["cash_free_credit"].fillna(0.0) + frame[
            "margin_free_credit"
        ].fillna(0.0)
        frame["debit_to_credit"] = frame["debit_balances"] / frame["credit_total"].replace(
            0.0, np.nan
        )
        frame = frame.set_index("available_date")
        error = ""
    except (requests.RequestException, OSError, ValueError, KeyError) as exc:
        payload = b""
        status = "failed"
        error = str(exc)
        frame = pd.DataFrame()
    inventory = pd.DataFrame(
        [
            {
                "source": "finra_margin",
                "year": "",
                "url": FINRA_MARGIN_URL,
                "status": status,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest() if payload else "",
                "error": error,
            }
        ]
    )
    return frame, inventory


def finra_approximate_release_date(reference_month: object) -> pd.Timestamp:
    month = pd.Timestamp(reference_month)
    following = month + pd.offsets.MonthBegin(1)
    first = pd.Timestamp(following.year, following.month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=days_to_friday + 14)


def build_cftc_candidates(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
) -> list[SignalCandidate]:
    if panel.empty:
        return []
    index = prices.index
    net = _weekly_feature(panel, "net_position", index)
    change = _weekly_feature(panel, "net_change_4w", index)
    concentration = _weekly_feature(panel, "concentration", index)
    net_z = _rolling_z(net, 156, 52)
    change_z = _rolling_z(change, 104, 39)
    concentration_z = _rolling_z(concentration, 156, 52)
    available = [ticker for ticker in net.columns if ticker in prices.columns]
    net_z = net_z[available]
    change_z = change_z[available]
    concentration_z = concentration_z[available]
    price_momentum = prices[available].pct_change(63, fill_method=None)
    score_map = {
        "level_trend": net_z,
        "change_trend": change_z,
        "contrarian_level": -net_z,
        "balanced": 0.55 * net_z + 0.45 * change_z - 0.20 * concentration_z,
        "washout_reversal": -net_z + 0.50 * (price_momentum > 0.0).astype(float),
        "deconcentration": net_z - 0.50 * concentration_z,
    }
    candidates: list[SignalCandidate] = []
    for style, scores in score_map.items():
        for top_n in (2, 3):
            for minimum in (-0.5, 0.0):
                weights = _cross_sectional_weights(scores, top_n=top_n, minimum=minimum)
                candidates.append(
                    SignalCandidate(
                        name=f"cftc_{style}_n{top_n}_m{minimum:+.1f}",
                        family="cftc_positioning",
                        hypothesis=(
                            "Use causally available CFTC positioning level, change, and concentration "
                            "to select long-only ETF exposures."
                        ),
                        target_weights=weights,
                        signal_lag_days=1,
                    )
                )
    return candidates


def build_options_candidates(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
) -> list[SignalCandidate]:
    required = {"vix", "vix3m", "skew"}
    if panel.empty or not required.issubset(panel.columns):
        return []
    features = panel.reindex(prices.index).ffill(limit=5)
    term = features["vix"] / features["vix3m"].replace(0.0, np.nan)
    realized = prices["SPY"].pct_change(fill_method=None).rolling(21).std() * np.sqrt(252) * 100
    premium = features["vix"] - realized
    term_z = _rolling_z(term.to_frame("term"), 504, 126)["term"]
    vix_z = _rolling_z(features[["vix"]], 504, 126)["vix"]
    skew_z = _rolling_z(features[["skew"]], 504, 126)["skew"]
    premium_z = _rolling_z(premium.to_frame("premium"), 504, 126)["premium"]
    composite = -0.45 * term_z - 0.30 * vix_z - 0.15 * skew_z + 0.10 * premium_z
    rules: dict[str, pd.Series] = {}
    for threshold in (0.90, 0.95, 1.00, 1.05):
        rules[f"term_below_{threshold:.2f}"] = term < threshold
    for threshold in (-0.5, 0.0, 0.5):
        rules[f"composite_above_{threshold:+.1f}"] = composite > threshold
    rules["contango_not_high_vix"] = (term < 1.0) & (vix_z < 1.0)
    rules["contango_or_fear_premium"] = (term < 0.98) | ((premium_z > 1.0) & (term_z < 0.5))
    rules["no_joint_tail_warning"] = ~((term_z > 1.0) & (skew_z > 1.0))
    candidates: list[SignalCandidate] = []
    for asset in ("SPY", "QQQ"):
        for rule_name, risk_on in rules.items():
            for partial in (0.0, 0.5):
                risk_weight = pd.Series(np.where(risk_on, 1.0, partial), index=prices.index)
                weights = pd.DataFrame({asset: risk_weight, "BIL": 1.0 - risk_weight})
                candidates.append(
                    SignalCandidate(
                        name=f"options_{asset.lower()}_{rule_name}_floor{int(partial * 100)}",
                        family="options_implied_state",
                        hypothesis=(
                            "Use Cboe term structure, SKEW, and implied-realized volatility to "
                            "gate long-only equity exposure."
                        ),
                        target_weights=weights,
                        signal_lag_days=2,
                    )
                )
    return candidates


def build_margin_candidates(
    prices: pd.DataFrame,
    margin: pd.DataFrame,
) -> list[SignalCandidate]:
    if margin.empty:
        return []
    monthly = margin.copy()
    debit_growth_3m = monthly["debit_balances"].pct_change(3, fill_method=None)
    debit_growth_12m = monthly["debit_balances"].pct_change(12, fill_method=None)
    leverage_z = _rolling_z(monthly[["debit_to_credit"]], 60, 24)["debit_to_credit"]
    monthly_rules = {
        "growth3_positive": debit_growth_3m > 0.0,
        "growth3_not_euphoric": (debit_growth_3m > -0.05) & (leverage_z < 1.5),
        "growth12_positive": debit_growth_12m > 0.0,
        "leverage_below_z1": leverage_z < 1.0,
        "deleveraging_contrarian": (debit_growth_3m < 0.0) & (leverage_z < 0.5),
        "balanced": (debit_growth_12m > 0.0) & (leverage_z < 1.5),
    }
    candidates: list[SignalCandidate] = []
    for asset in ("SPY", "QQQ"):
        trend = prices[asset] > prices[asset].rolling(200, min_periods=150).mean()
        for rule_name, monthly_rule in monthly_rules.items():
            daily_rule = monthly_rule.astype("boolean").reindex(
                prices.index.union(monthly_rule.index)
            )
            daily_rule = daily_rule.ffill().reindex(prices.index).fillna(False).astype(bool)
            variants = {"pure": daily_rule, "trend_confirmed": daily_rule & trend}
            for suffix, risk_on in variants.items():
                risk_weight = risk_on.fillna(False).astype(float)
                weights = pd.DataFrame({asset: risk_weight, "BIL": 1.0 - risk_weight})
                candidates.append(
                    SignalCandidate(
                        name=f"margin_{asset.lower()}_{rule_name}_{suffix}",
                        family="margin_crowding",
                        hypothesis=(
                            "Use FINRA customer margin balances, with a conservative publication lag, "
                            "as a long-only leverage and crowding state."
                        ),
                        target_weights=weights,
                        signal_lag_days=2,
                    )
                )
    return candidates


def select_family_winners(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    selection = candidate_metrics[
        candidate_metrics["period"].isin(["development", "validation"])
    ].copy()
    pivot = selection.pivot_table(
        index=["candidate", "family"],
        columns="period",
        values=["cagr", "sharpe", "max_drawdown"],
    )
    rows: list[dict[str, object]] = []
    for (candidate, family), row in pivot.iterrows():
        required = [
            ("cagr", "development"),
            ("cagr", "validation"),
            ("sharpe", "development"),
            ("sharpe", "validation"),
        ]
        if any(pd.isna(row.get(key)) for key in required):
            continue
        score = (
            0.50 * min(float(row[("cagr", "development")]), float(row[("cagr", "validation")]))
            + 0.25 * float(row[("sharpe", "development")])
            + 0.25 * float(row[("sharpe", "validation")])
        )
        rows.append({"candidate": candidate, "family": family, "selection_score": score})
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    winners = (
        ranked.sort_values(
            ["family", "selection_score", "candidate"], ascending=[True, False, True]
        )
        .groupby("family", as_index=False)
        .head(1)
    )
    metric_columns = [
        column
        for column in (
            "cagr",
            "sharpe",
            "max_drawdown",
            "average_risk_exposure",
            "average_turnover",
        )
        if column in candidate_metrics.columns
    ]
    metrics = candidate_metrics.pivot_table(
        index="candidate",
        columns="period",
        values=metric_columns,
    )
    output: list[dict[str, object]] = []
    for _, winner in winners.iterrows():
        name = str(winner["candidate"])
        metric = metrics.loc[name]
        output.append(
            {
                **winner.to_dict(),
                "development_cagr": metric.get(("cagr", "development"), np.nan),
                "validation_cagr": metric.get(("cagr", "validation"), np.nan),
                "holdout_cagr": metric.get(("cagr", "holdout"), np.nan),
                "forward_cagr": metric.get(("cagr", "forward"), np.nan),
                "common_cagr": metric.get(("cagr", "common_full"), np.nan),
                "common_sharpe": metric.get(("sharpe", "common_full"), np.nan),
                "common_max_drawdown": metric.get(("max_drawdown", "common_full"), np.nan),
                "common_risk_exposure": metric.get(
                    ("average_risk_exposure", "common_full"), np.nan
                ),
                "common_turnover": metric.get(("average_turnover", "common_full"), np.nan),
            }
        )
    return pd.DataFrame(output).sort_values("family")


def add_edge_reads(
    selection: pd.DataFrame,
    control_metrics: pd.DataFrame,
    *,
    benchmark: str = "spy_buy_hold",
) -> pd.DataFrame:
    if selection.empty:
        return selection
    controls = control_metrics.pivot(index="control", columns="period")
    if benchmark not in controls.index:
        output = selection.copy()
        output["edge_read"] = "benchmark_missing"
        return output
    reference = controls.loc[benchmark]
    balanced = controls.loc["balanced_60_40"] if "balanced_60_40" in controls.index else None
    rows: list[dict[str, object]] = []
    for _, row in selection.iterrows():
        beats_both = float(row["holdout_cagr"]) > float(
            reference.get(("cagr", "holdout"), np.inf)
        ) and float(row["forward_cagr"]) > float(reference.get(("cagr", "forward"), np.inf))
        sharpe_edge = float(row["common_sharpe"]) > float(
            reference.get(("sharpe", "common_full"), np.inf)
        )
        drawdown_improvement = float(row["common_max_drawdown"]) - float(
            reference.get(("max_drawdown", "common_full"), -1.0)
        )
        balanced_sharpe = (
            float(balanced.get(("sharpe", "common_full"), np.inf))
            if balanced is not None
            else np.inf
        )
        if beats_both and sharpe_edge:
            edge_read = "promising_return_edge"
        elif drawdown_improvement >= 0.10 and float(row["common_sharpe"]) >= balanced_sharpe:
            edge_read = "risk_diversifier"
        else:
            edge_read = "no_robust_edge"
        rows.append({**row.to_dict(), "edge_read": edge_read})
    return pd.DataFrame(rows)


def build_trade_bot_overlay_metrics(
    candidates: list[SignalCandidate],
    results: dict[str, BacktestResult],
    prices: pd.DataFrame,
    controls: dict[str, pd.Series],
) -> pd.DataFrame:
    if "trade_bot_primary" not in controls:
        return pd.DataFrame()
    trade_bot = controls["trade_bot_primary"].reindex(prices.index).fillna(0.0)
    bil = prices["BIL"].pct_change(fill_method=None).fillna(0.0)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        weights = results[candidate.name].weights.reindex(prices.index).fillna(0.0)
        exposure = (1.0 - weights.get("BIL", pd.Series(0.0, index=weights.index))).clip(0.0, 1.0)
        overlay_turnover = exposure.diff().abs().fillna(exposure.abs())
        overlay_returns = (
            exposure * trade_bot + (1.0 - exposure) * bil - overlay_turnover * 5.0 / 10_000.0
        )
        for period, (start, end) in PERIODS.items():
            period_returns = _period_slice(overlay_returns, start, end)
            period_index = period_returns.index
            rows.append(
                {
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "hypothesis": f"Gate Trade Bot with {candidate.name} executable exposure.",
                    "period": period,
                    **_metric_row(period_returns),
                    "average_risk_exposure": float(exposure.reindex(period_index).mean()),
                    "average_turnover": float(overlay_turnover.reindex(period_index).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_combination_returns(
    candidates: list[SignalCandidate],
    results: dict[str, BacktestResult],
    family_selection: pd.DataFrame,
    prices: pd.DataFrame,
    controls: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    if family_selection.empty:
        return {}
    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    winners = [str(value) for value in family_selection["candidate"]]
    risk_exposures: list[pd.Series] = []
    for name in winners:
        weights = results[name].weights
        risk_exposures.append(
            (1.0 - weights.get("BIL", pd.Series(0.0, index=weights.index))).clip(0, 1)
        )
    consensus = pd.concat(risk_exposures, axis=1).mean(axis=1).reindex(prices.index).fillna(0.0)
    execution = ExecutionConfig(
        initial_capital=100_000.0,
        transaction_cost_bps=5.0,
        rebalance="daily",
        signal_lag_days=1,
    )
    combinations: dict[str, pd.Series] = {}
    for asset in ("SPY", "QQQ"):
        weights = pd.DataFrame({asset: consensus, "BIL": 1.0 - consensus}, index=prices.index)
        combinations[f"all_signal_consensus_{asset.lower()}"] = run_backtest(
            f"all_signal_consensus_{asset.lower()}", prices[[asset, "BIL"]], weights, execution
        ).returns
    if "trade_bot_primary" in controls:
        aligned = pd.concat(
            [controls["trade_bot_primary"], combinations["all_signal_consensus_spy"]], axis=1
        ).fillna(0.0)
        combinations["half_trade_bot_half_signal_consensus"] = aligned.mean(axis=1)
    for first, second in (
        ("cftc_positioning", "options_implied_state"),
        ("cftc_positioning", "margin_crowding"),
        ("options_implied_state", "margin_crowding"),
    ):
        selected = family_selection[family_selection["family"].isin([first, second])]
        if len(selected) != 2:
            continue
        pair = [candidate_by_name[str(name)] for name in selected["candidate"]]
        pair_results = [results[candidate.name] for candidate in pair]
        exposure = (
            pd.concat(
                [
                    1.0 - result.weights.get("BIL", pd.Series(0.0, index=result.weights.index))
                    for result in pair_results
                ],
                axis=1,
            )
            .mean(axis=1)
            .clip(0, 1)
        )
        weights = (
            pd.DataFrame({"SPY": exposure, "BIL": 1.0 - exposure}, index=exposure.index)
            .reindex(prices.index)
            .fillna({"SPY": 0.0, "BIL": 1.0})
        )
        name = f"pair_{first}_plus_{second}"
        combinations[name] = run_backtest(name, prices[["SPY", "BIL"]], weights, execution).returns
    return combinations


def build_source_feasibility(
    raw_dir: Path,
    *,
    cftc_inventory: pd.DataFrame,
    options_inventory: pd.DataFrame,
    margin_inventory: pd.DataFrame,
) -> pd.DataFrame:
    def status(frame: pd.DataFrame) -> str:
        return (
            "acquired"
            if not frame.empty and frame["status"].isin(["downloaded", "cached"]).all()
            else "incomplete"
        )

    sec_raw = Path("research/capital_cycle/raw/sec_companyfacts")
    sec_files = list(sec_raw.glob("CIK*.json")) if sec_raw.exists() else []
    rows = [
        {
            "frontier": "cftc_positioning",
            "status": status(cftc_inventory),
            "backtest_authority": "retrospective_research",
            "timestamp_quality": "high",
            "coverage": f"{len(cftc_inventory)} official annual archives",
            "constraint": "Historical trader classifications are backcast by CFTC and weaken backward in time.",
        },
        {
            "frontier": "options_implied_state",
            "status": status(options_inventory),
            "backtest_authority": "retrospective_research",
            "timestamp_quality": "high",
            "coverage": "VIX, VIX3M, and SKEW official daily histories",
            "constraint": "Historical public put-call bulk series was not admitted; no paid DataShop proxy.",
        },
        {
            "frontier": "margin_crowding",
            "status": status(margin_inventory),
            "backtest_authority": "diagnostic_only",
            "timestamp_quality": "medium",
            "coverage": "FINRA monthly balances from 1997",
            "constraint": "Exact historical publication dates are unavailable; third-Friday lag is conservative but approximate.",
        },
        {
            "frontier": "fund_flows_short_interest",
            "status": "blocked",
            "backtest_authority": "none",
            "timestamp_quality": "unverified",
            "coverage": "No stable free point-in-time ETF-flow and security-level short-interest archive",
            "constraint": "Do not substitute price crowding for the requested non-price information.",
        },
        {
            "frontier": "sec_fundamental_events",
            "status": "blocked" if not sec_files else "local_archive_available",
            "backtest_authority": "none" if not sec_files else "diagnostic_only",
            "timestamp_quality": "high_when_acquired",
            "coverage": f"{len(sec_files)} governed local Company Facts files",
            "constraint": (
                "Official SEC issuer and bulk endpoints returned HTTP 403 in this environment; "
                "the existing filing-date importer remains ready but no backtest is fabricated."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _weekly_feature(panel: pd.DataFrame, column: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    pivot = panel.pivot(index="available_date", columns="ticker", values=column).sort_index()
    expanded = pivot.reindex(index.union(pivot.index)).sort_index().ffill()
    return expanded.reindex(index)


def _rolling_z(frame: pd.DataFrame, window: int, minimum: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=minimum).mean()
    std = frame.rolling(window, min_periods=minimum).std().replace(0.0, np.nan)
    return (frame - mean) / std


def _cross_sectional_weights(
    scores: pd.DataFrame,
    *,
    top_n: int,
    minimum: float,
) -> pd.DataFrame:
    ranks = scores.rank(axis=1, ascending=False, method="first")
    selected = ranks.le(top_n) & scores.ge(minimum)
    count = selected.sum(axis=1).replace(0, np.nan)
    weights = selected.astype(float).div(count, axis=0).fillna(0.0)
    weights["BIL"] = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    return weights


def _run_candidate(
    candidate: SignalCandidate,
    prices: pd.DataFrame,
    *,
    transaction_cost_bps: float,
) -> BacktestResult:
    tickers = list(candidate.target_weights.columns)
    execution = ExecutionConfig(
        initial_capital=100_000.0,
        transaction_cost_bps=transaction_cost_bps,
        rebalance="daily",
        signal_lag_days=candidate.signal_lag_days,
    )
    return run_backtest(
        candidate.name,
        prices[tickers],
        candidate.target_weights,
        execution,
    )


def _candidate_metric_frame(
    candidates: list[SignalCandidate],
    results: dict[str, BacktestResult],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        for period, (start, end) in PERIODS.items():
            period_returns = _period_slice(results[candidate.name].returns, start, end)
            period_index = period_returns.index
            weights = results[candidate.name].weights.reindex(period_index)
            turnover = results[candidate.name].turnover.reindex(period_index)
            rows.append(
                {
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "hypothesis": candidate.hypothesis,
                    "period": period,
                    **_metric_row(period_returns),
                    "average_risk_exposure": float(
                        (1.0 - weights.get("BIL", pd.Series(0.0, index=weights.index))).mean()
                    ),
                    "average_turnover": float(turnover.mean()),
                }
            )
    return pd.DataFrame(rows)


def _return_metric_frame(
    returns_by_name: dict[str, pd.Series],
    *,
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, returns in returns_by_name.items():
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    label: name,
                    "period": period,
                    **_metric_row(_period_slice(returns, start, end)),
                }
            )
    return pd.DataFrame(rows)


def _cost_stress(
    candidates: list[SignalCandidate],
    selection: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    lookup = {candidate.name: candidate for candidate in candidates}
    rows: list[dict[str, object]] = []
    for _, winner in selection.iterrows():
        candidate = lookup[str(winner["candidate"])]
        for cost in (5.0, 10.0, 25.0):
            result = _run_candidate(candidate, prices, transaction_cost_bps=cost)
            rows.append(
                {
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "cost_bps": cost,
                    **_metric_row(_period_slice(result.returns, COMMON_START, None)),
                }
            )
    return pd.DataFrame(rows)


def _cached_get(url: str, path: Path, *, refresh: bool) -> tuple[bytes, str]:
    if path.exists() and not refresh:
        return path.read_bytes(), "cached"
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload, "downloaded"


def _load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(path).sort_index()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index)).tz_localize(None)
    return prices.loc[~prices.index.duplicated(keep="last")]


def _render_report(
    selection: pd.DataFrame,
    overlay_selection: pd.DataFrame,
    combinations: pd.DataFrame,
    controls: pd.DataFrame,
    feasibility: pd.DataFrame,
    cost_stress: pd.DataFrame,
    *,
    variants: int,
    data_end: pd.Timestamp,
) -> str:
    lines = [
        "# Public Non-Price Signal Frontiers",
        "",
        f"Evidence through {data_end.date()}; {variants} fixed-grid variants across three testable families.",
        "",
        "Winners use only 2012-2019 development/validation evidence. The 2020-2022 holdout and 2023+ forward windows are reported after selection, but remain retrospective rather than live OOS evidence.",
        "",
        "## Family Winners",
        "",
        "| family | selected candidate | dev CAGR | validation CAGR | holdout CAGR | forward CAGR | full CAGR | Sharpe | max DD | risk exposure | turnover | read |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in selection.iterrows():
        lines.append(
            "| {family} | `{candidate}` | {development_cagr:.1%} | {validation_cagr:.1%} | "
            "{holdout_cagr:.1%} | {forward_cagr:.1%} | {common_cagr:.1%} | "
            "{common_sharpe:.2f} | {common_max_drawdown:.1%} | "
            "{common_risk_exposure:.1%} | {common_turnover:.1%} | {edge_read} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Trade Bot Gating Test",
            "",
            "Each selected non-price signal controls how much of the existing Trade Bot return stream is retained, with the remainder in BIL. Selection still uses only 2012-2019 evidence; the verdict benchmark is unmodified Trade Bot.",
            "",
            "| family | selected gate | holdout CAGR | forward CAGR | full CAGR | Sharpe | max DD | exposure | read |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in overlay_selection.iterrows():
        lines.append(
            "| {family} | `{candidate}` | {holdout_cagr:.1%} | {forward_cagr:.1%} | "
            "{common_cagr:.1%} | {common_sharpe:.2f} | {common_max_drawdown:.1%} | "
            "{common_risk_exposure:.1%} | {edge_read} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Combinations",
            "",
            "| combination | holdout CAGR | forward CAGR | full CAGR | Sharpe | max DD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not combinations.empty:
        pivot = combinations.pivot(index="combination", columns="period")
        for name in pivot.index:
            row = pivot.loc[name]
            lines.append(
                f"| `{name}` | {float(row.get(('cagr', 'holdout'), np.nan)):.1%} | "
                f"{float(row.get(('cagr', 'forward'), np.nan)):.1%} | "
                f"{float(row.get(('cagr', 'common_full'), np.nan)):.1%} | "
                f"{float(row.get(('sharpe', 'common_full'), np.nan)):.2f} | "
                f"{float(row.get(('max_drawdown', 'common_full'), np.nan)):.1%} |"
            )
    lines.extend(
        [
            "",
            "## Same-Period Controls",
            "",
            "| control | holdout CAGR | forward CAGR | full CAGR | Sharpe | max DD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not controls.empty:
        pivot = controls.pivot(index="control", columns="period")
        for name in pivot.index:
            row = pivot.loc[name]
            lines.append(
                f"| `{name}` | {float(row.get(('cagr', 'holdout'), np.nan)):.1%} | "
                f"{float(row.get(('cagr', 'forward'), np.nan)):.1%} | "
                f"{float(row.get(('cagr', 'common_full'), np.nan)):.1%} | "
                f"{float(row.get(('sharpe', 'common_full'), np.nan)):.2f} | "
                f"{float(row.get(('max_drawdown', 'common_full'), np.nan)):.1%} |"
            )
    lines.extend(
        [
            "",
            "## Source Gates",
            "",
            "| frontier | status | authority | timestamp quality | constraint |",
            "|---|---|---|---|---|",
        ]
    )
    for _, row in feasibility.iterrows():
        lines.append(
            f"| {row['frontier']} | {row['status']} | {row['backtest_authority']} | "
            f"{row['timestamp_quality']} | {row['constraint']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- CFTC observations are Tuesday positions made available Friday; weights execute no earlier than the next modeled session.",
            "- CFTC itself warns that historical trader classifications are backcast and less accurate farther back in time.",
            "- FINRA margin release dates are conservatively approximated, so that family remains diagnostic only.",
            "- SEC event work remains blocked rather than replaced with a survivor-biased current-stock sample.",
            "- No result is promoted. A useful family must survive the untouched windows, cost stress, and a new prospective freeze.",
        ]
    )
    if not cost_stress.empty:
        worst = cost_stress.loc[cost_stress["cost_bps"].eq(25.0)].sort_values("cagr")
        lines.extend(
            [
                "",
                "## Cost Stress",
                "",
                f"At 25 bps, family-winner CAGR ranged from {worst['cagr'].min():.1%} to {worst['cagr'].max():.1%}.",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test official public non-price signal frontiers.")
    parser.add_argument("--prices", default="data/cache/yahoo_prices.parquet")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--raw", default="research/public_signal_frontiers/raw")
    parser.add_argument("--output", default="reports/public_signal_frontiers")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_public_signal_frontier_survey(
        prices_path=args.prices,
        config_path=args.config,
        raw_dir=args.raw,
        output_dir=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        refresh=args.refresh,
    )
    print(result.report_path)


if __name__ == "__main__":
    main()
