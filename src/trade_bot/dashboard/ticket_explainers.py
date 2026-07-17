from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from trade_bot import DEFAULTS as defaults
from trade_bot.dashboard.instrument_registry import instrument_profile


@dataclass(frozen=True)
class TicketExplainer:
    term: str
    category: str
    plain_english: str
    calculation: str
    how_to_read: str
    caution: str
    kind: str = "Ticket"
    aliases: tuple[str, ...] = ()


TICKET_EXPLAINERS: tuple[TicketExplainer, ...] = (
    TicketExplainer(
        term="Recommendation Ticket",
        category="Forward Test",
        kind="Workflow",
        plain_english=(
            "A locked paper or live instruction generated from the current trade decision for one ticker."
        ),
        calculation=(
            "Built from target weight, current weight, account value, reference price, price band, "
            "and size band."
        ),
        how_to_read=(
            "Treat it as an auditable recommendation to review, execute, skip, or expire."
        ),
        caution=(
            "A ticket freezes the recommendation context at creation time; refresh before acting on stale tickets."
        ),
        aliases=("ticket", "recommendation_tickets", "locked ticket", "trade ticket"),
    ),
    TicketExplainer(
        term="Open Ticket",
        category="Forward Test",
        kind="Workflow",
        plain_english="A recommendation ticket that has not been executed, skipped, or expired.",
        calculation="Ticket status equals open in the local trade journal.",
        how_to_read="Open tickets are pending review before the next execution window.",
        caution="Multiple open tickets for the same account or ticker can mean the book is not reconciled.",
        aliases=("open_ticket_count", "open tickets", "pending ticket"),
    ),
    TicketExplainer(
        term="Forward Test",
        category="Forward Test",
        kind="Workflow",
        plain_english=(
            "The paper/live workflow that locks recommendations, logs executions, and audits what happened."
        ),
        calculation="Uses local tickets, executions, book alignment, and valuations stored in the journal.",
        how_to_read=(
            "Use this before live money: lock the recommendation, record exact fills, then compare forward results."
        ),
        caution=(
            "Backtests and forward tests answer different questions; forward tests include human timing and execution misses."
        ),
        aliases=("paper test", "paper trading", "trade journal", "journal"),
    ),
    TicketExplainer(
        term="Book Alignment",
        category="Forward Test",
        kind="Workflow",
        plain_english="Comparison between the logged current book and the latest target posture.",
        calculation="Current logged holdings by ticker minus target weights from the trade-decision layer.",
        how_to_read="Small drift means minor cleanup; large drift means current holdings differ materially from target.",
        caution="The result is only as good as the logged executions and latest prices.",
        aliases=("book-aware recommendation", "book drift", "alignment_status"),
    ),
    TicketExplainer(
        term="Ticket ID",
        category="Ticket Field",
        plain_english="Unique local identifier for one locked recommendation ticket.",
        calculation="Generated UUID when the recommendation set is locked.",
        how_to_read="Use it to connect a recommendation to an execution or status change.",
        caution="A short ticket label is convenient, but the full ID is the audit key.",
        aliases=("ticket_id", "recommendation_id"),
    ),
    TicketExplainer(
        term="Decision ID",
        category="Ticket Field",
        plain_english="Identifier for the decision snapshot that produced one or more tickets.",
        calculation="Generated UUID when the full recommendation set is locked.",
        how_to_read="Use it to trace a ticket back to the risk status, scenario map, and target posture that created it.",
        caution="If you refresh the dashboard, a new decision can supersede an older decision ID.",
        aliases=("decision_id", "snapshot id"),
    ),
    TicketExplainer(
        term="Status",
        category="Ticket Field",
        plain_english="Ticket lifecycle state: open, executed, skipped, or expired.",
        calculation="Stored status in the local recommendation_tickets table.",
        how_to_read="Open needs review; executed is acted on; skipped is intentionally ignored; expired is stale.",
        caution="Status is human-maintained unless an execution is explicitly linked to the ticket.",
        aliases=("ticket status", "open", "executed", "skipped", "expired"),
    ),
    TicketExplainer(
        term="Mode",
        category="Ticket Field",
        plain_english="Whether the ticket or execution belongs to paper tracking or live tracking.",
        calculation="Stored as the selected mode when locking tickets or logging executions.",
        how_to_read="Paper is simulation; live is real execution tracking.",
        caution="Do not mix paper and live records in the same account label if you want clean audit trails.",
        aliases=("paper", "live", "journal_mode"),
    ),
    TicketExplainer(
        term="Account",
        category="Ticket Field",
        plain_english="Local account label used to separate books, sleeves, or monitoring runs.",
        calculation="Stored as the selected account when locking tickets or logging executions.",
        how_to_read="Use different labels for different paper books, live accounts, or strategy sleeves.",
        caution="This is not broker authentication; it is a local grouping key.",
        aliases=("account label", "journal_account"),
    ),
    TicketExplainer(
        term="Strategy Name",
        category="Ticket Field",
        plain_english="Strategy or operating-system label that generated a ticket.",
        calculation="Stored from the selected strategy label when tickets are locked.",
        how_to_read="Use it to separate champion, challenger, and reference recommendations.",
        caution="A generic label can make later performance attribution harder.",
        aliases=("strategy_name", "strategy label"),
    ),
    TicketExplainer(
        term="Ticker",
        category="Ticket Field",
        plain_english="Tradable stock or ETF symbol for the ticket or execution.",
        calculation="Copied from the position plan or execution form.",
        how_to_read="This is the security that would be bought, sold, tracked, or valued.",
        caution="A ticker can be used as a risk proxy without being an allowed holding; check the strategy context.",
        aliases=("symbol", "asset"),
    ),
    TicketExplainer(
        term="Side",
        category="Ticket Field",
        plain_english="Execution direction: BUY adds shares; SELL trims an existing long position.",
        calculation="Positive target notional becomes BUY; negative target notional becomes SELL.",
        how_to_read="SELL means reduce or exit a long position. The system is not designed to short.",
        caution="Side is broker-facing; source action explains the model-level target change.",
        aliases=("BUY", "SELL", "execution side"),
    ),
    TicketExplainer(
        term="Source Action",
        category="Ticket Field",
        plain_english="Model-level direction before broker translation: ADD or REDUCE.",
        calculation="Derived from the sign of target weight minus current weight.",
        how_to_read="ADD usually maps to BUY; REDUCE usually maps to SELL.",
        caution="Source action does not include fill quality, taxes, or whether the ticket was actually executed.",
        aliases=("source_action", "ADD", "REDUCE", "action"),
    ),
    TicketExplainer(
        term="Current Weight",
        category="Ticket Field",
        plain_english="Current model or logged account weight used when building the ticket.",
        calculation="Current ticker value divided by account value, or the assumed current weight in the decision plan.",
        how_to_read="This is the starting point before applying the target posture.",
        caution="If logged executions are incomplete, current weight can be wrong.",
        aliases=("current_weight", "current position"),
    ),
    TicketExplainer(
        term="Target Weight",
        category="Ticket Field",
        plain_english="Scenario- and risk-adjusted portfolio weight the ticket is trying to move toward.",
        calculation="Final target from the trade-decision layer after scenario and risk constraints.",
        how_to_read="This is the desired destination weight, not a guarantee that the trade should be executed immediately.",
        caution="Targets can change after a new daily update or a material market move.",
        aliases=("target_weight", "scenario_adjusted_weight", "target posture"),
    ),
    TicketExplainer(
        term="Delta Weight",
        category="Ticket Field",
        plain_english="Difference between target weight and current weight.",
        calculation="Target weight minus current weight.",
        how_to_read="Positive means add exposure; negative means reduce exposure.",
        caution="Small deltas may be ignored by minimum trade size or human execution bands.",
        aliases=("delta_weight", "target change", "max target change"),
    ),
    TicketExplainer(
        term="Reference Price",
        category="Ticket Field",
        plain_english="Latest available price used when the ticket was created.",
        calculation="Most recent non-missing price from the loaded market data.",
        how_to_read="Use it as the anchor for share counts and price bands.",
        caution="It is not a live quote or guaranteed fill price.",
        aliases=("reference_price", "ticket price"),
    ),
    TicketExplainer(
        term="Price Band",
        category="Ticket Field",
        plain_english="Acceptable review range around the ticket reference price.",
        calculation="Reference price times one plus or minus the configured price-band percentage.",
        how_to_read="If the market price moves outside the band, refresh or review before execution.",
        caution="A band is a human review guardrail, not a broker limit order by itself.",
        aliases=("price band", "limit_low", "limit_high", "price_low", "price_high"),
    ),
    TicketExplainer(
        term="Target Notional",
        category="Ticket Field",
        plain_english="Dollar value implied by the requested allocation change.",
        calculation="Delta weight times account value.",
        how_to_read="Positive is buy notional; negative is sell notional before side translation.",
        caution="Actual execution can differ because of size bands, whole shares, prices, and skipped tickets.",
        aliases=("target_notional", "trade dollars"),
    ),
    TicketExplainer(
        term="Size Band",
        category="Ticket Field",
        plain_english="Allowed dollar-size range around the target notional.",
        calculation="Absolute target notional times one plus or minus the configured size-band percentage.",
        how_to_read="Lets a human execute a practical size without pretending the model is penny-exact.",
        caution="Too-wide bands can weaken auditability; too-tight bands can make execution annoying.",
        aliases=("size band", "min_notional", "max_notional", "dollar_low", "dollar_high"),
    ),
    TicketExplainer(
        term="Share Range",
        category="Ticket Field",
        plain_english="Suggested low/high share quantity for the ticket.",
        calculation="Min and max notional divided by reference price, optionally rounded to whole shares.",
        how_to_read="Use it as an execution range after checking price and account constraints.",
        caution="Share counts can become stale quickly when prices move.",
        aliases=("min_shares", "max_shares", "share_low", "share_high"),
    ),
    TicketExplainer(
        term="Rationale",
        category="Ticket Field",
        plain_english="Human-readable reason attached to the ticket when it was created.",
        calculation="Copied from the book-alignment explanation or trade-decision explanation.",
        how_to_read="Read this before executing so the trade is tied back to the model's evidence.",
        caution="Rationale is a snapshot; it does not update when new evidence arrives.",
        aliases=("rationale", "reason", "ticket rationale"),
    ),
    TicketExplainer(
        term="Execution",
        category="Execution Field",
        kind="Workflow",
        plain_english="A logged paper or live fill with ticker, side, quantity, price, timestamp, and notes.",
        calculation="Manual form entry saved to the local executions table.",
        how_to_read="Executions are the audit trail used for book alignment, monitoring, and taxable-lot reconstruction.",
        caution="If you do not log the real fill, forward performance will diverge from actual results.",
        aliases=("execution_id", "logged execution", "fill"),
    ),
    TicketExplainer(
        term="Quantity",
        category="Execution Field",
        plain_english="Number of shares recorded in an execution.",
        calculation="Manual execution input, or selected from a ticket's suggested share range.",
        how_to_read="Quantity times price gives gross notional before fees.",
        caution="Fractional share support depends on the selected workflow and broker reality.",
        aliases=("quantity", "shares"),
    ),
    TicketExplainer(
        term="Executed Price",
        category="Execution Field",
        plain_english="Actual paper or live fill price recorded for an execution.",
        calculation="Manual execution input saved as price.",
        how_to_read="Compare it to the ticket reference price and price band to inspect execution shortfall.",
        caution="Paper prices should be realistic; optimistic fills overstate forward performance.",
        aliases=("price", "executed_price", "fill price"),
    ),
    TicketExplainer(
        term="Execution Notes",
        category="Execution Field",
        plain_english="Optional notes saved with an execution.",
        calculation="Manual text field in the execution log.",
        how_to_read="Use notes to capture discretion, skipped constraints, or why a fill differed from the ticket.",
        caution="Notes are local audit context; they are not used as model inputs unless explicitly added later.",
        aliases=("notes", "execution notes"),
    ),
    TicketExplainer(
        term="SPY",
        category="Ticker",
        kind="Ticker",
        plain_english="SPDR S&P 500 ETF Trust, used here as the broad U.S. equity benchmark and beta anchor.",
        calculation="Pulled as a price series and used in baselines, beta-adjusted delta, and relative comparisons.",
        how_to_read="SPY is the main broad-market reference for whether strategies beat simple U.S. equity exposure.",
        caution="SPY is cap-weighted and can still be concentrated in mega-cap themes.",
        aliases=("S&P 500", "S&P500", "market beta", "broad equity"),
    ),
    TicketExplainer(
        term="QQQ",
        category="Ticker",
        kind="Ticker",
        plain_english="Invesco QQQ Trust, used as a Nasdaq-100 and mega-cap growth/AI-beta benchmark.",
        calculation="Pulled as a price series and used in baselines, growth comparisons, and factor attribution.",
        how_to_read="QQQ is the high-growth hurdle: strategies should justify any lower return with materially better risk.",
        caution="QQQ can be heavily concentrated in technology and AI-linked leadership.",
        aliases=("Nasdaq 100", "growth benchmark", "QQQM"),
    ),
    TicketExplainer(
        term="BIL",
        category="Ticker",
        kind="Ticker",
        plain_english="SPDR Bloomberg 1-3 Month T-Bill ETF, used as the main defensive cash/T-bill proxy.",
        calculation="Pulled as a price series and used when strategies raise defensive exposure.",
        how_to_read="BIL usually means the bot is parking risk budget in a cash-like Treasury bill sleeve.",
        caution="BIL is low volatility, not zero risk; yield, fees, and tax treatment still matter.",
        aliases=("cash", "t-bill", "treasury bill", "defensive"),
    ),
    TicketExplainer(
        term="IWM",
        category="Ticker",
        kind="Ticker",
        plain_english="iShares Russell 2000 ETF, used as a small-cap/high-beta risk-on sleeve.",
        calculation="Pulled as a price series and used in risk-on target postures and relative breadth checks.",
        how_to_read="IWM exposure usually means the system wants broader or higher-beta domestic equity risk.",
        caution="Small caps can be rate-, credit-, and liquidity-sensitive.",
        aliases=("Russell 2000", "small caps", "high beta"),
    ),
    TicketExplainer(
        term="VT",
        category="Ticker",
        kind="Ticker",
        plain_english="Vanguard Total World Stock ETF, used as a global equity sleeve and simple global benchmark.",
        calculation="Pulled as a price series when included in baselines or KISS-style reference portfolios.",
        how_to_read="VT represents global stock exposure instead of a U.S.-only equity benchmark.",
        caution="Global equity can lag U.S. tech-led markets for long stretches.",
        aliases=("global equity", "total world stock"),
    ),
    TicketExplainer(
        term="USFR",
        category="Ticker",
        kind="Ticker",
        plain_english="WisdomTree Floating Rate Treasury Fund, a defensive floating-rate Treasury proxy.",
        calculation="Pulled as a price series when included in cash/T-bill reference sleeves.",
        how_to_read="USFR is a defensive cash-like sleeve with floating-rate Treasury exposure.",
        caution="It is not identical to BIL, cash, or a money-market fund.",
        aliases=("floating rate treasury", "cash proxy"),
    ),
    TicketExplainer(
        term="GLDM",
        category="Ticker",
        kind="Ticker",
        plain_english="SPDR Gold MiniShares Trust, used as a gold sleeve proxy.",
        calculation="Pulled as a price series when included in multi-asset reference or defensive portfolios.",
        how_to_read="Gold exposure can diversify equity/rate shocks when the signal supports it.",
        caution="Gold can underperform for long periods and is not a guaranteed crisis hedge.",
        aliases=("gold", "GLD"),
    ),
    TicketExplainer(
        term="FBTC",
        category="Ticker",
        kind="Ticker",
        plain_english="Fidelity Wise Origin Bitcoin Fund, used as a bitcoin sleeve proxy in reference portfolios.",
        calculation="Pulled as a price series when available in crypto or KISS-style reference sleeves.",
        how_to_read="FBTC is high-volatility risk exposure, not a defensive asset.",
        caution="Bitcoin ETFs have short live histories and large drawdown risk.",
        aliases=("bitcoin", "BTC", "crypto"),
    ),
)


def _ticker_lookup_explainers() -> tuple[TicketExplainer, ...]:
    explicit_tickers = {
        explainer.term.upper()
        for explainer in TICKET_EXPLAINERS
        if explainer.kind.lower() == "ticker"
    }
    ticker_groups = _ticker_groups_from_defaults()
    generated: list[TicketExplainer] = []
    for ticker in sorted(ticker_groups):
        if ticker in explicit_tickers:
            continue
        groups = tuple(sorted(ticker_groups[ticker]))
        role = _ticker_role(groups)
        profile = instrument_profile(ticker)
        article = _article_for_role(role)
        if profile is not None:
            plain_english = (
                f"{profile.identity} Trade-bot tracks it as {article} {role} ticker or proxy."
            )
            calculation = (
                f"Pulled as a price series when available. Local identity metadata: "
                f"{profile.name}; {profile.asset_type}; "
                f"{profile.sector or 'unclassified'}"
                f"{f' / {profile.industry}' if profile.industry else ''}. Used by "
                "strategies, risk diagnostics, tactical matrices, or research watchlists "
                "depending on the selected context."
            )
            how_to_read = (
                f"{profile.description} In trade-bot, read {ticker} through its active "
                "context: it may be a direct holding, benchmark, factor proxy, or watch-only "
                "confirmation signal."
            )
            caution = (
                "The identity metadata explains what the instrument is, but lookup coverage "
                "does not mean it is automatically tradable, approved for the current account, "
                "or allowed to drive allocation sizing."
            )
            aliases = tuple(dict.fromkeys((*groups, *profile.aliases, profile.name)))
        else:
            plain_english = (
                f"{ticker} is tracked by trade-bot as {article} {role} ticker or proxy."
            )
            calculation = (
                "Pulled as a price series when available and used by strategies, "
                "risk diagnostics, tactical matrices, or research watchlists depending "
                "on the selected context."
            )
            how_to_read = (
                f"Read {ticker} through its active context: it may be a direct holding, "
                "a benchmark, a factor proxy, or a watch-only confirmation signal."
            )
            caution = (
                "Lookup coverage does not mean the ticker is automatically tradable, "
                "approved for the current account, or allowed to drive allocation sizing."
            )
            aliases = groups
        generated.append(
            TicketExplainer(
                term=ticker,
                category="Ticker",
                kind="Ticker",
                plain_english=plain_english,
                calculation=calculation,
                how_to_read=how_to_read,
                caution=caution,
                aliases=aliases,
            )
        )
    return tuple(generated)


def _ticker_groups_from_defaults() -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for name in dir(defaults):
        if not name.startswith("DEFAULT_"):
            continue
        value = getattr(defaults, name)
        if name.endswith(("_TICKERS", "_PROXIES", "_ASSET_PROXIES")):
            _add_ticker_values(groups, value, _default_name_to_group(name))
    return groups


def _add_ticker_values(groups: dict[str, set[str]], value: object, group: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _add_ticker_values(groups, nested, f"{group}: {key}")
        return
    if isinstance(value, str):
        _add_ticker(groups, value, group)
        return
    if not isinstance(value, Iterable):
        return
    for item in value:
        if isinstance(item, str):
            _add_ticker(groups, item, group)
        elif isinstance(item, Iterable):
            for nested in item:
                if isinstance(nested, str):
                    _add_ticker(groups, nested, group)


def _add_ticker(groups: dict[str, set[str]], value: str, group: str) -> None:
    ticker = value.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker):
        return
    groups.setdefault(ticker, set()).add(group)


def _default_name_to_group(name: str) -> str:
    group = name.removeprefix("DEFAULT_")
    for suffix in ("_TICKERS", "_ASSET_PROXIES", "_PROXIES"):
        group = group.removesuffix(suffix)
    return group.lower().replace("_", " ")


def _ticker_role(groups: tuple[str, ...]) -> str:
    joined = " ".join(groups)
    if "ai" in joined or "semiconductor" in joined or "growth" in joined:
        return "AI/growth, software, or semiconductor"
    if "defensive" in joined or "tbill" in joined:
        return "defensive cash/T-bill"
    if "credit" in joined:
        return "credit"
    if "duration" in joined or "rates" in joined:
        return "rates/duration"
    if "commodity" in joined or "energy" in joined or "gold" in joined:
        return "commodity/inflation"
    if "international" in joined or "global" in joined:
        return "global equity"
    if "volatility" in joined:
        return "volatility/liquidity"
    if "broad equity" in joined or "sector" in joined or "cyclical" in joined:
        return "equity"
    return "tracked"


def _article_for_role(role: str) -> str:
    if role[:1].lower() in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


ALL_TICKET_EXPLAINERS: tuple[TicketExplainer, ...] = (
    *TICKET_EXPLAINERS,
    *_ticker_lookup_explainers(),
)


def ticket_help(term_name: str) -> str | None:
    explainer = ticket_detail(term_name)
    if explainer is None:
        return None
    return (
        f"{explainer.plain_english}\n\n"
        f"How to read: {explainer.how_to_read}\n\n"
        f"Watch out: {explainer.caution}"
    )


def ticket_detail(term_name: str) -> TicketExplainer | None:
    return _EXPLAINER_BY_KEY.get(_normalize_key(term_name))


def ticket_categories() -> tuple[str, ...]:
    return tuple(dict.fromkeys(explainer.category for explainer in ALL_TICKET_EXPLAINERS))


def ticket_guide_frame(
    *,
    category: str | None = None,
    search: str = "",
) -> pd.DataFrame:
    rows = [
        {
            "term": explainer.term,
            "kind": explainer.kind,
            "category": explainer.category,
            "plain_english": explainer.plain_english,
            "calculation": explainer.calculation,
            "how_to_read": explainer.how_to_read,
            "caution": explainer.caution,
            "aliases": ", ".join(explainer.aliases),
        }
        for explainer in ALL_TICKET_EXPLAINERS
    ]
    frame = pd.DataFrame(rows)
    if category:
        frame = frame[frame["category"] == category]
    query = search.strip().lower()
    if query:
        haystack = frame.astype(str).agg(" ".join, axis=1).str.lower()
        normalized_query = _normalize_key(query)
        normalized_haystack = haystack.map(_normalize_key)
        frame = frame[
            haystack.str.contains(re.escape(query), na=False)
            | normalized_haystack.str.contains(re.escape(normalized_query), na=False)
        ]
        if not frame.empty:
            frame = frame.assign(search_rank=_search_rank(frame, query)).sort_values(
                ["search_rank", "term"],
                kind="stable",
            )
            frame = frame.drop(columns=["search_rank"])
    return frame.reset_index(drop=True)


def ticket_column_help() -> dict[str, str]:
    help_by_column: dict[str, str] = {}
    for explainer in ALL_TICKET_EXPLAINERS:
        help_text = ticket_help(explainer.term)
        if not help_text:
            continue
        for key in (explainer.term, *explainer.aliases):
            normalized = _normalize_key(key)
            help_by_column[normalized] = help_text
    return help_by_column


def _build_explainer_lookup() -> dict[str, TicketExplainer]:
    lookup: dict[str, TicketExplainer] = {}
    for explainer in ALL_TICKET_EXPLAINERS:
        for key in (explainer.term, *explainer.aliases):
            lookup[_normalize_key(key)] = explainer
    return lookup


def _normalize_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("$", " dollar ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _search_rank(frame: pd.DataFrame, query: str) -> pd.Series:
    terms = frame["term"].astype(str).str.lower()
    aliases = frame.get("aliases", pd.Series("", index=frame.index)).astype(str).str.lower()
    normalized_query = _normalize_key(query)
    normalized_terms = terms.map(_normalize_key)
    exact_term = (terms == query) | (normalized_terms == normalized_query)
    exact_alias = aliases.str.split(", ").apply(
        lambda values: query in values or normalized_query in {_normalize_key(value) for value in values}
    )
    contains_term = terms.str.contains(re.escape(query), na=False) | normalized_terms.str.contains(
        re.escape(normalized_query),
        na=False,
    )
    return pd.Series(
        [
            0 if exact else 1 if alias else 2 if contains else 3
            for exact, alias, contains in zip(
                exact_term,
                exact_alias,
                contains_term,
                strict=False,
            )
        ],
        index=frame.index,
    )


_EXPLAINER_BY_KEY = _build_explainer_lookup()
