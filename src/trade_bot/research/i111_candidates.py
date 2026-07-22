from __future__ import annotations

from dataclasses import dataclass

from trade_bot.config import BotConfig, StrategyConfig
from trade_bot.research.upside_capture import _candidate_rounds

DEFAULT_I111_PREFIX = "i111_reentry_vol_target_fast_21d"
DEFAULT_UPSIDE_TOP_NAMES = (
    "r18_min025_vol185_guard145",
    "r19_min025_vol185_guard17_mult65",
    "r19_min025_vol185_guard18_mult70",
    "r19_min025_vol185_guard16_mult60",
    "r20_min025_vol19_guard15_mult60",
    "r21_min025_vol185_guard145_floor60",
)


@dataclass(frozen=True)
class I111Candidate:
    name: str
    source_group: str
    strategy: StrategyConfig
    overlay: dict[str, object] | None = None


def build_i111_candidates(
    config: BotConfig,
    *,
    strategy_prefix: str = DEFAULT_I111_PREFIX,
    include_upside_research: bool = True,
    upside_top_names: tuple[str, ...] = DEFAULT_UPSIDE_TOP_NAMES,
) -> tuple[I111Candidate, ...]:
    candidates: list[I111Candidate] = []
    configured_names = [
        name
        for name in config.strategies
        if name == config.primary_strategy or name.startswith(strategy_prefix)
    ]
    for name in sorted(dict.fromkeys(configured_names)):
        candidates.append(
            I111Candidate(
                name=name,
                source_group="configured_i111",
                strategy=config.strategies[name],
            )
        )
    if include_upside_research:
        base = config.strategies.get(config.primary_strategy or "")
        if base is not None:
            round_candidates = {candidate.name: candidate for candidate in _candidate_rounds(base)}
            for name in upside_top_names:
                candidate = round_candidates.get(name)
                if candidate is None:
                    continue
                candidates.append(
                    I111Candidate(
                        name=name,
                        source_group="upside_capture_research",
                        strategy=candidate.strategy,
                        overlay=candidate.overlay,
                    )
                )
    return tuple(candidates)
