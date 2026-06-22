from __future__ import annotations

from pytest import approx

from trade_bot.config import configured_tickers, load_config


def test_load_config_applies_hard_ticker_exclusions(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        data:
          start: "2020-01-01"
        execution: {}
        universe:
          ai_beta:
            - SPY
            - ORCL
            - QQQ
        strategies:
          fixed:
            type: fixed_allocation
            tickers:
              - SPY
              - ORCL
            allocation_weights:
              SPY: 0.5
              ORCL: 0.5
          cycle:
            type: ai_risk_cycle_overlay
            tickers:
              - SPY
              - ORCL
              - QQQ
            satellite_tickers:
              - ORCL
              - QQQ
            defensive_ticker: BIL
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert "ORCL" not in config.universe["ai_beta"]
    assert "ORCL" not in config.strategies["fixed"].tickers
    assert "ORCL" not in config.strategies["cycle"].tickers
    assert "ORCL" not in config.strategies["cycle"].satellite_tickers
    assert config.strategies["fixed"].allocation_weights == {"SPY": approx(1.0)}
    assert "ORCL" not in configured_tickers(config)
