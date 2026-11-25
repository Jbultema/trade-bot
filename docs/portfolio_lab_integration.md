# Portfolio Lab Integration

Trade Bot remains the owner of its strategic candidates, native risk/recovery rules and cycle
context. Its current primary strategy/configuration is not replaced by Portfolio Lab research.
Cycle Tracker remains contextual rather than an automatic cross-specialist allocation override.

Export with this checkout's interpreter:

```sh
poetry run python -m trade_bot.research.portfolio_export --output ../portfolio-lab/data/bundles/RUN/trade-bot
```

The output is immutable, hash-bound and research-only. Three policies are exported: the configured
i111-derived primary, gentle continuous protection, and staged recovery. Native volatility and
drawdown controls are recomputed causally at signal close for common-clock replay. This is a
reconstruction, not identical native return history. Portfolio Lab owns the common next-close
execution, cash, self-financing costs and drift. Native settings are unchanged.

The gentle and recovery policies are retained research hypotheses from adaptive_response_lab,
not newly independent discoveries. Their source universe contains selected modern technology
winners and cannot establish survivor-free future performance. The meta allocator must compete
against fixed blends, static-risk and risk-matched controls. It has no operating authority.
