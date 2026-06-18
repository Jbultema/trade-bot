from __future__ import annotations

from trade_bot.data.fred_data import parse_fred_csv


def test_parse_fred_csv_handles_missing_dots() -> None:
    csv_text = "observation_date,DGS10\n2024-01-01,4.00\n2024-01-02,.\n2024-01-03,4.10\n"

    series = parse_fred_csv(csv_text, "DGS10")

    assert series.shape[0] == 2
    assert series.iloc[-1] == 4.10
