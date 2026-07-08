from __future__ import annotations

from trade_bot.dashboard.navigation import dashboard_section_names, section_guide
from trade_bot.DEFAULTS import DEFAULT_DASHBOARD_SECTIONS


def test_dashboard_section_guides_cover_default_sections() -> None:
    assert dashboard_section_names() == DEFAULT_DASHBOARD_SECTIONS
    section_names = dashboard_section_names()
    research_index = section_names.index("Research Lab")
    assert section_names[research_index + 1] == "Simulation Lab"
    assert section_names[research_index + 2] == "Launch Lab"

    for section in DEFAULT_DASHBOARD_SECTIONS:
        guide = section_guide(section)
        assert guide.name == section
        assert guide.role
        assert guide.primary_question.endswith("?")
        assert guide.first_read
        assert guide.next_step
        assert guide.runtime
        assert guide.runtime_note

    assert section_guide("Simulation Lab").runtime == "Heavy"
    assert section_guide("Launch Lab").runtime == "Heavy"
    assert section_guide("Research Lab").runtime == "Heavy"


def test_unknown_section_falls_back_to_command_center_guide() -> None:
    assert section_guide("unknown").name == "Command Center"
