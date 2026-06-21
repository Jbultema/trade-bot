"""Backward-compatible re-export for centralized project defaults.

New code should import from :mod:`trade_bot.DEFAULTS`. This shim exists so older
notebooks, scripts, or cached Streamlit sessions do not fail immediately after
the module rename.
"""

from trade_bot.DEFAULTS import *  # noqa: F401,F403
