from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class PerfSample:
    name: str
    elapsed_ms: float


@contextmanager
def timed(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        samples = st.session_state.setdefault("dashboard_v2_perf_samples", [])
        samples.append(PerfSample(name=name, elapsed_ms=elapsed_ms))


def render_perf_footer() -> None:
    samples = st.session_state.get("dashboard_v2_perf_samples", [])
    if not samples:
        return
    with st.expander("V2 render timings", expanded=False):
        st.dataframe(
            [
                {"step": sample.name, "elapsed_ms": round(sample.elapsed_ms, 1)}
                for sample in samples[-20:]
            ],
            width="stretch",
            hide_index=True,
        )
