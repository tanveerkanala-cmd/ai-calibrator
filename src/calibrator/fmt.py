"""Honest percentage formatting — a displayed number must never contradict the truth.

Plain ``{:.0%}`` lies at the boundaries (audit: display-honesty):
- 249/250 (0.996) renders "100%" right next to a failing test;
- 1/250 (0.004) renders "0%" although something passed;
- a drift delta of -0.4% renders "Δ +0%" while the gate fails on it.

Rules here: "100%" only when the rate is exactly 1, "0%" only when exactly 0;
imperfect-but-rounding-to-the-pole values render ">99%" / "<1%". Deltas render
with one decimal and never collapse a real change to ±0%.
"""

from __future__ import annotations


def pct(x: float) -> str:
    """Honest whole-number percent for rates in [0, 1]."""
    if x == 0:
        return "0%"
    if x == 1:
        return "100%"
    r = round(x * 100)
    if r >= 100:
        return ">99%"
    if r <= 0:
        return "<1%"
    return f"{r}%"


def pct_delta(x: float) -> str:
    """Signed percent for deltas — a real change never displays as ±0%."""
    if x == 0:
        return "±0%"
    s = f"{x:+.1%}"
    if s in ("+0.0%", "-0.0%"):
        return "+<0.1%" if x > 0 else "-<0.1%"  # tiny but real — never ±0%
    return s
