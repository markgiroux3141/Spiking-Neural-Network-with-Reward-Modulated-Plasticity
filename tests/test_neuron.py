"""SPEC §4.1 acceptance: constant suprathreshold input -> stable rate that
increases with I; zero input + no noise -> silent."""
import numpy as np

from config import Config
from snn.neuron import LIFLayer


def _fire_count(I_value, steps=1000):
    cfg = Config()
    layer = LIFLayer(8, cfg)
    I = np.full(8, I_value)
    count = 0
    for _ in range(steps):
        count += layer.step(I).sum()
    return count


def test_silent_with_zero_input():
    assert _fire_count(0.0) == 0


def test_fires_when_suprathreshold():
    # target V = V_rest + R_m*I = -65 + 30 = -35 mV > V_thresh (-50) -> fires
    assert _fire_count(30.0) > 0


def test_rate_monotonic_in_input():
    low = _fire_count(20.0)
    high = _fire_count(60.0)
    assert high > low
