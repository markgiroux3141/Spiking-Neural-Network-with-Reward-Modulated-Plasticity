"""SPEC §4.5 acceptance: under prolonged noise the network stays bounded -- mean
|W| finite, and it neither fully saturates nor falls fully silent."""
import numpy as np

from config import Config
from snn.network import FeedForwardSNN
from snn import stability


def test_weights_stay_bounded_under_noise():
    cfg = Config()
    net = FeedForwardSNN(cfg, cfg.rng(0))
    rng = cfg.rng(7)
    n_input = cfg.derived()["n_input"]

    steps = 100_000
    for t in range(steps):
        in_spikes = rng.random(n_input) < 0.05      # sparse random input
        net.step(in_spikes)
        net.apply_reward(rng.normal(0.0, 0.1))       # noisy, zero-mean reward

    h = stability.weight_health((net.syn_ih, net.syn_ho))
    assert np.isfinite(h["mean_abs_w"])
    assert h["mean_abs_w"] <= cfg.w_max
    assert h["frac_at_max"] < 0.99    # not fully saturated
    assert h["frac_zero"] < 0.99      # not fully silent
