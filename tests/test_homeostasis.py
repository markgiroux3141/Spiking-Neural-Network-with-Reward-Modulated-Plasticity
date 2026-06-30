"""SPEC §4.4 acceptance: a disconnected neuron self-fires toward target_rate;
once it receives strong real drive its boredom term falls to ~0."""
import numpy as np

from config import Config
from snn.neuron import LIFLayer
from snn.homeostasis import Homeostasis


def test_free_neuron_reaches_target_rate():
    cfg = Config()
    layer = LIFLayer(50, cfg)
    homeo = Homeostasis(50, cfg, cfg.rng(1))
    spikes = 0
    steps = 20000
    for _ in range(steps):
        s = layer.step(homeo.noise_current())   # boredom only, no synaptic input
        homeo.observe(s)
        spikes += s.sum()
    hz = spikes / (50 * steps * cfg.dt / 1000.0)
    # should settle in the neighborhood of target_rate (not silent, not runaway)
    assert cfg.target_rate * 0.5 < hz < cfg.target_rate * 1.6


def test_boredom_falls_under_strong_drive():
    cfg = Config()
    layer = LIFLayer(50, cfg)
    homeo = Homeostasis(50, cfg, cfg.rng(2))
    # drive every neuron hard with constant suprathreshold current
    for _ in range(3000):
        s = layer.step(np.full(50, 40.0))
        homeo.observe(s)
    boredom = np.clip(cfg.target_rate - homeo.rate_est, 0.0, None)
    assert boredom.mean() < 0.1   # well-driven -> boredom ~ 0
