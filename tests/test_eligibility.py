"""SPEC §4.2 acceptance: a pre-then-post pair raises e; it decays to ~37% after
tau_e with no further activity; a post-then-pre pair lowers e."""
import numpy as np

from config import Config
from snn.synapses import Synapses


def _single_synapse():
    cfg = Config()
    cfg.connection_prob = 1.0
    syn = Synapses(1, 1, cfg, cfg.rng(0))
    syn.mask[:] = True
    syn.e[:] = 0.0
    return cfg, syn


def test_pre_before_post_potentiates():
    cfg, syn = _single_synapse()
    syn.update(np.array([True]), np.array([False]))   # pre fires
    syn.update(np.array([False]), np.array([True]))   # then post fires
    assert syn.e[0, 0] > 0


def test_post_before_pre_depresses():
    cfg, syn = _single_synapse()
    syn.update(np.array([False]), np.array([True]))   # post fires
    syn.update(np.array([True]), np.array([False]))   # then pre fires
    assert syn.e[0, 0] < 0


def test_eligibility_decays_to_37pct_after_tau_e():
    cfg, syn = _single_synapse()
    syn.update(np.array([True]), np.array([False]))
    syn.update(np.array([False]), np.array([True]))
    e0 = syn.e[0, 0]
    n = int(round(cfg.tau_e / cfg.dt))
    for _ in range(n):
        syn.update(np.array([False]), np.array([False]))
    ratio = syn.e[0, 0] / e0
    assert 0.30 < ratio < 0.43   # ~exp(-1) = 0.368
