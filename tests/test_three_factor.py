"""SPEC §4.3 acceptance: with positive eligibility, R>baseline grows the weight,
R<baseline shrinks it, and stale (≈0) eligibility barely moves it."""
import numpy as np

from config import Config
from snn.synapses import Synapses


def _syn_with_eligibility(e_value):
    cfg = Config()
    cfg.connection_prob = 1.0
    syn = Synapses(1, 1, cfg, cfg.rng(0))
    syn.mask[:] = True
    syn.W[:] = 0.5
    syn.e[:] = e_value
    return cfg, syn


def test_reward_above_baseline_grows_weight():
    cfg, syn = _syn_with_eligibility(1.0)
    w0 = syn.W[0, 0]
    syn.reward_update(delta_R=+0.5)   # R - baseline > 0
    assert syn.W[0, 0] > w0


def test_reward_below_baseline_shrinks_weight():
    cfg, syn = _syn_with_eligibility(1.0)
    w0 = syn.W[0, 0]
    syn.reward_update(delta_R=-0.5)   # R - baseline < 0
    assert syn.W[0, 0] < w0


def test_stale_eligibility_barely_moves_weight():
    cfg, syn = _syn_with_eligibility(1e-6)
    w0 = syn.W[0, 0]
    syn.reward_update(delta_R=+1.0)
    assert abs(syn.W[0, 0] - w0) < 1e-6
