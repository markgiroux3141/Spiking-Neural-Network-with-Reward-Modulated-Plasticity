"""ROADMAP Phase A: inhibitory (Dale's-law) presynaptic neurons deliver negative
current and suppress postsynaptic firing; their weights are held fixed in v1."""
import numpy as np

from config import Config
from snn.neuron import LIFLayer
from snn.synapses import Synapses


def test_inhibitory_presynaptic_delivers_negative_current():
    cfg = Config()
    cfg.connection_prob = 1.0
    syn = Synapses(1, 1, cfg, cfg.rng(0), pre_sign=np.array([-1.0]))
    syn.mask[:] = True
    syn.W[:] = 0.5
    I = syn.current(np.array([True]))
    assert I[0] < 0


def test_inhibition_suppresses_postsynaptic_firing():
    cfg = Config()
    cfg.connection_prob = 1.0
    # excitatory drive alone makes the post neuron fire; adding a co-active
    # inhibitory input of equal weight should cancel/suppress it.
    exc = Synapses(1, 1, cfg, cfg.rng(1), pre_sign=np.array([+1.0]))
    inh = Synapses(1, 1, cfg, cfg.rng(2), pre_sign=np.array([-1.0]))
    exc.mask[:] = True; exc.W[:] = 0.5
    inh.mask[:] = True; inh.W[:] = 0.5

    def count(use_inh):
        layer = LIFLayer(1, cfg)
        spk = np.array([True])
        n = 0
        for _ in range(500):
            I = exc.current(spk)
            if use_inh:
                I = I + inh.current(spk)
            n += layer.step(I).sum()
        return n

    assert count(use_inh=False) > 0            # excitation alone fires
    assert count(use_inh=True) < count(use_inh=False)   # inhibition suppresses


def test_inhibitory_weights_are_fixed_in_v1():
    cfg = Config()
    cfg.inhib_plastic = False
    # one inhibitory presynaptic column (non-plastic) + reward update must not move it
    syn = Synapses(2, 1, cfg, cfg.rng(3),
                   pre_sign=np.array([+1.0, -1.0]),
                   pre_plastic=np.array([True, False]))
    syn.mask[:] = True
    syn.W[:] = 0.4
    syn.e[:] = 1.0
    w_inh_before = syn.W[0, 1]
    syn.reward_update(delta_R=+0.5)
    assert syn.W[0, 1] == w_inh_before          # inhibitory column unchanged
    assert syn.W[0, 0] > 0.4                     # excitatory column learned
