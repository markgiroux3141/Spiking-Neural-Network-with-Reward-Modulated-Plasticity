"""Feed-forward reward-modulated SNN: input -> hidden -> output (SPEC §4.3, §6).

Owns the layers, the two synapse groups, per-layer homeostasis, the reward
baseline, and the three-factor weight update. One `step(in_spikes)` advances the
whole network one timestep; `apply_reward(R)` performs the (R - baseline) update
using the current eligibility traces.

Per-step order (SPEC §6):
    deliver synaptic current -> integrate LIF -> detect spikes
    -> update STDP traces -> write eligibility -> decay eligibility
    -> update homeostasis
Reward is applied separately, right after, against the *current* eligibility.
"""
from __future__ import annotations

import numpy as np

from config import Config
from snn.neuron import LIFLayer
from snn.synapses import Synapses
from snn.homeostasis import Homeostasis
from snn import stability


class FeedForwardSNN:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        d = cfg.derived()
        self.n_input = d["n_input"]
        self.n_hidden = cfg.n_hidden
        self.n_output = cfg.n_output

        self.hidden = LIFLayer(self.n_hidden, cfg)
        self.output = LIFLayer(self.n_output, cfg)

        self.syn_ih = Synapses(self.n_input, self.n_hidden, cfg, rng)
        self.syn_ho = Synapses(self.n_hidden, self.n_output, cfg, rng)

        self.homeo_h = Homeostasis(self.n_hidden, cfg, rng)
        self.homeo_o = Homeostasis(self.n_output, cfg, rng)

        self.baseline = 0.0
        self.alpha_baseline = d["alpha_baseline"]
        self.learning_enabled = True
        self.step_count = 0

        # last-step state, exposed for visualization
        self.in_spikes = np.zeros(self.n_input, dtype=bool)
        self.last_delta_R = 0.0
        self.last_R = 0.0

    def reset_state(self) -> None:
        """Reset dynamics for a new episode; keep learned weights."""
        self.hidden.reset()
        self.output.reset()
        self.syn_ih.reset_state()
        self.syn_ho.reset_state()
        self.homeo_h.reset()
        self.homeo_o.reset()

    def step(self, in_spikes: np.ndarray) -> np.ndarray:
        """Advance one timestep. Returns the output-layer spike vector."""
        self.in_spikes = in_spikes.astype(bool)

        # hidden layer: synaptic drive from input + boredom noise
        I_hidden = self.syn_ih.current(self.in_spikes) + self.homeo_h.noise_current()
        hidden_spikes = self.hidden.step(I_hidden)

        # output layer: synaptic drive from hidden + boredom noise
        I_output = self.syn_ho.current(hidden_spikes) + self.homeo_o.noise_current()
        output_spikes = self.output.step(I_output)

        # plasticity: STDP writes to eligibility traces (not weights)
        self.syn_ih.update(self.in_spikes, hidden_spikes)
        self.syn_ho.update(hidden_spikes, output_spikes)

        # homeostasis tracks each layer's firing rate
        self.homeo_h.observe(hidden_spikes)
        self.homeo_o.observe(output_spikes)

        self.step_count += 1
        return output_spikes

    def apply_reward(self, R: float) -> None:
        """Three-factor update against reward prediction error (SPEC §4.3)."""
        self.last_R = float(R)
        delta_R = R - self.baseline
        self.last_delta_R = float(delta_R)
        if self.learning_enabled:
            self.syn_ih.reward_update(delta_R)
            self.syn_ho.reward_update(delta_R)
            stability.maintain((self.syn_ih, self.syn_ho), self.step_count, self.cfg)
        # baseline tracks expected reward (EMA); update regardless of learning
        self.baseline = self.alpha_baseline * self.baseline + (1 - self.alpha_baseline) * R

    # --- introspection helpers for the UI -------------------------------
    @property
    def hidden_spikes(self) -> np.ndarray:
        return self.hidden.spikes

    @property
    def output_spikes(self) -> np.ndarray:
        return self.output.spikes
