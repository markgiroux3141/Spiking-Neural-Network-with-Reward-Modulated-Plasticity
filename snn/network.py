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

        # Dale's law (ROADMAP Phase A): tag a fraction of hidden neurons inhibitory.
        n_inh = int(round(cfg.inhib_fraction * self.n_hidden))
        self.hidden_inhib = np.zeros(self.n_hidden, dtype=bool)
        if n_inh > 0:
            inh_idx = rng.choice(self.n_hidden, size=n_inh, replace=False)
            self.hidden_inhib[inh_idx] = True
        hidden_sign = np.where(self.hidden_inhib, -1.0, 1.0)

        # input -> hidden: input is all excitatory, all synapses plastic. This is
        # where the network LEARNS which inputs drive the inhibitory neurons.
        self.syn_ih = Synapses(self.n_input, self.n_hidden, cfg, rng)
        # hidden -> output: inhibitory hidden neurons deliver negative current;
        # v1 keeps those inhibitory columns structural (non-plastic), so learning
        # stays on the excitatory hidden->output (approach) synapses.
        self.syn_ho = Synapses(
            self.n_hidden, self.n_output, cfg, rng,
            pre_sign=hidden_sign,
            pre_plastic=(~self.hidden_inhib if not cfg.inhib_plastic
                         else np.ones(self.n_hidden, dtype=bool)),
        )
        # Structured DIRECTIONAL inhibition (ROADMAP Phase A v2): each inhibitory
        # hidden neuron vetoes exactly ONE motor pool, round-robin over
        # turn-left(1), turn-right(2), forward(0). Undirected random inhibition
        # (v1) just froze agents in place (suppressing escape too); routing
        # inhibition to specific pools lets "red on a side" suppress the ipsilateral
        # turn -> the agent turns the other way -> away from red. The motor
        # targeting is innate (aversive reflexes are genetically structured), while
        # WHICH inputs drive each inhibitory neuron stays plastic (input->hidden).
        targets = [1, 2, 0]   # turn-left, turn-right, forward (encoding.OUT_*)
        wmag = min(cfg.w_max, 0.8 * cfg.inhib_w_scale)
        for r, j in enumerate(np.where(self.hidden_inhib)[0]):
            t = targets[r % len(targets)]
            self.syn_ho.mask[:, j] = False
            self.syn_ho.W[:, j] = 0.0
            self.syn_ho.mask[t, j] = True
            self.syn_ho.W[t, j] = wmag

        self.homeo_h = Homeostasis(self.n_hidden, cfg, rng)
        self.homeo_o = Homeostasis(self.n_output, cfg, rng)

        self.learning_enabled = True
        self.step_count = 0

        # TD critic (ROADMAP Phase A.2): a linear value V(s) = w_v . f(s) + b_v over
        # a smoothed sensory-feature vector. Plasticity is driven by the TD error
        # delta = R + gamma*V(s') - V(s) instead of (R - baseline). The critic
        # learns the value/shaping from experience (replaces the hand-shaped
        # potential), and the relative signal symmetrizes seeking and avoidance.
        self.alpha_critic = d["alpha_critic"]
        # critic features are normalized to ~[0,1]: a raw spike-EMA at sparse ~2 Hz
        # is ~0.002 (too small to learn), but scaling all the way to Hz (~100) makes
        # the TD bootstrap diverge. Normalize so a neuron firing at max_input_rate
        # maps to ~1 -> stable AND learnable.
        self.feat_scale = 1000.0 / (cfg.max_input_rate * cfg.dt)
        self.w_v = np.zeros(self.n_input)
        self.b_v = 0.0
        self.x_critic = np.zeros(self.n_input)       # smoothed current features
        self.x_critic_prev = np.zeros(self.n_input)  # features of previous state
        self.e_v = np.zeros(self.n_input)            # critic eligibility trace (TD(λ))
        self.e_v_b = 0.0                             # ... for the bias
        self.V_prev = 0.0
        self.value = 0.0
        self.baseline = 0.0   # exposed to the inspector as the learned value V(s)

        # last-step state, exposed for visualization
        self.in_spikes = np.zeros(self.n_input, dtype=bool)
        self.last_delta_R = 0.0
        self.last_R = 0.0

    def reset_state(self) -> None:
        """Reset dynamics for a new episode; keep learned weights (incl. critic)."""
        self.hidden.reset()
        self.output.reset()
        self.syn_ih.reset_state()
        self.syn_ho.reset_state()
        self.homeo_h.reset()
        self.homeo_o.reset()
        self.x_critic[:] = 0.0
        self.x_critic_prev[:] = 0.0
        self.e_v[:] = 0.0
        self.e_v_b = 0.0
        self.V_prev = 0.0

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

        # smoothed sensory features (firing rate, Hz) for the critic's value
        self.x_critic = (self.alpha_critic * self.x_critic
                         + (1 - self.alpha_critic) * self.in_spikes * self.feat_scale)

        self.step_count += 1
        return output_spikes

    def apply_reward(self, R: float) -> None:
        """Drive plasticity by the TD prediction error (ROADMAP Phase A.2):
        delta = R + gamma*V(s') - V(s), with a learned linear value V."""
        self.last_R = float(R)
        cfg = self.cfg
        V_now = float(np.clip(self.w_v @ self.x_critic + self.b_v,
                              -cfg.value_clip, cfg.value_clip))
        delta = R + cfg.td_gamma * V_now - self.V_prev

        # critic update (TD(λ)): accumulate an eligibility trace of the states being
        # valued (decaying at γλ), then nudge all of them by δ. This propagates the
        # in-region reward backward along the approach/retreat path, so the distance
        # senses ("region ahead") acquire value -> a spatial gradient for navigation.
        decay = cfg.td_gamma * cfg.td_lambda
        self.e_v = decay * self.e_v + self.x_critic_prev
        self.e_v_b = decay * self.e_v_b + 1.0
        self.w_v += cfg.td_lr * delta * self.e_v
        self.b_v += cfg.td_lr * delta * self.e_v_b

        delta_actor = float(np.clip(delta, -cfg.delta_clip, cfg.delta_clip))
        self.last_delta_R = delta_actor
        if self.learning_enabled:
            self.syn_ih.reward_update(delta_actor)
            self.syn_ho.reward_update(delta_actor)
            stability.maintain((self.syn_ih, self.syn_ho), self.step_count, self.cfg)

        self.V_prev = V_now
        self.x_critic_prev = self.x_critic.copy()
        self.value = V_now
        self.baseline = V_now   # inspector shows the learned value

    # --- introspection helpers for the UI -------------------------------
    @property
    def hidden_spikes(self) -> np.ndarray:
        return self.hidden.spikes

    @property
    def output_spikes(self) -> np.ndarray:
        return self.output.spikes
