"""Synapses: weight matrix, STDP traces, eligibility trace (SPEC §4.2, §4.5).

A dense connection from a pre layer (n_pre) to a post layer (n_post):
    W[post, pre]   weights
    e[post, pre]   eligibility (same shape) -- STDP writes HERE, not to W
    mask[post, pre] fixed connectivity (preserves sparsity through updates)

CRITICAL (SPEC §4.2): STDP modifies the eligibility trace `e`, never `W` directly.
`W` only changes when reward arrives (reward_update, driven by network.py §4.3).
This is what lets credit assignment bridge the action->reward delay.
"""
from __future__ import annotations

import numpy as np

from config import Config


class Synapses:
    def __init__(self, n_pre: int, n_post: int, cfg: Config,
                 rng: np.random.Generator):
        self.n_pre = n_pre
        self.n_post = n_post
        self.cfg = cfg
        d = cfg.derived()
        self.alpha_pre = d["alpha_pre"]
        self.alpha_post = d["alpha_post"]
        self.alpha_e = d["alpha_e"]

        # sparse random connectivity, weights in [0, w_init_scale]
        self.mask = rng.random((n_post, n_pre)) < cfg.connection_prob
        self.W = rng.random((n_post, n_pre)) * cfg.w_init_scale * self.mask
        self.e = np.zeros((n_post, n_pre), dtype=np.float64)

        self.x_pre = np.zeros(n_pre, dtype=np.float64)
        self.x_post = np.zeros(n_post, dtype=np.float64)

    def reset_state(self) -> None:
        """Clear dynamic state (traces, eligibility) but keep learned weights."""
        self.e[:] = 0.0
        self.x_pre[:] = 0.0
        self.x_post[:] = 0.0

    def current(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Synaptic current delivered to the post layer this step: gain * W @ spikes."""
        return self.cfg.syn_gain * (self.W @ pre_spikes.astype(np.float64))

    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        """STDP onto eligibility, then decay traces/eligibility (SPEC §4.2)."""
        # decay traces first so potentiation/depression read the pre-spike history
        self.x_pre *= self.alpha_pre
        self.x_post *= self.alpha_post

        # pre-before-post -> potentiate: posts that just spiked, weighted by pre trace
        if post_spikes.any():
            self.e[post_spikes, :] += self.cfg.A_plus * self.x_pre[None, :]
        # post-before-pre -> depress: pres that just spiked, weighted by post trace
        if pre_spikes.any():
            self.e[:, pre_spikes] -= self.cfg.A_minus * self.x_post[:, None]

        # this step's spikes bump the traces (seen by *future* steps)
        self.x_pre[pre_spikes] += 1.0
        self.x_post[post_spikes] += 1.0

        # eligibility forgets at tau_e; restrict to existing connections
        self.e *= self.alpha_e
        self.e *= self.mask

    def reward_update(self, delta_R: float) -> None:
        """Three-factor weight update (SPEC §4.3): dW = lr * (R - baseline) * e."""
        self.W += self.cfg.learning_rate * delta_R * self.e
        np.clip(self.W, self.cfg.w_min, self.cfg.w_max, out=self.W)
        self.W *= self.mask

    def decay_weights(self) -> None:
        """Tiny pull of weights toward 0 to prune unused synapses (SPEC §4.5)."""
        self.W -= self.cfg.weight_decay * self.W
        self.W *= self.mask

    def normalize(self) -> None:
        """Rescale each post neuron's incoming weights to a target sum, creating
        competition between inputs (SPEC §4.5)."""
        s = self.W.sum(axis=1, keepdims=True)
        scale = np.where(s > 1e-12, self.cfg.norm_target / s, 1.0)
        self.W *= scale
        np.clip(self.W, self.cfg.w_min, self.cfg.w_max, out=self.W)
