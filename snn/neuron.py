"""LIF neuron layer (SPEC §4.1).

A vectorized layer of N leaky integrate-and-fire neurons, all updated per
timestep. Membrane dynamics use exact exponential leak so the step is stable for
any dt:

    V <- V_rest + (V - V_rest) * alpha_m + R_m * I * (1 - alpha_m)

which is the closed-form solution of  tau_m * dV/dt = -(V - V_rest) + R_m * I
over one dt (alpha_m = exp(-dt/tau_m)). A neuron spikes when V crosses V_thresh,
is reset to V_reset, and is clamped for `refractory_steps`.
"""
from __future__ import annotations

import numpy as np

from config import Config


class LIFLayer:
    def __init__(self, n: int, cfg: Config):
        self.n = n
        self.cfg = cfg
        d = cfg.derived()
        self.alpha_m = d["alpha_m"]
        self.refractory_steps = d["refractory_steps"]

        self.V = np.full(n, cfg.V_rest, dtype=np.float64)
        self.spikes = np.zeros(n, dtype=bool)
        self.refrac = np.zeros(n, dtype=np.int64)  # steps remaining in refractory

    def reset(self) -> None:
        self.V[:] = self.cfg.V_rest
        self.spikes[:] = False
        self.refrac[:] = 0

    def step(self, I: np.ndarray) -> np.ndarray:
        """Advance one timestep given input current I (shape (n,)).

        Returns the boolean spike vector for this step.
        """
        cfg = self.cfg
        free = self.refrac <= 0

        # exact exponential integration of leak + input, only for non-refractory
        target = cfg.V_rest + cfg.R_m * I
        self.V[free] = target[free] + (self.V[free] - target[free]) * self.alpha_m

        # refractory neurons are clamped at reset and tick down
        self.V[~free] = cfg.V_reset
        self.refrac[~free] -= 1

        self.spikes = self.V >= cfg.V_thresh
        self.V[self.spikes] = cfg.V_reset
        self.refrac[self.spikes] = self.refractory_steps
        return self.spikes
