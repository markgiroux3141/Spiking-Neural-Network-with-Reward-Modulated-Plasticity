"""Stability maintenance + health metrics (SPEC §4.5).

The per-synapse mechanics (clip, normalize, decay) live on Synapses; this module
schedules them and reports whether the network is healthy (neither saturated nor
silent) -- the quantity checked by the 100k-step noise test.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from config import Config
from snn.synapses import Synapses


def maintain(synapse_groups: Iterable[Synapses], step: int, cfg: Config) -> None:
    """Apply weight decay every step and normalization on the configured cadence."""
    for syn in synapse_groups:
        syn.decay_weights()
    if cfg.normalize_every > 0 and step % cfg.normalize_every == 0:
        for syn in synapse_groups:
            syn.normalize()


def weight_health(synapse_groups: Iterable[Synapses]) -> dict:
    """Summary stats used to confirm weights stay bounded (not saturated/silent)."""
    groups = list(synapse_groups)
    mean_abs = np.mean([np.abs(s.W[s.mask]).mean() for s in groups if s.mask.any()])
    frac_at_max = np.mean([
        (np.isclose(s.W[s.mask], s.cfg.w_max)).mean() for s in groups if s.mask.any()
    ])
    frac_zero = np.mean([
        (s.W[s.mask] <= 1e-9).mean() for s in groups if s.mask.any()
    ])
    return {
        "mean_abs_w": float(mean_abs),
        "frac_at_max": float(frac_at_max),
        "frac_zero": float(frac_zero),
    }
