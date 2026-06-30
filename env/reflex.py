"""Hand-wired reflex controller (SPEC §12.3 acceptance / §14 Phase 11).

A Braitenberg-style reflex that reads the vision rays directly and produces a
(forward, turn) command -- no SNN. It exists to prove the world/vision/motor
geometry is sound (an agent CAN reach green and avoid red) so that a later
learning failure can be localized to the spiking network rather than the
environment. Not used during SNN training.

    reward ahead      -> go forward
    reward on a side  -> turn toward it
    punishment / wall -> slow down and turn away
"""
from __future__ import annotations

import numpy as np

from config import Config
from env.vision import ray_angles
from env.regions import TYPE_REWARD, TYPE_PUNISH, TYPE_WALL, TYPE_AGENT, TYPE_NONE

# how attractive/repulsive each hit type is
_VALENCE = {
    TYPE_REWARD: +1.0,
    TYPE_PUNISH: -1.0,
    TYPE_WALL: -0.5,
    TYPE_AGENT: -0.1,
    TYPE_NONE: 0.0,
}

BASE_FORWARD = 0.0028   # cruise speed (world units/step) so it keeps exploring
FWD_GAIN = 0.004
TURN_GAIN = 0.12


def reflex_motor(proximity: np.ndarray, hit_type: np.ndarray, cfg: Config) -> np.ndarray:
    """proximity, hit_type: (M, n_rays). Returns motor (M, 2) = (forward, turn)."""
    offsets = ray_angles(cfg)                       # (n_rays,)  + = left
    valence = np.vectorize(_VALENCE.get)(hit_type).astype(float)  # (M, n_rays)
    salience = valence * proximity                  # (M, n_rays)

    # turn toward attractive things on a side, away from repulsive ones
    turn = TURN_GAIN * np.sum(salience * np.sin(offsets)[None, :], axis=1)
    # speed up toward reward ahead, slow for hazards ahead
    forward = BASE_FORWARD + FWD_GAIN * np.sum(salience * np.cos(offsets)[None, :], axis=1)
    forward = np.clip(forward, 0.0, None)
    return np.stack([forward, turn], axis=1)
