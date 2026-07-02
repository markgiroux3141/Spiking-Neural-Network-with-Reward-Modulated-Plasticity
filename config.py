"""All hyperparameters in one seed-controlled place (SPEC §7, §12).

Every subsystem reads from this dataclass so a single seed + config fully
determines a run. Values are *plausible starting points* and expected to be tuned;
see SPEC §7 "Tuning order when it won't learn".
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    # ---- reproducibility -------------------------------------------------
    seed: int = 0

    # ---- time (SPEC §1) --------------------------------------------------
    dt: float = 1.0  # ms per integration step

    # ---- LIF neuron (SPEC §4.1, §7) -------------------------------------
    tau_m: float = 20.0       # ms membrane leak time constant
    V_rest: float = -65.0     # mV
    V_reset: float = -65.0    # mV
    V_thresh: float = -50.0   # mV
    R_m: float = 1.0          # membrane resistance (scales input current)
    refractory_ms: float = 5.0

    # ---- STDP / eligibility (SPEC §4.2, §7) -----------------------------
    tau_pre: float = 20.0     # ms pre-synaptic trace decay
    tau_post: float = 20.0    # ms post-synaptic trace decay
    A_plus: float = 0.01      # potentiation amplitude
    A_minus: float = 0.012    # depression amplitude (> A_plus aids stability)
    tau_e: float = 500.0      # ms eligibility decay; must ~match action->reward delay

    # ---- three-factor update (SPEC §4.3) --------------------------------
    learning_rate: float = 0.01
    baseline_tau: float = 5_000.0   # ms EMA window (legacy; superseded by TD critic)
    # temporal-difference critic (ROADMAP Phase A.2): plasticity is driven by a TD
    # prediction error  delta = R + gamma*V(s') - V(s)  from a learned linear value
    # V(s) over the sensory state -- replacing BOTH the global baseline and the
    # hand-shaped potential. This is dopamine-as-prediction-error; it makes
    # "escaped red" as rewarding as "reached green" (symmetrizes seek/avoid).
    use_td_critic: bool = True
    td_gamma: float = 0.95       # per-step discount (~20-step value horizon)
    td_lambda: float = 0.92      # critic eligibility-trace decay (TD(lambda)): lets
                                 # value propagate backward across the approach path
                                 # so "region-ahead" inherits value from "in-region"
    td_lr: float = 0.01          # critic (value) learning rate (features ~[0,1])
    tau_critic: float = 100.0    # ms smoothing of sensory features for the critic
    delta_clip: float = 2.0      # clip on the TD error handed to the actor (stability)
    value_clip: float = 50.0     # clamp on V(s) to keep the bootstrap stable

    # ---- homeostasis / boredom (SPEC §4.4) ------------------------------
    # Sparse baseline activity so vision-driven firing stands out above noise.
    # Sparsity matters: dense high-rate firing makes STDP indiscriminate and
    # learning erases structure; a low setpoint lets eligibility capture real
    # pre->post causation (tuned finding).
    target_rate: float = 2.0  # Hz homeostatic setpoint
    tau_rate: float = 1000.0  # ms low-pass window for rate estimate
    noise_gain: float = 35.0  # current per unit boredom (gentle exploration)

    # ---- stability (SPEC §4.5) ------------------------------------------
    w_min: float = 0.0
    w_max: float = 1.0
    normalize_every: int = 500     # steps between synaptic normalization
    norm_target: float = 8.0       # target sum of incoming weights per post neuron
    weight_decay: float = 0.0      # off during learning; rely on normalization

    # ---- network shape (SPEC §7, §12.3) ---------------------------------
    # input size is derived from vision (n_rays * n_channels); see derived().
    n_hidden: int = 64
    n_output: int = 3              # forward, turn-left, turn-right (SPEC §12.3)
    connection_prob: float = 0.4   # sparsity of random initial connectivity

    # ---- excitatory/inhibitory split (Dale's law; ROADMAP Phase A) ------
    # A fraction of hidden neurons are inhibitory: they deliver NEGATIVE current
    # downstream, giving the network a withdrawal/veto pathway it can use to learn
    # avoidance. Cortex is ~80/20 E/I. v1: learning is on excitatory synapses only
    # (incl. which inputs drive the inhibitory neurons); inhibitory weights are
    # structural/fixed -- flip inhib_plastic for reward-modulated iSTDP later.
    inhib_fraction: float = 0.2
    inhib_plastic: bool = False
    inhib_w_scale: float = 1.0     # multiplier on inhibitory weight magnitudes
    w_init_scale: float = 0.5      # initial weight magnitude
    # synaptic current gain: decouples the [0,1] weight bounds from the membrane
    # current scale so a handful of presynaptic spikes can actually drive firing
    # (without it, weights <= 1 mV vs a ~15 mV threshold -> input can't compete
    # with boredom noise). Tuned so vision drives the network above the noise floor.
    syn_gain: float = 70.0

    # ---- vision / raycasting (SPEC §12.2, §12.3) ------------------------
    n_rays: int = 12
    fov: float = float(np.deg2rad(160.0))  # field of view spread across rays
    max_range: float = 0.6                 # ray length in world units
    # input channels per ray, one per meaningful hit type:
    vision_channels: tuple = ("reward", "punishment", "wall", "agent")
    max_input_rate: float = 100.0          # Hz Poisson rate at proximity=1 (strong drive)
    # interoception (ROADMAP Phase A): nociception/satiation sensors that fire when
    # the agent is INSIDE a punishment/reward region. Rays skip the region you're
    # in (exteroception = what's ahead), so without this the agent is blind to its
    # own danger and can't learn to escape red. 2 neurons: [in_reward, in_punish].
    interoception: bool = True

    # ---- motor decode (SPEC §12.3) --------------------------------------
    # Output spikes -> (forward, turn) via a smooth EMA firing-rate estimate (Hz).
    tau_motor: float = 100.0      # ms low-pass for motor rate (smooth movement)
    speed_scale: float = 3.0e-4   # world units/step per Hz of forward drive
    turn_scale: float = 5.0e-3    # radians/step per Hz of net turn drive
    motor_window: int = 60        # steps shown in the UI output raster

    # ---- environment / regions (SPEC §12.1) -----------------------------
    world_w: float = 1.0
    world_h: float = 1.0
    n_agents: int = 12
    n_reward_regions: int = 3
    n_punish_regions: int = 3
    region_radius: float = 0.12
    reward_magnitude: float = 1.0
    punish_magnitude: float = 1.0
    R_step: float = -0.001       # per-step efficiency pressure
    # small wall penalty: a large one injects destructive negative reward
    # correlated with motor activity (fast agents hit walls), which corrupts
    # the plasticity and erases learned structure (tuned finding).
    R_wall: float = -0.05        # penalty for hitting a wall
    # potential-based reward shaping: a hand-shaped attraction/repulsion field.
    # Superseded by the learned TD critic (Phase A.2) -- the critic learns the
    # value/shaping from experience instead of us coding it from privileged region
    # positions. Disabled by default (set > 0 to re-enable the hand-shaped version).
    shaping_gain: float = 0.0
    shaping_scale: float = 0.25  # length scale of the attraction/repulsion field
    region_drift: float = 0.0002 # slow random walk of region centers (0 = static)
    agent_radius: float = 0.012

    # ---- episode / training (SPEC §6) -----------------------------------
    steps_per_episode: int = 2000
    episodes: int = 100

    def derived(self) -> dict:
        """Quantities computed from the primary fields."""
        n_intero = 2 if self.interoception else 0
        n_input = self.n_rays * len(self.vision_channels) + n_intero
        return {
            "n_input": n_input,
            "n_intero": n_intero,
            "alpha_critic": np.exp(-self.dt / self.tau_critic),
            "alpha_pre": np.exp(-self.dt / self.tau_pre),
            "alpha_post": np.exp(-self.dt / self.tau_post),
            "alpha_e": np.exp(-self.dt / self.tau_e),
            "alpha_rate": np.exp(-self.dt / self.tau_rate),
            "alpha_motor": np.exp(-self.dt / self.tau_motor),
            "alpha_m": np.exp(-self.dt / self.tau_m),
            "alpha_baseline": np.exp(-self.dt / self.baseline_tau),
            "refractory_steps": int(round(self.refractory_ms / self.dt)),
        }

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)
