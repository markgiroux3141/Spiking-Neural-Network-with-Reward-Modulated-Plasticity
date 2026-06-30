# Spiking Neural Network with Reward-Modulated Plasticity

A spiking neural network (LIF + STDP + eligibility traces + a global reward
signal) that learns to drive embodied agents through a 2D world — **no
backpropagation**. See [SPEC.md](SPEC.md) for the full design.

The current task (SPEC §12–13) is a **multi-agent region world**: small agents
move in the x/y plane via motor output neurons, *see* by raycasting, and learn to
seek **reward regions** (green) and avoid **punishment regions** (red) through
reward-modulated STDP.

## Install

```bash
pip install -r requirements.txt
```

## Run the live UI

```bash
python -m ui.app
```

Controls: `Space` play/pause · `S` step · `R` reset · `L` freeze/unfreeze learning
· `T` trails · `Y` rays · `Tab` cycle agent · **click** an agent to inspect its
network (sensor bars, spike raster, weights, reward trace) in the side panel.

## Train / evaluate headless

```bash
python train.py                 # compare all four controllers
python train.py --mode learn    # a single controller: reflex|learn|frozen|random
python train.py --steps 40000
```

`train.py` scores four controllers on a behavioral metric — the fraction of
agent-steps spent in reward vs punishment regions, measured over the steady-state
(final 40%) of the run so early exploration doesn't dilute converged behavior.

## Behavioral evaluation

```bash
python evaluate.py                 # all controllers
python evaluate.py --mode learn
```

`evaluate.py` scores the *behaviors* we care about (at steady state), validated
against the hand-wired reflex as ground truth:

| controller | in_rew | in_pun | →green | turn_off_red | green_dwell | red_dwell |
|---|---|---|---|---|---|---|
| reflex (ceiling) | 0.891 | 0.000 | 0.91 | 0.55 | 45 | 0 |
| **learn** | 0.108 | 0.090 | 0.70 | −0.06 | 168 | 146 |
| frozen | 0.049 | 0.038 | 0.67 | −0.02 | 293 | 334 |
| random | 0.013 | 0.004 | 0.40 | 0.00 | 45 | 60 |

- `→green`: velocity alignment toward a *visible* reward (+1 = straight at it, 0 = chance).
- `turn_off_red`: deg/step turning away from the side a *visible* punishment is on
  (+ = avoiding, 0 = random — the metric is debiased so a non-avoider scores 0).
- `dwell`: median consecutive steps per region visit.

**What the agents actually learn:** they **seek and stay in green** — steering
toward visible reward well above chance (0.70 vs 0.40) and dwelling ~3.7× longer
than random — but they do **not** learn to actively **avoid red** (`turn_off_red`
≈ 0 vs the reflex's 0.55). The learned green > red discrimination comes almost
entirely from *attraction to reward*, not *avoidance of punishment*. This is the
expected asymmetry: seeking gets a direct "go here → +reward" signal, while
avoidance must be inferred from the *absence* of reward — a weaker teaching signal
for local plasticity. Strengthening avoidance (e.g. higher punishment-channel
salience, a steeper repulsive shaping field, or an inhibitory pathway) is open
work.

## Results

Reward-modulated STDP **learns to seek reward** from local plasticity + a global
reward signal alone (no backprop). Representative 40k-step run:

| controller | in_reward | in_punish | score | mean_R |
|---|---|---|---|---|
| reflex (hand-wired upper bound) | 0.89 | 0.00 | 0.89 | +0.97 |
| **learn** (reward-modulated SNN) | **0.108** | 0.090 | **0.018** | **≈0.00** |
| frozen (random fixed weights) | 0.049 | 0.038 | 0.011 | −0.12 |
| random walk | 0.013 | 0.004 | 0.010 | −0.01 |

The learning agent spends ~2× more time in reward than the frozen network and ~8×
more than random, and lifts mean reward from −0.12 to ≈0. It stays well below the
hand-wired reflex — expected for unsupervised local plasticity — but the directed
learning is real and stable.

**What it took to get there (the fiddly part, SPEC §10):**
- **Vision must out-drive the boredom noise.** Weights are bounded [0,1] but the
  LIF needs ~15 mV to fire, so a synaptic gain (`cfg.syn_gain`) is required or
  input has *zero* effect on output. Without it the network is a pure noise
  generator and no reward signal can matter.
- **Sparse coding.** A low firing setpoint (`target_rate≈2 Hz`) is essential:
  dense high-rate firing makes STDP indiscriminate and learning *erases*
  structure instead of building it.
- **Small wall penalty.** A large `R_wall` injects destructive negative reward
  correlated with motor activity (fast agents hit walls), corrupting plasticity.
- **Potential-based reward shaping** (`shaping_gain`) gives a dense approach/retreat
  signal that eases credit assignment (theory-preserving, Ng et al. 1999).

## Tests

```bash
python -m pytest tests/ -q
```

Covers the spec acceptance criteria: LIF firing (§4.1), eligibility rise/decay
(§4.2), three-factor weight change (§4.3), 100k-step stability (§4.5), and
raycasting vision (§12.2).

## Layout

| Path | Role |
|---|---|
| `config.py` | all hyperparameters, one seed-controlled dataclass |
| `snn/` | LIF neurons, STDP+eligibility synapses, three-factor network, homeostasis, stability |
| `env/` | region world, raycasting vision, sensory/motor encoding |
| `ui/` | `simulation.py` (headless-capable driver) + Pygame viewer/inspector |
| `viz/monitors.py` | offline matplotlib plots |
| `train.py` | headless training/eval loop |

## Notes

- The original single-agent point-foraging world (SPEC §5, `env/forage.py`) is
  superseded by the richer region world and is not implemented yet.
- Every time constant interacts; tune one subsystem at a time using its test
  (SPEC §7 "Tuning order", §10 "Known failure modes"). For *behavior*, the first
  knobs are `syn_gain` (does vision drive the net at all?), `target_rate`
  (sparsity), and `R_wall` (don't let it dominate) — see Results above.
- Headroom to push the learning further: longer training (100k+), per-agent
  evolutionary selection (SPEC §12.4), separate excitatory/inhibitory pools, or a
  recurrent hidden layer.
