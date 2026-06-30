"""Offline matplotlib monitors (SPEC §9).

Static plots for after-the-fact analysis, complementing the live Pygame UI:
spike raster, weight-matrix heatmap, reward / (R - baseline) curve, and a sample
eligibility-trace decay. Each takes plain arrays so it can be used from train.py
or a notebook without importing the whole sim.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_raster(spike_history, ax=None, title="spike raster"):
    """spike_history: (T, N) bool array (time x neurons)."""
    ax = ax or plt.gca()
    sh = np.asarray(spike_history)
    t, n = np.nonzero(sh)
    ax.scatter(t, n, s=2, marker="|")
    ax.set_xlabel("step")
    ax.set_ylabel("neuron")
    ax.set_title(title)
    return ax


def plot_weight_heatmap(W, ax=None, title="weights W[post, pre]"):
    ax = ax or plt.gca()
    im = ax.imshow(np.asarray(W), aspect="auto", cmap="viridis")
    ax.set_xlabel("pre")
    ax.set_ylabel("post")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_reward(reward, delta=None, ax=None, title="reward over time"):
    ax = ax or plt.gca()
    ax.plot(reward, label="R", lw=0.8)
    if delta is not None:
        ax.plot(delta, label="R - baseline", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title(title)
    return ax


def plot_eligibility(e_trace, ax=None, title="eligibility decay"):
    ax = ax or plt.gca()
    ax.plot(e_trace, lw=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("e")
    ax.set_title(title)
    return ax


def dashboard(spike_history, W, reward, delta=None, e_trace=None):
    """Compose the common monitors into one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_raster(spike_history, axes[0, 0])
    plot_weight_heatmap(W, axes[0, 1])
    plot_reward(reward, delta, axes[1, 0])
    if e_trace is not None:
        plot_eligibility(e_trace, axes[1, 1])
    fig.tight_layout()
    return fig
