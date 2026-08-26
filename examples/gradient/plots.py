"""Plots for the cyclic gradient-validation example."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.plot_style import set_plot_style


def save_hysteresis_animation(displacement, force, output, *, interval: int = 60):
    """Save an animated force-displacement trajectory with a moving marker."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    set_plot_style(20)
    displacement = np.asarray(displacement, dtype=float).reshape(-1)
    force = np.asarray(force, dtype=float).reshape(-1)
    if displacement.shape != force.shape or displacement.size < 2:
        raise ValueError("displacement and force must have matching non-empty histories")

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 5.0))
    axis.fill(displacement, force, color="#F3DFD5", alpha=0.24, edgecolor="none", zorder=0)
    axis.plot(displacement, force, color="#5B8FB9", linewidth=1.0, alpha=0.22, zorder=1)
    trail, = axis.plot([], [], color="#5B8FB9", linewidth=2.2, zorder=2)
    point, = axis.plot(
        [], [], "o", color="#D96C5F", markeredgecolor="white",
        markeredgewidth=1.0, markersize=8, zorder=3,
    )

    x_padding = 0.06 * max(float(np.ptp(displacement)), 1.0)
    y_padding = 0.06 * max(float(np.ptp(force)), 1.0)
    axis.set_xlim(float(displacement.min() - x_padding), float(displacement.max() + x_padding))
    axis.set_ylim(float(force.min() - y_padding), float(force.max() + y_padding))
    axis.set_xlabel("Tip displacement [m]")
    axis.set_ylabel("Applied force [N]")
    axis.set_title("Cyclic elastoplastic response")
    axis.grid(True, alpha=0.2)
    figure.tight_layout()

    def update(frame):
        trail.set_data(displacement[: frame + 1], force[: frame + 1])
        point.set_data([displacement[frame]], [force[frame]])
        return trail, point

    animation = FuncAnimation(
        figure,
        update,
        frames=displacement.size,
        interval=interval,
        blit=True,
        repeat=True,
    )
    animation.save(path, writer=PillowWriter(fps=max(1.0, 1000.0 / interval)))
    plt.close(figure)
    return path
