"""Plots for transient bridge corrosion inference."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from utils.plot_style import BLUE, ERROR_CMAP, RED, set_plot_style

set_plot_style(18)


def _bounded_kde_1d(values, grid, lower, upper):
    """Reflection-corrected KDE on a bounded physical interval."""
    kde = gaussian_kde(values)
    return kde(grid) + kde(2.0 * lower - grid) + kde(2.0 * upper - grid)


def _bounded_kde_2d(pair, xx, yy, x_bounds, y_bounds):
    """Reflection-corrected bivariate KDE within physical bounds."""
    kde = gaussian_kde(pair.T)
    density = np.zeros_like(xx)
    x_lower, x_upper = x_bounds
    y_lower, y_upper = y_bounds
    for x_eval in (xx, 2.0 * x_lower - xx, 2.0 * x_upper - xx):
        for y_eval in (yy, 2.0 * y_lower - yy, 2.0 * y_upper - yy):
            points = np.vstack((x_eval.ravel(), y_eval.ravel()))
            density += kde(points).reshape(xx.shape)
    return density


def plot_results(loss, samples, *, truth, upper_bound, output):
    """One-row summary: convergence trace and a 3x3 posterior matrix."""
    samples = 1.0e3 * np.asarray(samples).reshape(len(samples), -1)
    truth = 1.0e3 * np.asarray(truth).reshape(-1)
    bounds = np.full(truth.size, 1.0e3 * upper_bound)
    n_parameters = truth.size

    figure = plt.figure(figsize=(20.5, 8.6))
    outer = figure.add_gridspec(
        1, 2, width_ratios=(1.0, 1.0), wspace=0.055,
        left=0.050, right=0.975, bottom=0.105, top=0.875,
    )
    trace = figure.add_subplot(outer[0, 0])
    posterior_grid = outer[0, 1].subgridspec(
        n_parameters, n_parameters,
        wspace=0.040, hspace=0.065,
    )
    axes = np.empty((n_parameters, n_parameters), dtype=object)
    for row in range(n_parameters):
        for column in range(n_parameters):
            axes[row, column] = figure.add_subplot(posterior_grid[row, column])

    # Left panel: convergence.
    steps = np.arange(loss.size)
    window = min(15, max(1, loss.size // 5))
    if loss.size >= window:
        smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
        trace.plot(
            np.arange(window - 1, loss.size), smooth,
            color=RED, linewidth=2.7,
            label=f"{window}-step moving average", zorder=2,
        )
    trace.plot(
        steps, loss, color=BLUE, linewidth=1.35, alpha=0.72,
        label="Stochastic variational loss", zorder=3,
    )
    trace.set_xlabel("Iterations", fontsize=27)
    trace.set_ylabel("Variational loss", fontsize=27)
    trace.set_ylim(0.0, 4000.0)
    trace.set_yticks(np.arange(0.0, 4001.0, 1000.0))
    trace.tick_params(axis="both", labelsize=23, width=1.1, length=5)
    trace.grid(True, alpha=0.20)
    trace.legend(fontsize=19, loc="upper right")
    trace.set_box_aspect(0.82)

    # Local physical plotting ranges, clipped to [0, upper bound].
    limits = []
    for index in range(n_parameters):
        lower = min(float(samples[:, index].min()), float(truth[index]))
        upper = max(float(samples[:, index].max()), float(truth[index]))
        padding = max(0.10 * (upper - lower), 0.12)
        limits.append((
            max(0.0, lower - padding),
            min(float(bounds[index]), upper + padding),
        ))

    scatter_indices = np.linspace(
        0, len(samples) - 1, min(2200, len(samples)), dtype=int
    )
    scatter_samples = samples[scatter_indices]
    normalized_density = Normalize(vmin=0.0, vmax=1.0)

    for row in range(n_parameters):
        for column in range(n_parameters):
            axis = axes[row, column]
            x_limits, y_limits = limits[column], limits[row]

            if row == column:
                grid = np.linspace(*x_limits, 300)
                density = _bounded_kde_1d(
                    samples[:, row], grid, 0.0, bounds[row]
                )
                density /= max(float(density.max()), 1.0e-30)
                colors = ERROR_CMAP(normalized_density(density))
                axis.fill_between(
                    grid, 0.0, density,
                    color=BLUE, alpha=0.20, linewidth=0,
                )
                # Colour the marginal curve by the same continuous colormap.
                for index in range(grid.size - 1):
                    axis.plot(
                        grid[index:index + 2], density[index:index + 2],
                        color=colors[index], linewidth=2.0,
                    )
                axis.axvline(truth[row], color=RED, linewidth=1.8)
                axis.set(xlim=x_limits, ylim=(0.0, 1.08))
                axis.set_yticks([])

            elif row > column:
                pair = samples[:, [column, row]]
                xx, yy = np.meshgrid(
                    np.linspace(*x_limits, 110),
                    np.linspace(*y_limits, 110),
                )
                density = _bounded_kde_2d(
                    pair, xx, yy,
                    (0.0, bounds[column]), (0.0, bounds[row]),
                )
                density /= max(float(density.max()), 1.0e-30)
                # Keep the low-density exterior white, while the visible
                # posterior cloud itself still starts at the blue endpoint of
                # the shared red--blue scale.
                cloud_floor = 0.015
                axis.contourf(
                    xx, yy, density,
                    levels=np.linspace(cloud_floor, 1.0, 41),
                    cmap=ERROR_CMAP,
                    norm=Normalize(vmin=cloud_floor, vmax=1.0),
                    antialiased=True,
                )
                axis.contour(
                    xx, yy, density,
                    levels=(0.25, 0.50, 0.75),
                    colors=(ERROR_CMAP(0.08),),
                    linewidths=0.85,
                    alpha=0.85,
                )
                axis.scatter(
                    truth[column], truth[row], marker="x", s=58,
                    linewidths=2.0, color=RED, zorder=5,
                )
                axis.set(xlim=x_limits, ylim=y_limits)

            else:
                axis.scatter(
                    scatter_samples[:, column], scatter_samples[:, row],
                    s=13, alpha=0.82, linewidths=0,
                    color=ERROR_CMAP(0.0), rasterized=True,
                )
                axis.scatter(
                    truth[column], truth[row], marker="x", s=58,
                    linewidths=2.0, color=RED, zorder=5,
                )
                axis.set(xlim=x_limits, ylim=y_limits)

            axis.grid(True, alpha=0.10, linewidth=0.55)
            axis.tick_params(axis="both", labelsize=22, length=4.0)
            axis.locator_params(axis="x", nbins=3)
            if row != column:
                axis.locator_params(axis="y", nbins=3)

            if row < n_parameters - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(
                    rf"$s_{{{column + 1}}}$ [mm]", fontsize=25, labelpad=7
                )
            if column > 0 or row == column:
                axis.set_yticklabels([])
            else:
                axis.set_ylabel(
                    rf"$s_{{{row + 1}}}$ [mm]", fontsize=25, labelpad=7
                )

    figure.text(
        0.265, 0.905, "Variational inference convergence",
        ha="center", va="center", fontsize=27,
    )
    figure.text(
        0.745, 0.905, "Posterior dependence",
        ha="center", va="center", fontsize=27,
    )
    # The top-left diagonal panel uses density vertically. Overlay a transparent
    # physical s1 axis so the first row has the same left-side ticks and label
    # as the other two rows without changing the marginal-density coordinates.
    top_left = axes[0, 0].get_position()
    row_axis = figure.add_axes(
        [top_left.x0, top_left.y0, top_left.width, top_left.height],
        frameon=False,
    )
    row_axis.patch.set_visible(False)
    row_axis.set_xlim(0.0, 1.0)
    row_axis.set_ylim(limits[0])
    row_axis.xaxis.set_visible(False)
    row_axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    row_axis.tick_params(
        axis="y", which="major", labelsize=22, length=4.0,
        left=True, labelleft=True, right=False, labelright=False,
    )
    row_axis.set_ylabel(r"$s_1$ [mm]", fontsize=25, labelpad=7)
    row_axis.set_zorder(axes[0, 0].get_zorder() + 1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output
