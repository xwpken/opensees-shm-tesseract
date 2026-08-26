"""Posterior plots for the static-frame inference example."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from utils.plot_style import BLUE, ERROR_CMAP, RED


def plot_inference_summary(
    vi_loss,
    samples,
    *,
    truth,
    initial,
    parameter_labels,
    response_metrics,
    output,
):
    samples = np.asarray(samples).reshape(len(samples), -1)
    truth = np.asarray(truth).reshape(-1)
    initial = np.asarray(initial).reshape(-1)
    mean = samples.mean(axis=0)
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)

    figure, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))
    steps = np.arange(len(vi_loss))
    window = min(20, max(1, len(vi_loss) // 5))
    if len(vi_loss) >= window:
        smooth = np.convolve(vi_loss, np.ones(window) / window, mode="valid")
        axes[0].plot(
            np.arange(window - 1, len(vi_loss)),
            smooth,
            color=RED,
            linewidth=2.0,
            label=f"{window}-step moving average",
            zorder=2,
        )
    axes[0].plot(
        steps,
        vi_loss,
        color=BLUE,
        linewidth=1.1,
        alpha=0.70,
        label="Stochastic variational loss",
        zorder=3,
    )
    axes[0].set(
        xlabel="Iterations",
        ylabel="Variational loss",
        title="Variational inference convergence",
        ylim=(0.0, None),
    )
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=19)

    positions = np.arange(truth.size)
    axes[1].errorbar(
        positions,
        1.0e3 * mean,
        yerr=1.0e3 * np.vstack((mean - lower, upper - mean)),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.6,
        alpha=0.95,
        capsize=3,
        label=r"$95\%$ VI",
    )
    axes[1].scatter(positions, 1.0e3 * truth, marker="x", color=RED, label="Truth")
    axes[1].scatter(
        positions,
        1.0e3 * initial,
        marker="s",
        facecolors="none",
        edgecolors=BLUE,
        label="Initial",
    )
    displayed = 1.0e3 * np.concatenate((lower, upper, truth, initial))
    padding = 0.08 * np.ptp(displayed)
    axes[1].set_xticks(positions, parameter_labels)
    axes[1].set(
        ylabel="Thickness loss [mm]",
        title="Approximate posterior marginals",
        ylim=(0.0, float(displayed.max() + max(2.2, padding))),
    )
    axes[1].grid(True, axis="y", alpha=0.2)
    axes[1].legend(
        loc="upper center",
        ncol=3,
        fontsize=21,
        handlelength=1.2,
        handletextpad=0.45,
        columnspacing=0.9,
        borderpad=0.35,
    )
    axes[1].tick_params(axis="x", labelsize=18)

    x = np.arange(2)
    width = 0.24
    axes[2].bar(
        x - width / 2,
        response_metrics["clean"],
        width,
        color=BLUE,
        label="Clean truth",
    )
    axes[2].bar(
        x + width / 2,
        response_metrics["observed"],
        width,
        color=RED,
        label="Noisy data",
    )
    axes[2].axhline(1.0, color="0.35", linestyle="--", linewidth=1.2, label="Noise level")
    axes[2].set_xticks(x, ("Initial", "Posterior mean"))
    axes[2].set(
        ylabel="Whitened RMSE",
        title="Response-fit improvement",
        ylim=(0.0, None),
    )
    axes[2].grid(True, axis="y", alpha=0.2)
    axes[2].legend(fontsize=19)

    for axis in axes:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.tick_params(axis="both", which="major", width=1.0, length=4)

    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_posterior_matrix(
    samples,
    *,
    truth,
    bounds,
    parameter_labels,
    output,
):
    """Plot bounded marginals, joint densities, and posterior samples."""
    samples = 1.0e3 * np.asarray(samples).reshape(len(samples), -1)
    truth = 1.0e3 * np.asarray(truth).reshape(-1)
    bounds = 1.0e3 * np.asarray(bounds).reshape(-1)
    n_parameters = samples.shape[1]

    display_limits = []
    for index in range(n_parameters):
        lower = min(float(samples[:, index].min()), float(truth[index]))
        upper = max(float(samples[:, index].max()), float(truth[index]))
        padding = max(0.10 * (upper - lower), 0.12)
        physical_upper = float(bounds[index])
        limits = (
            max(0.0, lower - padding),
            min(physical_upper, upper + padding),
        )
        kde_bandwidth = float(np.sqrt(np.cov(samples[:, index]))) * float(
            gaussian_kde(samples[:, index]).factor
        )
        edge_padding = max(3.0 * kde_bandwidth, 0.12)
        display_limits.append((
            limits[0] - edge_padding if limits[0] == 0.0 else limits[0],
            limits[1] + edge_padding if limits[1] == physical_upper else limits[1],
        ))

    indices = np.linspace(0, len(samples) - 1, min(1500, len(samples)), dtype=int)
    scatter_samples = samples[indices]
    figure, axes = plt.subplots(
        n_parameters,
        n_parameters,
        figsize=(20, 20),
        squeeze=False,
    )

    for row in range(n_parameters):
        for column in range(n_parameters):
            axis = axes[row, column]
            x_display_limits, y_display_limits = display_limits[column], display_limits[row]

            if row == column:
                grid = np.linspace(*x_display_limits, 300)
                kde = gaussian_kde(samples[:, row])
                density = kde(grid)
                density /= max(float(density.max()), 1.0e-30)
                colors = ERROR_CMAP(density)
                axis.fill_between(grid, density, color=BLUE, alpha=0.20, linewidth=0)
                for index in range(grid.size - 1):
                    axis.plot(
                        grid[index:index + 2],
                        density[index:index + 2],
                        color=colors[index],
                        linewidth=2.0,
                    )
                axis.axvline(truth[row], color=RED, linewidth=1.8)
                axis.set(xlim=x_display_limits, ylim=(0.0, 1.08))
                axis.set_yticks([])
            elif row > column:
                pair = samples[:, [column, row]]
                xx, yy = np.meshgrid(
                    np.linspace(*x_display_limits, 110),
                    np.linspace(*y_display_limits, 110),
                )
                kde = gaussian_kde(pair.T)
                density = kde(
                    np.vstack((xx.ravel(), yy.ravel()))
                ).reshape(xx.shape)
                density /= max(float(density.max()), 1.0e-30)
                cloud_floor = 0.015
                levels = np.linspace(cloud_floor, 1.0, 41)
                axis.contourf(
                    xx,
                    yy,
                    density,
                    levels=levels,
                    cmap=ERROR_CMAP,
                    vmin=cloud_floor,
                    vmax=1.0,
                    antialiased=True,
                )
                axis.contour(
                    xx,
                    yy,
                    density,
                    levels=(0.25, 0.50, 0.75),
                    colors=(ERROR_CMAP(0.08),),
                    linewidths=0.85,
                    alpha=0.85,
                )
                axis.scatter(
                    truth[column], truth[row], marker="x", s=58,
                    linewidths=2.0, color=RED, zorder=5,
                )
                axis.set(xlim=x_display_limits, ylim=y_display_limits)
            else:
                axis.scatter(
                    scatter_samples[:, column],
                    scatter_samples[:, row],
                    s=13,
                    alpha=0.82,
                    linewidths=0,
                    color=ERROR_CMAP(0.0),
                    rasterized=True,
                )
                axis.scatter(
                    truth[column], truth[row], marker="x", s=58,
                    linewidths=2.0, color=RED, zorder=5,
                )
                axis.set(xlim=x_display_limits, ylim=y_display_limits)

            axis.grid(True, alpha=0.12, linewidth=0.55)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            if row != column:
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.tick_params(labelsize=16, length=3)

            if row < n_parameters - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(parameter_labels[column] + " [mm]", fontsize=18)
            if column > 0 or row == column:
                axis.set_yticklabels([])
            else:
                axis.set_ylabel(parameter_labels[row] + " [mm]", fontsize=18)

    handles = [
        Line2D(
            [0], [0],
            color=ERROR_CMAP(0.08),
            lw=1.5,
            label="Joint KDE",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            color=ERROR_CMAP(0.0), alpha=0.75, label="Posterior samples",
        ),
        Line2D(
            [0], [0], marker="x", linestyle="none", markersize=7,
            markeredgewidth=1.5, color=RED, label="Truth",
        ),
    ]
    figure.suptitle("Posterior dependence", fontsize=27, y=0.995)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        fontsize=20,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.07,
        top=0.935,
        wspace=0.08,
        hspace=0.08,
    )

    # The top-left diagonal panel uses density on its vertical axis. Overlay a
    # transparent physical axis so the first row retains the same parameter
    # ticks and label as every other row of the matrix.
    top_left = axes[0, 0].get_position()
    row_axis = figure.add_axes(
        [top_left.x0, top_left.y0, top_left.width, top_left.height],
        frameon=False,
    )
    row_axis.patch.set_visible(False)
    row_axis.set_xlim(0.0, 1.0)
    row_axis.set_ylim(display_limits[0])
    row_axis.xaxis.set_visible(False)
    row_axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    row_axis.tick_params(
        axis="y",
        which="major",
        labelsize=16,
        length=3,
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
    )
    row_axis.set_ylabel(parameter_labels[0] + " [mm]", fontsize=18)
    row_axis.set_zorder(axes[0, 0].get_zorder() + 1)

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
