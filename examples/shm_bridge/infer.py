"""BlackJAX variational inference for transient-excitation-based bridge damage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from blackjax.vi import fullrank_vi
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.shm_bridge.transient import (
    DAMAGE_LABELS,
    initial_severity,
    true_severity,
)
from examples.shm_bridge.validate_gradient import build_predictor
from utils.plot_style import BLUE, ERROR_CMAP, RED

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
})

MAX_SEVERITY = 0.0124
NOISE_FRACTION = 0.02
NOISE_FLOOR = 1.0e-4 * 9.80665



def logit(probability):
    probability = jnp.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return jnp.log(probability) - jnp.log1p(-probability)


def severity_from_unconstrained(position):
    return MAX_SEVERITY * jax.nn.sigmoid(position).reshape(len(DAMAGE_LABELS))


def whitened_rmse(prediction, target, noise_scale):
    return float(np.sqrt(np.mean(((prediction - target) / noise_scale) ** 2)))


def build_problem(seed: int = 2026):
    mesh, predict = build_predictor()
    clean = predict(jnp.asarray(true_severity()))
    channel_rms = jnp.sqrt(jnp.mean(clean**2, axis=0))
    noise_scale = jnp.sqrt((NOISE_FRACTION * channel_rms) ** 2 + NOISE_FLOOR**2)
    noise = noise_scale * jax.random.normal(
        jax.random.PRNGKey(seed), clean.shape, dtype=jnp.float64
    )
    observation = jax.lax.stop_gradient(clean + noise)
    initial_position = logit(jnp.asarray(initial_severity()) / MAX_SEVERITY)

    def logdensity(position):
        severity = severity_from_unconstrained(position)
        prediction = predict(severity)
        normalized_residual = (prediction - observation) / noise_scale
        log_likelihood = -0.5 * jnp.sum(normalized_residual**2)
        probability = jax.nn.sigmoid(position)
        log_prior = jnp.sum(jnp.log(probability) + jnp.log1p(-probability))
        return log_likelihood + log_prior

    return mesh, predict, logdensity, observation, clean, noise_scale, initial_position


def diagnostics(predict, clean, noise_scale):
    truth = true_severity()
    initial_prediction = np.asarray(predict(jnp.asarray(initial_severity())))
    signal_rmse = whitened_rmse(initial_prediction, np.asarray(clean), np.asarray(noise_scale))
    num_parameters = len(DAMAGE_LABELS)
    jacobian = np.empty((clean.size, num_parameters))
    step = 2.0e-5
    for index in range(num_parameters):
        plus = truth.copy()
        minus = truth.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            (np.asarray(predict(jnp.asarray(plus))) - np.asarray(predict(jnp.asarray(minus))))
            / (2.0 * step)
            / np.asarray(noise_scale)
        ).reshape(-1)
    weighted = jacobian * 1.0e-3
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    correlation = np.corrcoef(weighted, rowvar=False)
    max_correlation = np.max(
        np.abs(correlation - np.eye(num_parameters))
    )
    return {
        "initial_prediction": initial_prediction,
        "signal_rmse": signal_rmse,
        "singular_values": singular_values,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "correlation": correlation,
        "max_correlation": float(max_correlation),
    }


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


def plot_results(loss, samples, output):
    """One-row summary: convergence trace and a 3x3 posterior matrix."""
    samples = 1.0e3 * np.asarray(samples).reshape(len(samples), -1)
    truth = 1.0e3 * np.asarray(true_severity()).reshape(-1)
    bounds = np.full(truth.size, 1.0e3 * MAX_SEVERITY)
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

def run(num_steps, num_elbo_samples, posterior_samples, learning_rate):
    mesh, predict, logdensity, observation, clean, noise_scale, initial_position = build_problem()
    diagnostic = diagnostics(predict, clean, noise_scale)
    print(f"Initial signal/noise RMSE   : {diagnostic['signal_rmse']:.3f}")
    print(f"Sensitivity singular values: {diagnostic['singular_values']}")
    print(f"Sensitivity condition no.  : {diagnostic['condition_number']:.3f}")
    print(f"Max sensitivity correlation: {diagnostic['max_correlation']:.3f}")
    if diagnostic["signal_rmse"] < 3.0:
        raise RuntimeError("initial and true damage states are not distinguishable")
    if diagnostic["condition_number"] > 100.0:
        raise RuntimeError("dynamic inverse problem is ill-conditioned")

    constant_steps = max(1, int(0.8 * num_steps))
    schedule = optax.join_schedules(
        [
            optax.constant_schedule(learning_rate),
            optax.cosine_decay_schedule(
                learning_rate, max(1, num_steps - constant_steps), alpha=0.1
            ),
        ],
        [constant_steps],
    )
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    algorithm = fullrank_vi.as_top_level_api(
        logdensity, optimizer, num_samples=num_elbo_samples
    )
    dimension = initial_position.size
    chol_params = jnp.concatenate((
        jnp.full(dimension, -1.2),
        jnp.zeros(dimension * (dimension - 1) // 2),
    ))
    state = algorithm.init(initial_position)
    state = state._replace(
        mu=initial_position,
        chol_params=chol_params,
        opt_state=optimizer.init((initial_position, chol_params)),
    )

    step = jax.jit(algorithm.step)
    key = jax.random.PRNGKey(42)
    loss = np.empty(num_steps + 1)
    start = perf_counter()
    for iteration in range(num_steps):
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        loss[iteration] = float(info.elbo)
        if not np.isfinite(loss[iteration]):
            raise FloatingPointError(f"non-finite VI loss at step {iteration}")
        if iteration == 0 or (iteration + 1) % max(1, num_steps // 10) == 0:
            print(
                f"iteration {iteration:4d}/{num_steps}: "
                f"training loss = {loss[iteration]:.6e}"
            )
    key, final_key = jax.random.split(key)
    _, final_info = step(final_key, state)
    loss[num_steps] = float(final_info.elbo)
    elapsed = perf_counter() - start

    key, sample_key = jax.random.split(key)
    unconstrained_samples = algorithm.sample(sample_key, state, posterior_samples)
    severity_samples = np.asarray(
        jax.vmap(severity_from_unconstrained)(unconstrained_samples)
    )
    posterior_mean = severity_samples.mean(axis=0)
    posterior_std = severity_samples.std(axis=0)
    lower, upper = np.quantile(severity_samples, [0.025, 0.975], axis=0)
    coverage = (true_severity() >= lower) & (true_severity() <= upper)

    initial_prediction = diagnostic["initial_prediction"]
    posterior_prediction = np.asarray(predict(jnp.asarray(posterior_mean)))
    observation_np = np.asarray(observation)
    clean_np = np.asarray(clean)
    noise_np = np.asarray(noise_scale)
    initial_clean_rmse = whitened_rmse(initial_prediction, clean_np, noise_np)
    posterior_clean_rmse = whitened_rmse(posterior_prediction, clean_np, noise_np)
    initial_data_rmse = whitened_rmse(initial_prediction, observation_np, noise_np)
    posterior_data_rmse = whitened_rmse(posterior_prediction, observation_np, noise_np)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / "shm_bridge.npz"
    np.savez(
        result_path,
        loss=loss,
        posterior_samples=severity_samples,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        posterior_correlation=np.corrcoef(severity_samples, rowvar=False),
        lower=lower,
        upper=upper,
        coverage=coverage,
        true_severity=true_severity(),
        initial_severity=initial_severity(),
        observation=observation_np,
        clean_observation=clean_np,
        noise_scale=noise_np,
        initial_clean_rmse=initial_clean_rmse,
        posterior_clean_rmse=posterior_clean_rmse,
        initial_data_rmse=initial_data_rmse,
        posterior_data_rmse=posterior_data_rmse,
        sensitivity_singular_values=diagnostic["singular_values"],
        sensitivity_condition_number=diagnostic["condition_number"],
        max_sensitivity_correlation=diagnostic["max_correlation"],
    )
    figure_path = ROOT / "figs" / "shm_bridge_inference.png"
    plot_results(
        loss,
        severity_samples,
        figure_path,
    )

    print()
    print(f"Damage elements             : {mesh.candidate_damage_elements}")
    num_parameters = len(DAMAGE_LABELS)
    num_vi_parameters = num_parameters + num_parameters * (num_parameters + 1) // 2
    print(f"Physical parameters         : {num_parameters}")
    print(f"VI parameters               : {num_vi_parameters}")
    print(f"Iterations                  : {num_steps}")
    print(f"ELBO samples/step           : {num_elbo_samples}")
    print(f"Learning rate               : {learning_rate:.4f}")
    print(f"Elapsed                     : {elapsed:.2f} s")
    print(f"Initial severity [mm]       : {1.0e3 * initial_severity()}")
    print(f"True severity [mm]          : {1.0e3 * true_severity()}")
    print(f"Posterior mean [mm]         : {1.0e3 * posterior_mean}")
    print(f"Parameter error, initial    : {np.linalg.norm(initial_severity()-true_severity()):.6e}")
    print(f"Parameter error, posterior  : {np.linalg.norm(posterior_mean-true_severity()):.6e}")
    print(f"Clean RMSE, initial         : {initial_clean_rmse:.3f} noise std")
    print(f"Clean RMSE, posterior       : {posterior_clean_rmse:.3f} noise std")
    print(f"Noisy-data RMSE, initial    : {initial_data_rmse:.3f} noise std")
    print(f"Noisy-data RMSE, posterior  : {posterior_data_rmse:.3f} noise std")
    print(f"95% marginal coverage       : {coverage.sum()}/{coverage.size}")
    print(f"Results                     : {result_path}")
    print(f"Figure                      : {figure_path}")
    return loss, severity_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--elbo-samples", type=int, default=4)
    parser.add_argument("--posterior-samples", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=5.0e-2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.steps, args.elbo_samples, args.posterior_samples, args.learning_rate)
