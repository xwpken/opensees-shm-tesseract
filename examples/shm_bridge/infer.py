"""BlackJAX variational inference for transient-excitation-based bridge damage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from blackjax.vi import fullrank_vi

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.shm_bridge.experiment import (
    DAMAGE_LABELS,
    initial_severity,
    true_severity,
)
from examples.shm_bridge.plots import plot_results
from examples.shm_bridge.validate_gradient import build_predictor

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
        truth=true_severity(),
        upper_bound=MAX_SEVERITY,
        output=figure_path,
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
