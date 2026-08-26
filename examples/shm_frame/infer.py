"""BlackJAX variational inference for eight corrosion parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.shm_frame.model import (
    DAMAGE_LABELS,
    corrosion_upper_bounds,
    initial_corrosion,
    make_opensees_program,
    make_section_spec,
    observation_noise_scale,
    observe,
    true_corrosion,
)
from examples.shm_frame.plots import plot_inference_summary, plot_posterior_matrix
from utils.plot_style import set_plot_style

set_plot_style(25)

SECTION_API = ROOT / "tesseracts" / "section_properties" / "tesseract_api.py"
OPENSEES_API = ROOT / "tesseracts" / "opensees_ddm" / "tesseract_api.py"
PARAMETER_LABELS = tuple(
    f"S{site + 1}\n{kind}"
    for site in range(len(DAMAGE_LABELS))
    for kind in ("flange", "web")
)


def logit(probability):
    probability = jnp.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return jnp.log(probability) - jnp.log1p(-probability)


def corrosion_from_unconstrained(position):
    bounds = jnp.asarray(corrosion_upper_bounds())
    return jax.nn.sigmoid(position).reshape(bounds.shape) * bounds


def build_problem(
    seed: int = 2026,
    noise_fraction: float = 0.02,
    add_observation_noise: bool = True,
):
    section_spec = make_section_spec()
    section_tesseract = Tesseract.from_tesseract_api(SECTION_API)
    opensees_tesseract = Tesseract.from_tesseract_api(OPENSEES_API)

    pristine = apply_tesseract(
        section_tesseract,
        {
            "section_spec": section_spec,
            "corrosion": jnp.zeros((1, 2), dtype=jnp.float64),
        },
    )["properties"][0]
    program = make_opensees_program(float(pristine[0]), float(pristine[1]))

    def predict(corrosion):
        properties = apply_tesseract(
            section_tesseract,
            {"section_spec": section_spec, "corrosion": corrosion},
            vmap_method="sequential",
        )["properties"]
        raw = apply_tesseract(
            opensees_tesseract,
            {"program": program, "parameters": properties[:, :2].reshape(-1)},
            vmap_method="sequential",
        )["responses"]
        return observe(raw)

    clean_observation = predict(jnp.asarray(true_corrosion()))
    noise_scale = observation_noise_scale(clean_observation, noise_fraction)
    if add_observation_noise:
        noise = noise_scale * jax.random.normal(
            jax.random.PRNGKey(seed), clean_observation.shape, dtype=jnp.float64
        )
    else:
        noise = jnp.zeros_like(clean_observation)
    observation = jax.lax.stop_gradient(clean_observation + noise)

    bounds = jnp.asarray(corrosion_upper_bounds())
    initial_position = logit(jnp.asarray(initial_corrosion()) / bounds).reshape(-1)

    def logdensity(position):
        prediction = predict(corrosion_from_unconstrained(position))
        normalized_residual = (prediction - observation) / noise_scale
        log_likelihood = -0.5 * jnp.sum(normalized_residual**2)
        # Uniform prior in physical corrosion coordinates. The sigmoid Jacobian
        # is included because VI is performed in unconstrained coordinates.
        probability = jax.nn.sigmoid(position)
        log_prior = jnp.sum(jnp.log(probability) + jnp.log1p(-probability))
        return log_likelihood + log_prior

    return (
        logdensity,
        predict,
        observation,
        clean_observation,
        noise_scale,
        initial_position,
    )


def whitened_rmse(prediction, target, noise_scale):
    return float(np.sqrt(np.mean(((prediction - target) / noise_scale) ** 2)))


def identifiability_diagnostics(predict, clean_observation, noise_scale):
    """Check signal strength and independence of all sensitivities."""
    truth = jnp.asarray(true_corrosion())
    initial_prediction = np.asarray(predict(jnp.asarray(initial_corrosion())))
    signal_rmse = whitened_rmse(
        initial_prediction,
        np.asarray(clean_observation),
        np.asarray(noise_scale),
    )

    def predict_flat(parameters):
        return predict(parameters.reshape(truth.shape))

    jacobian = np.asarray(jax.jacrev(predict_flat)(truth.reshape(-1)))
    weighted_jacobian = jacobian * 1.0e-3 / np.asarray(noise_scale)[:, None]
    singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
    correlations = np.corrcoef(weighted_jacobian, rowvar=False)
    max_correlation = np.max(np.abs(correlations - np.eye(correlations.shape[0])))
    return {
        "initial_prediction": initial_prediction,
        "signal_rmse": signal_rmse,
        "rank": int(np.linalg.matrix_rank(weighted_jacobian)),
        "singular_values": singular_values,
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "max_correlation": float(max_correlation),
    }


def run(
    num_steps: int,
    num_elbo_samples: int,
    posterior_samples: int,
    learning_rate: float,
    noise_fraction: float,
    add_observation_noise: bool,
):
    (
        logdensity,
        predict,
        observation,
        clean_observation,
        noise_scale,
        initial_position,
    ) = build_problem(
        noise_fraction=noise_fraction,
        add_observation_noise=add_observation_noise,
    )
    diagnostics = identifiability_diagnostics(predict, clean_observation, noise_scale)
    print(f"Initial signal/noise RMSE   : {diagnostics['signal_rmse']:.3f}")
    num_parameters = initial_position.size
    print(f"Sensitivity rank            : {diagnostics['rank']}/{num_parameters}")
    print(f"Sensitivity singular values : {diagnostics['singular_values']}")
    print(f"Sensitivity condition no.   : {diagnostics['condition_number']:.3f}")
    print(f"Max sensitivity correlation : {diagnostics['max_correlation']:.3f}")
    if diagnostics["signal_rmse"] < 3.0:
        raise RuntimeError("initial damage hypothesis is not distinguishable from truth")
    if diagnostics["rank"] < num_parameters or diagnostics["condition_number"] > 75.0:
        raise RuntimeError("the corrosion parameters are not identifiable")
    if diagnostics["max_correlation"] > 0.8:
        raise RuntimeError("the sensitivity directions are too strongly correlated")

    constant_steps = max(1, int(0.8 * num_steps))
    schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(learning_rate),
            optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=max(1, num_steps - constant_steps),
                alpha=0.1,
            ),
        ],
        boundaries=[constant_steps],
    )
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    algorithm = blackjax.fullrank_vi(
        logdensity,
        optimizer,
        num_samples=num_elbo_samples,
    )
    state = algorithm.init(initial_position)
    dimension = initial_position.size
    initial_cholesky = jnp.concatenate((
        -1.2 * jnp.ones(dimension),
        jnp.zeros(dimension * (dimension - 1) // 2),
    ))
    state = state._replace(
        mu=initial_position,
        chol_params=initial_cholesky,
        opt_state=optimizer.init((initial_position, initial_cholesky)),
    )

    step = jax.jit(algorithm.step)
    key = jax.random.PRNGKey(42)
    vi_loss = np.empty(num_steps + 1)
    start = perf_counter()
    for iteration in range(num_steps):
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        vi_loss[iteration] = float(info.elbo)
        if not np.isfinite(vi_loss[iteration]):
            raise FloatingPointError(f"non-finite ELBO at iteration {iteration}")
        if iteration == 0 or iteration % max(1, num_steps // 10) == 0:
            print(
                f"iteration {iteration:4d}/{num_steps}: "
                f"training loss = {vi_loss[iteration]:.6e}"
            )

    key, final_key = jax.random.split(key)
    _, final_info = step(final_key, state)
    vi_loss[num_steps] = float(final_info.elbo)
    if not np.isfinite(vi_loss[num_steps]):
        raise FloatingPointError(f"non-finite ELBO at iteration {num_steps}")
    print(
        f"iteration {num_steps:4d}/{num_steps}: "
        f"training loss = {vi_loss[num_steps]:.6e}"
    )
    elapsed = perf_counter() - start

    key, sample_key = jax.random.split(key)
    unconstrained_samples = algorithm.sample(sample_key, state, posterior_samples)
    corrosion_samples = np.asarray(
        jax.vmap(corrosion_from_unconstrained)(unconstrained_samples)
    )
    posterior_mean = corrosion_samples.mean(axis=0)
    posterior_std = corrosion_samples.std(axis=0)
    lower, upper = np.quantile(corrosion_samples, [0.025, 0.975], axis=0)
    coverage = (true_corrosion() >= lower) & (true_corrosion() <= upper)
    initial_error = np.linalg.norm(initial_corrosion() - true_corrosion())
    posterior_error = np.linalg.norm(posterior_mean - true_corrosion())

    prediction_mean = np.asarray(predict(jnp.asarray(posterior_mean)))
    initial_prediction = diagnostics["initial_prediction"]
    observation_np = np.asarray(observation)
    clean_np = np.asarray(clean_observation)
    noise_scale_np = np.asarray(noise_scale)
    initial_data_rmse = whitened_rmse(initial_prediction, observation_np, noise_scale_np)
    posterior_data_rmse = whitened_rmse(prediction_mean, observation_np, noise_scale_np)
    initial_clean_rmse = whitened_rmse(initial_prediction, clean_np, noise_scale_np)
    posterior_clean_rmse = whitened_rmse(prediction_mean, clean_np, noise_scale_np)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    np.savez(
        results_dir / "shm_frame.npz",
        vi_loss=vi_loss,
        posterior_samples=corrosion_samples,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        coverage=coverage,
        true_corrosion=true_corrosion(),
        initial_corrosion=initial_corrosion(),
        corrosion_upper_bounds=corrosion_upper_bounds(),
        observation=observation_np,
        clean_observation=clean_np,
        noise_scale=noise_scale_np,
        noise_fraction=noise_fraction,
        initial_data_rmse=initial_data_rmse,
        posterior_data_rmse=posterior_data_rmse,
        initial_clean_rmse=initial_clean_rmse,
        posterior_clean_rmse=posterior_clean_rmse,
        sensitivity_singular_values=diagnostics["singular_values"],
        sensitivity_condition_number=diagnostics["condition_number"],
        max_sensitivity_correlation=diagnostics["max_correlation"],
    )
    summary_path = ROOT / "figs" / "shm_frame_summary.png"
    posterior_path = ROOT / "figs" / "shm_frame_posterior.png"
    response_metrics = {
        "clean": (initial_clean_rmse, posterior_clean_rmse),
        "observed": (initial_data_rmse, posterior_data_rmse),
    }
    plot_inference_summary(
        vi_loss,
        corrosion_samples,
        truth=true_corrosion(),
        initial=initial_corrosion(),
        parameter_labels=PARAMETER_LABELS,
        response_metrics=response_metrics,
        output=summary_path,
    )
    matrix_labels = tuple(
        rf"$S_{{{site + 1}}}:t_{{{symbol}}}$"
        for site in range(len(DAMAGE_LABELS))
        for symbol in ("f", "w")
    )
    plot_posterior_matrix(
        corrosion_samples,
        truth=true_corrosion(),
        bounds=corrosion_upper_bounds(),
        parameter_labels=matrix_labels,
        output=posterior_path,
    )

    print()
    print(f"Iterations                    : {num_steps}")
    print(f"Observation noise fraction    : {noise_fraction:.4f}")
    print(f"Random observation noise      : {add_observation_noise}")
    print(f"ELBO samples/step             : {num_elbo_samples}")
    print("Variational family            : Gaussian")
    print(f"Elapsed                       : {elapsed:.2f} s")
    print(f"VI initial corrosion [mm]     :\n{1.0e3 * initial_corrosion()}")
    print("Physical priors               : flange U(0,12), web U(0,8) mm")
    print(f"Posterior mean [mm]           :\n{1.0e3 * posterior_mean}")
    print(f"Parameter error, initial      : {initial_error:.6e}")
    print(f"Parameter error, posterior    : {posterior_error:.6e}")
    print(f"Clean response RMSE, initial  : {initial_clean_rmse:.3f} noise std")
    print(f"Clean RMSE at posterior mean  : {posterior_clean_rmse:.3f} noise std")
    print(f"Noisy-data RMSE, initial      : {initial_data_rmse:.3f} noise std")
    print(f"Data RMSE at posterior mean   : {posterior_data_rmse:.3f} noise std")
    print(f"95% marginal coverage         : {coverage.sum()}/{coverage.size}")
    print(f"Results                       : {results_dir / 'shm_frame.npz'}")
    print(f"Summary figure                : {summary_path}")
    print(f"Posterior matrix              : {posterior_path}")
    return vi_loss, corrosion_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--elbo-samples", type=int, default=8)
    parser.add_argument("--posterior-samples", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-2)
    parser.add_argument("--noise-fraction", type=float, default=2.0e-2)
    parser.add_argument(
        "--no-observation-noise",
        dest="add_observation_noise",
        action="store_false",
    )
    parser.set_defaults(add_observation_noise=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.steps,
        args.elbo_samples,
        args.posterior_samples,
        args.learning_rate,
        args.noise_fraction,
        args.add_observation_noise,
    )
