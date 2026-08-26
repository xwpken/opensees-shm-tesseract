"""Two-Tesseract static SHM pipeline: section FEM -> OpenSees DDM."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.shm_frame.model import (
    DAMAGE_LABELS,
    LOAD_CASES,
    OBSERVATION_LABELS,
    initial_corrosion,
    make_opensees_program,
    make_section_spec,
    observe,
    true_corrosion,
)
from utils.gradient_check import compute_gradient_metrics, plot_gradient_check

SECTION_API = ROOT / "tesseracts" / "section_properties" / "tesseract_api.py"
OPENSEES_API = ROOT / "tesseracts" / "opensees_ddm" / "tesseract_api.py"


def timed(function, repeats=3):
    function()
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        samples.append(perf_counter() - start)
    return result, float(np.median(samples))


def main():
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

    def predict_raw(corrosion):
        section_properties = apply_tesseract(
            section_tesseract,
            {"section_spec": section_spec, "corrosion": corrosion},
        )["properties"]
        area_and_inertia = section_properties[:, :2].reshape(-1)
        return apply_tesseract(
            opensees_tesseract,
            {"program": program, "parameters": area_and_inertia},
        )["responses"]

    def predict(corrosion):
        return observe(predict_raw(corrosion))

    truth = jnp.asarray(true_corrosion())
    observation = jax.lax.stop_gradient(predict(truth))
    response_scale = jax.lax.stop_gradient(
        jnp.maximum(jnp.abs(observation), 0.1 * jnp.median(jnp.abs(observation)))
    )

    def loss(corrosion):
        residual = (predict(corrosion) - observation) / response_scale
        return 0.5 * jnp.mean(residual**2)

    guess = jnp.asarray(initial_corrosion())
    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    (loss_value, gradient), elapsed = timed(lambda: value_and_gradient(guess))

    # Independent end-to-end central-difference check of the composed pipeline.
    gradient_fd = np.empty_like(np.asarray(guess))
    guess_np = np.asarray(guess)
    start = perf_counter()
    for index in np.ndindex(guess_np.shape):
        step = 1.0e-6
        plus = guess_np.copy()
        minus = guess_np.copy()
        plus[index] += step
        minus[index] -= step
        gradient_fd[index] = (
            float(loss(jnp.asarray(plus))) - float(loss(jnp.asarray(minus)))
        ) / (2.0 * step)
    finite_difference_time = perf_counter() - start

    np.testing.assert_allclose(
        np.asarray(gradient), gradient_fd, rtol=2.0e-3, atol=1.0e-7
    )
    metrics = compute_gradient_metrics(np.asarray(gradient), gradient_fd)
    figure_path = plot_gradient_check(
        np.asarray(gradient),
        gradient_fd,
        parameter_shape=(len(DAMAGE_LABELS), 2),
        parameter_labels=("flange", "web"),
        times=(elapsed, finite_difference_time),
        output=ROOT / "figs" / "shm_frame_gradient.png",
    )

    print(f"Problem      : infer {len(DAMAGE_LABELS)} spatially separated local corrosion sites")
    print(f"Damage sites : {', '.join(DAMAGE_LABELS)}")
    print(f"Damage vars  : {2 * len(DAMAGE_LABELS)} = {len(DAMAGE_LABELS)} sites × (flange, web loss)")
    print(f"Load cases   : {len(LOAD_CASES)}")
    print(f"Observations : {len(OBSERVATION_LABELS)} direct nodal responses")
    print(f"Loss         : {float(loss_value):.6e}")
    print(f"Max abs error: {metrics['max_absolute']:.3e}")
    print(f"Relative L2  : {metrics['relative_l2']:.3e}")
    print(f"Max rel error: {metrics['max_active_relative']:.3e}")
    print(f"Sign mismatch: {metrics['sign_mismatches']}")
    print(f"JAX gradient : {1e3 * elapsed:.2f} ms")
    print(f"Finite diff  : {1e3 * finite_difference_time:.2f} ms")
    print(f"Figure       : {figure_path}")
    print("PASS")


if __name__ == "__main__":
    main()
