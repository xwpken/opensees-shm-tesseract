"""Validate the complete corrosion-to-hysteretic-work gradient."""

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

from examples.gradient.model import N_SPRINGS, PARAMETER_LABELS, make_problem
from examples.gradient.plots import save_hysteresis_animation
from utils.gradient_check import compute_gradient_metrics, plot_gradient_check

SECTION_API = ROOT / "tesseracts" / "section_properties" / "tesseract_api.py"
OPENSEES_API = ROOT / "tesseracts" / "opensees_ddm" / "tesseract_api.py"


def hysteretic_work(displacement, force):
    """Discrete path integral integral(F du), including the initial zero state."""
    displacement = jnp.concatenate((jnp.zeros(1), displacement.reshape(-1)))
    force = jnp.concatenate((jnp.zeros(1), force.reshape(-1)))
    return jnp.sum(0.5 * (force[1:] + force[:-1]) * jnp.diff(displacement))


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
    return np.asarray(result, dtype=float), float(np.median(samples))


def main():
    section_spec, program, corrosion_np, force_np = make_problem()
    corrosion = jnp.asarray(corrosion_np)
    force = jnp.asarray(force_np)
    section_tesseract = Tesseract.from_tesseract_api(SECTION_API)
    opensees_tesseract = Tesseract.from_tesseract_api(OPENSEES_API)

    def response(theta):
        properties = apply_tesseract(
            section_tesseract,
            {"section_spec": section_spec, "corrosion": theta},
            vmap_method="sequential",
        )["properties"]
        return apply_tesseract(
            opensees_tesseract,
            {"program": program, "parameters": properties[:, 0]},
            vmap_method="sequential",
        )["responses"][:, 0]

    def objective(theta):
        return hysteretic_work(response(theta), force)

    gradient_function = jax.jit(jax.grad(objective))
    gradient_jax, time_jax = timed(lambda: gradient_function(corrosion))

    def finite_difference_gradient():
        gradient = np.empty_like(corrosion_np)
        for index in np.ndindex(corrosion_np.shape):
            step = 1.0e-6
            plus = corrosion_np.copy()
            minus = corrosion_np.copy()
            plus[index] += step
            minus[index] -= step
            gradient[index] = (
                float(objective(jnp.asarray(plus)))
                - float(objective(jnp.asarray(minus)))
            ) / (2.0 * step)
        return gradient

    gradient_fd, time_fd = timed(finite_difference_gradient)
    np.testing.assert_allclose(gradient_jax, gradient_fd, rtol=2.0e-5, atol=1.0e-2)

    displacement = np.asarray(response(corrosion), dtype=float)
    energy = float(objective(corrosion))
    metrics = compute_gradient_metrics(gradient_jax, gradient_fd)
    validation_path = plot_gradient_check(
        gradient_jax,
        gradient_fd,
        parameter_shape=(N_SPRINGS, len(PARAMETER_LABELS)),
        parameter_labels=PARAMETER_LABELS,
        times=(time_jax, time_fd),
        output=ROOT / "figs" / "gradient_validation.png",
        actual_label="composed gradient",
        method_label="Two\nTesseracts",
        gradient_unit="J/m",
    )

    plot_displacement = np.concatenate(([0.0], displacement))
    plot_force = np.concatenate(([0.0], force_np))
    hysteresis_animation_path = save_hysteresis_animation(
        plot_displacement,
        plot_force,
        ROOT / "figs" / "gradient_hysteresis.gif",
        interval=60,
    )

    print("Model        : 12 serial Steel01 truss elements")
    print("Pipeline     : corrosion → section properties → areas → OpenSees → work")
    print(f"Parameters   : {corrosion_np.size} = 12 × (flange loss, web loss)")
    print(f"Load steps   : {force_np.size}")
    print(f"Hysteretic work: {energy:.6e} J")
    print(f"Relative L2  : {metrics['relative_l2']:.3e}")
    print(f"Max rel error: {metrics['max_active_relative']:.3e}")
    print(f"Sign mismatch: {metrics['sign_mismatches']}")
    print()
    print(f"JAX/two Tesseracts : {time_jax:.2f} s")
    print(f"Finite difference   : {time_fd:.2f} s")
    print(f"Speedup             : {time_fd / time_jax:.2f}x")
    print(f"Hysteresis GIF      : {hysteresis_animation_path}")
    print(f"Validation figure   : {validation_path}")
    print("PASS")


if __name__ == "__main__":
    main()
