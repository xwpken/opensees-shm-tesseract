"""Validate the three-parameter transient-response gradient end to end."""

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

from examples.shm_bridge.experiment import (
    DAMAGE_LABELS,
    corrosion_from_severity,
    initial_severity,
    make_section_spec,
    make_transient_program,
    observe,
    section_properties_to_parameters,
    true_severity,
)
from examples.shm_bridge.model import make_bridge_mesh
from utils.gradient_check import compute_gradient_metrics

SECTION_API = ROOT / "tesseracts" / "section_properties" / "tesseract_api.py"
OPENSEES_API = ROOT / "tesseracts" / "opensees_ddm" / "tesseract_api.py"


def build_predictor():
    mesh = make_bridge_mesh()
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
    program = make_transient_program(mesh)

    def predict(severity):
        properties = apply_tesseract(
            section_tesseract,
            {
                "section_spec": section_spec,
                "corrosion": corrosion_from_severity(severity),
            },
            vmap_method="sequential",
        )["properties"]
        parameters = section_properties_to_parameters(mesh, properties, pristine)
        response = apply_tesseract(
            opensees_tesseract,
            {"program": program, "parameters": parameters},
            vmap_method="sequential",
        )["responses"]
        return observe(response)

    return mesh, predict


def main():
    mesh, predict = build_predictor()
    truth = jnp.asarray(true_severity())
    target = jax.lax.stop_gradient(predict(truth))
    scale = jax.lax.stop_gradient(
        jnp.maximum(jnp.sqrt(jnp.mean(target**2, axis=0)), 1.0e-9)
    )

    def loss(severity):
        residual = (predict(severity) - target) / scale
        return 0.5 * jnp.mean(residual**2)

    initial = jnp.asarray(initial_severity())
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    start = perf_counter()
    value, gradient = value_and_grad(initial)
    gradient.block_until_ready()
    elapsed = perf_counter() - start

    finite_difference = np.empty(len(DAMAGE_LABELS))
    start_fd = perf_counter()
    for index in range(len(DAMAGE_LABELS)):
        step = 2.0e-5
        plus = np.asarray(initial).copy()
        minus = np.asarray(initial).copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            float(loss(jnp.asarray(plus))) - float(loss(jnp.asarray(minus)))
        ) / (2.0 * step)
    elapsed_fd = perf_counter() - start_fd

    metrics = compute_gradient_metrics(np.asarray(gradient), finite_difference)
    np.testing.assert_allclose(
        np.asarray(gradient), finite_difference, rtol=3.0e-3, atol=1.0e-7
    )
    print(f"Damage elements : {mesh.candidate_damage_elements}")
    print(f"Physical params : {len(DAMAGE_LABELS)} local corrosion severities")
    print(f"Observations    : {target.shape[0]} time samples x {target.shape[1]} absolute-acceleration channels")
    print(f"Loss            : {float(value):.6e}")
    print(f"Max abs error   : {metrics['max_absolute']:.3e}")
    print(f"Relative L2     : {metrics['relative_l2']:.3e}")
    print(f"Max rel error   : {metrics['max_active_relative']:.3e}")
    print(f"Sign mismatch   : {metrics['sign_mismatches']}")
    print(f"JAX/Tesseract   : {elapsed:.2f} s")
    print(f"Finite difference: {elapsed_fd:.2f} s")
    print("PASS")


if __name__ == "__main__":
    main()
