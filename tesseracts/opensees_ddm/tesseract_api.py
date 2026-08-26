"""Tesseract API for a model-independent OpenSees DDM program runner."""

from typing import Any

import solver
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    program: str
    parameters: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    responses: Differentiable[Array[(None, None), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(
        responses=solver.apply(inputs.program, inputs.parameters),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    if "parameters" not in vjp_inputs or "responses" not in vjp_outputs:
        return {}
    return {
        "parameters": solver.vector_jacobian_product(
            inputs.program,
            inputs.parameters,
            cotangent_vector["responses"],
        )
    }


def abstract_eval(abstract_inputs):
    steps, channels = solver.output_shape(abstract_inputs.program)
    return {
        "responses": ShapeDType(
            shape=(steps, channels),
            dtype="float64",
        )
    }
