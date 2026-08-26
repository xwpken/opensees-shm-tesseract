"""Tesseract API for locally corroded I-section analysis."""

from typing import Any

import section_solver
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    section_spec: str
    corrosion: Differentiable[Array[(None, 2), Float64]]


class OutputSchema(BaseModel):
    properties: Differentiable[Array[(None, 6), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(
        properties=section_solver.apply(inputs.section_spec, inputs.corrosion),
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    if "corrosion" not in vjp_inputs or "properties" not in vjp_outputs:
        return {}
    return {
        "corrosion": section_solver.vector_jacobian_product(
            inputs.section_spec,
            inputs.corrosion,
            cotangent_vector["properties"],
        )
    }


def abstract_eval(abstract_inputs):
    n_sections = abstract_inputs.corrosion.shape[0]
    return {
        "properties": ShapeDType(shape=(n_sections, 6), dtype="float64"),
    }
