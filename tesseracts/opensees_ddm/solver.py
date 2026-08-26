"""Problem-independent OpenSeesPy runner with native DDM Jacobian contractions."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import openseespy.opensees as ops

_NODE_RESPONSES = {
    "node_disp": (ops.nodeDisp, ops.sensNodeDisp),
    "node_vel": (ops.nodeVel, ops.sensNodeVel),
    "node_accel": (ops.nodeAccel, ops.sensNodeAccel),
}
_CONTROLLED_COMMANDS = {
    "wipe",
    "analysis",
    "analyze",
    "computeGradients",
    "sensitivityAlgorithm",
}


def parse_program(program: str | dict[str, Any]) -> dict[str, Any]:
    spec = json.loads(program) if isinstance(program, str) else program
    required = {"commands", "analysis", "parameter_tags", "responses"}
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"program is missing fields: {sorted(missing)}")
    return spec


def output_shape(program: str | dict[str, Any]) -> tuple[int, int]:
    spec = parse_program(program)
    return int(spec["analysis"]["steps"]), len(spec["responses"])


def _resolve(value: Any, parameters: np.ndarray) -> Any:
    if isinstance(value, dict) and set(value) == {"parameter"}:
        return float(parameters[int(value["parameter"])])
    if isinstance(value, list):
        return [_resolve(item, parameters) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, parameters) for key, item in value.items()}
    return value


def build_domain(spec: dict[str, Any], parameters: np.ndarray, sensitivity: bool = False) -> None:
    ops.wipe()
    for command in spec["commands"]:
        name = command["name"]
        if name.startswith("_") or name in _CONTROLLED_COMMANDS:
            raise ValueError(f"command {name!r} is managed by the runner")
        function = getattr(ops, name, None)
        if function is None:
            raise ValueError(f"unknown OpenSees command: {name}")
        function(*_resolve(command.get("args", []), parameters))

    if sensitivity:
        ops.sensitivityAlgorithm("-computeAtEachStep")
    ops.analysis(spec["analysis"]["type"])


def analyze_step(analysis: dict[str, Any]) -> None:
    if analysis["type"].lower() == "transient":
        result = ops.analyze(1, float(analysis["dt"]))
    else:
        result = ops.analyze(1)
    if result != 0:
        raise RuntimeError(f"OpenSees analysis failed with code {result}")


def _response(response: dict[str, Any]) -> float:
    try:
        value_function = _NODE_RESPONSES[response["type"]][0]
    except KeyError as error:
        raise ValueError(f"unsupported response type: {response['type']}") from error
    return float(value_function(int(response["node"]), int(response["dof"])))


def _response_sensitivity(response: dict[str, Any], parameter_tag: int) -> float:
    sensitivity_function = _NODE_RESPONSES[response["type"]][1]
    return float(
        sensitivity_function(
            int(response["node"]),
            int(response["dof"]),
            int(parameter_tag),
        )
    )


def apply(program: str | dict[str, Any], parameters) -> np.ndarray:
    """Run the serialized OpenSees program and return response history (time, channel)."""
    spec = parse_program(program)
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    tags = tuple(int(tag) for tag in spec["parameter_tags"])
    if theta.size != len(tags):
        raise ValueError("parameters and parameter_tags must have the same length")

    build_domain(spec, theta, sensitivity=False)
    values = np.empty(output_shape(spec), dtype=float)
    for step in range(values.shape[0]):
        analyze_step(spec["analysis"])
        values[step] = [_response(response) for response in spec["responses"]]
    return values


def vector_jacobian_product(
    program: str | dict[str, Any],
    parameters,
    response_cotangent,
) -> np.ndarray:
    """Return (d responses / d parameters).T @ response_cotangent using DDM."""
    spec = parse_program(program)
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    tags = tuple(int(tag) for tag in spec["parameter_tags"])
    cotangent = np.asarray(response_cotangent, dtype=float)
    expected_shape = output_shape(spec)

    if theta.size != len(tags):
        raise ValueError("parameters and parameter_tags must have the same length")
    if cotangent.shape != expected_shape:
        raise ValueError(
            f"response cotangent has shape {cotangent.shape}; expected {expected_shape}"
        )

    build_domain(spec, theta, sensitivity=True)
    gradient = np.zeros(theta.size, dtype=float)

    for step in range(expected_shape[0]):
        analyze_step(spec["analysis"])
        for response_index, response in enumerate(spec["responses"]):
            weight = cotangent[step, response_index]
            if weight == 0.0:
                continue
            gradient += weight * np.asarray(
                [_response_sensitivity(response, tag) for tag in tags],
                dtype=float,
            )

    return gradient
