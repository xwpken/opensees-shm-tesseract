"""End-to-end cyclic elastoplastic gradient-validation model."""

from __future__ import annotations

import json

import numpy as np

N_SPRINGS = 12
PARAMETER_LABELS = (r"$\Delta t_f$", r"$\Delta t_w$")
PEAK_FORCE = 4.0e6
YIELD_STRESS = 250.0e6
ELASTIC_MODULUS = 200.0e9
HARDENING_RATIO = 0.02


def _parameter(index: int):
    return {"parameter": index}


def _cyclic_load_path() -> np.ndarray:
    """Full symmetric reversal, sampled around yielding and load reversals."""
    return np.concatenate([
        np.linspace(0.0, 1.0, 21),
        np.linspace(1.0, -1.0, 41)[1:],
        np.linspace(-1.0, 1.0, 41)[1:],
        np.linspace(1.0, 0.0, 21)[1:],
    ])


def make_section_spec() -> str:
    """Nominal I-section used by the section-properties Tesseract."""
    return json.dumps({
        "d": 0.30,
        "b": 0.25,
        "tf": 0.020,
        "tw": 0.020,
        "root_radius": 0.006,
        "root_points": 8,
        "mesh_size": 5.0e-4,
    })


def corrosion_parameters() -> np.ndarray:
    """Independent flange and web losses for the twelve truss elements [m]."""
    phase = np.linspace(0.0, 2.0 * np.pi, N_SPRINGS, endpoint=False)
    flange = 0.0050 + 0.0030 * np.sin(phase)
    web = 0.0035 + 0.0020 * np.cos(phase)
    return np.column_stack((flange, web))


def make_problem():
    """Return section specification, OpenSees program, corrosion, and force history."""
    load_path = _cyclic_load_path()
    commands = [{"name": "model", "args": ["basic", "-ndm", 1, "-ndf", 1]}]

    for node in range(1, N_SPRINGS + 2):
        commands.append({"name": "node", "args": [node, float(node - 1)]})
    commands.append({"name": "fix", "args": [1, 1]})

    for element in range(1, N_SPRINGS + 1):
        commands.extend([
            {
                "name": "uniaxialMaterial",
                "args": [
                    "Steel01",
                    element,
                    YIELD_STRESS,
                    ELASTIC_MODULUS,
                    HARDENING_RATIO,
                ],
            },
            {
                "name": "element",
                "args": ["truss", element, element, element + 1, _parameter(element - 1), element],
            },
            {
                "name": "parameter",
                "args": [element, "element", element, "A"],
            },
        ])

    commands.extend([
        {
            "name": "timeSeries",
            "args": [
                "Path",
                1,
                "-dt",
                1.0,
                "-values",
                *load_path.tolist(),
                "-useLast",
            ],
        },
        {"name": "pattern", "args": ["Plain", 1, 1]},
        {"name": "load", "args": [N_SPRINGS + 1, PEAK_FORCE]},
        {"name": "constraints", "args": ["Plain"]},
        {"name": "numberer", "args": ["Plain"]},
        {"name": "system", "args": ["UmfPack"]},
        {"name": "test", "args": ["NormDispIncr", 1.0e-10, 100]},
        {"name": "algorithm", "args": ["NewtonLineSearch"]},
        {"name": "integrator", "args": ["LoadControl", 1.0]},
    ])

    program = {
        "commands": commands,
        "analysis": {"type": "Static", "steps": load_path.size - 1},
        "parameter_tags": list(range(1, N_SPRINGS + 1)),
        "responses": [
            {"type": "node_disp", "node": N_SPRINGS + 1, "dof": 1},
        ],
    }
    force_history = PEAK_FORCE * load_path[1:]
    return make_section_spec(), json.dumps(program), corrosion_parameters(), force_history
