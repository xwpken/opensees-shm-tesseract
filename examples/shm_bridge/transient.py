"""Transient-excitation forward model for three-parameter bridge damage inference."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from examples.shm_bridge.model import (
    CONCRETE_DENSITY,
    CONCRETE_E,
    CONCRETE_NU,
    DECK_THICKNESS,
    SECTION_PROPERTIES,
    STEEL_E,
    STEEL_G,
    TRUSS_AREAS,
    BridgeMesh,
)

STEEL_DENSITY = 7850.0
DAMPING_RATIO = 0.02
REFERENCE_FREQUENCY = 3.0
GROUND_MOTION_DT = 0.02
GROUND_MOTION_DURATION = 1.5
GROUND_MOTION_PGA = 0.20
EXCITATION_DURATION = 0.50
OBSERVATION_STRIDE = 2
ACCELEROMETER_NODE_TAGS = (21, 27, 31, 36, 38, 46)
ACCELEROMETER_DOFS = (1, 2, 3)

DAMAGE_LABELS = (
    "Left near top chord",
    "Middle far bottom chord",
    "Right far top chord",
)


def make_section_spec() -> str:
    """Reference I-section used to obtain corrosion reduction factors."""
    return json.dumps({
        "d": 0.30,
        "b": 0.25,
        "tf": 0.020,
        "tw": 0.020,
        "root_radius": 0.006,
        "root_points": 8,
        "mesh_size": 1.0e-4,
    })


def initial_severity() -> np.ndarray:
    return np.full(len(DAMAGE_LABELS), 0.0020)


def true_severity() -> np.ndarray:
    return np.asarray([0.0090, 0.0100, 0.0110])


def corrosion_from_severity(severity):
    severity = jnp.asarray(severity).reshape(len(DAMAGE_LABELS))
    return jnp.stack((severity, 0.8 * severity), axis=1)


def make_base_excitation(
    *,
    dt: float = GROUND_MOTION_DT,
    duration: float = GROUND_MOTION_DURATION,
    pga: float = GROUND_MOTION_PGA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic two-component designed transient base-acceleration histories in g."""
    time = np.arange(0.0, duration + 0.5 * dt, dt)
    envelope = np.where(
        time <= EXCITATION_DURATION,
        np.sin(np.pi * time / EXCITATION_DURATION) ** 2,
        0.0,
    )
    longitudinal = envelope * (
        np.sin(2.0 * np.pi * 4.72 * time)
        + 0.45 * np.sin(2.0 * np.pi * 6.45 * time + 0.60)
        + 0.15 * np.sin(2.0 * np.pi * 2.20 * time + 1.10)
    )
    transverse = envelope * (
        np.sin(2.0 * np.pi * 4.72 * time + 0.40)
        + 0.50 * np.sin(2.0 * np.pi * 6.45 * time + 1.20)
        + 0.15 * np.sin(2.0 * np.pi * 3.10 * time)
    )
    longitudinal *= pga / np.max(np.abs(longitudinal))
    transverse *= 0.85 * pga / np.max(np.abs(transverse))
    longitudinal[0] = transverse[0] = 0.0
    longitudinal[-1] = transverse[-1] = 0.0
    return time, longitudinal, transverse


def _nominal_line_properties(element) -> tuple[float, float, float, float]:
    if element.formulation == "truss":
        return (TRUSS_AREAS[element.group], 0.0, 0.0, 0.0)
    return SECTION_PROPERTIES[element.group]



def section_properties_to_parameters(mesh: BridgeMesh, section_properties, pristine):
    """Scale each candidate's nominal OpenSees properties by section-FE ratios."""
    properties = jnp.asarray(section_properties)
    pristine = jnp.asarray(pristine)
    area_ratio = properties[:, 0] / pristine[0]
    strong_ratio = properties[:, 1] / pristine[1]
    weak_ratio = properties[:, 2] / pristine[2]

    line_by_tag = {element.tag: element for element in mesh.lines}
    values = []
    for index, tag in enumerate(mesh.candidate_damage_elements):
        element = line_by_tag[tag]
        area, iy, iz, _ = _nominal_line_properties(element)
        if element.formulation == "truss":
            values.append(jnp.asarray(area) * area_ratio[index])
        else:
            values.extend((
                jnp.asarray(area) * area_ratio[index],
                jnp.asarray(iy) * weak_ratio[index],
                jnp.asarray(iz) * strong_ratio[index],
            ))
    return jnp.stack(values)


def _lumped_node_masses(mesh: BridgeMesh) -> dict[int, float]:
    masses = {tag: 0.0 for tag in mesh.nodes}
    for element in mesh.lines:
        i, j = element.nodes
        area = _nominal_line_properties(element)[0]
        length = np.linalg.norm(np.subtract(mesh.nodes[j], mesh.nodes[i]))
        member_mass = STEEL_DENSITY * area * length
        masses[i] += 0.5 * member_mass
        masses[j] += 0.5 * member_mass
    for shell in mesh.shells:
        points = np.asarray([mesh.nodes[tag] for tag in shell.nodes])
        area = np.linalg.norm(points[1] - points[0]) * np.linalg.norm(points[3] - points[0])
        shell_mass = CONCRETE_DENSITY * DECK_THICKNESS * area
        for node in shell.nodes:
            masses[node] += 0.25 * shell_mass
    return masses


def _parameter(index: int):
    return {"parameter": index}


def make_transient_program(mesh: BridgeMesh) -> str:
    """Serialize the bridge as a transient OpenSees-DDM program."""
    time, acceleration_x, acceleration_y = make_base_excitation()
    commands: list[dict] = [{"name": "model", "args": ["basic", "-ndm", 3, "-ndf", 6]}]
    for tag, xyz in mesh.nodes.items():
        commands.append({"name": "node", "args": [tag, *xyz]})
    for tag, restraint in mesh.supports.items():
        commands.append({"name": "fix", "args": [tag, *restraint]})
    for tag, mass in _lumped_node_masses(mesh).items():
        commands.append({"name": "mass", "args": [tag, mass, mass, mass, 0.0, 0.0, 0.0]})

    commands.extend([
        {"name": "uniaxialMaterial", "args": ["Elastic", 1, STEEL_E]},
        {
            "name": "section",
            "args": [
                "ElasticMembranePlateSection", 100, CONCRETE_E,
                CONCRETE_NU, DECK_THICKNESS, 0.0,
            ],
        },
        {"name": "geomTransf", "args": ["Linear", 1, 0.0, 0.0, 1.0]},
        {"name": "geomTransf", "args": ["Linear", 2, 1.0, 0.0, 0.0]},
    ])

    candidates = set(mesh.candidate_damage_elements)
    parameter_index = 0
    parameter_specs: list[tuple[int, int, str]] = []
    for element in mesh.lines:
        i, j = element.nodes
        area, iy, iz, torsion = _nominal_line_properties(element)
        if element.formulation == "truss":
            if element.tag in candidates:
                area_value = _parameter(parameter_index)
                parameter_specs.append((parameter_index + 1, element.tag, "A"))
                parameter_index += 1
            else:
                area_value = area
            commands.append({
                "name": "element",
                "args": ["truss", element.tag, i, j, area_value, 1],
            })
            continue

        direction = np.subtract(mesh.nodes[j], mesh.nodes[i])
        transform = 2 if abs(direction[2]) > 0.95 * np.linalg.norm(direction) else 1
        if element.tag in candidates:
            area_value = _parameter(parameter_index)
            iy_value = _parameter(parameter_index + 1)
            iz_value = _parameter(parameter_index + 2)
            section_tag = 20000 + element.tag
            integration_tag = 30000 + element.tag
            commands.extend([
                {
                    "name": "section",
                    "args": [
                        "Elastic", section_tag, STEEL_E, area_value, iz_value,
                        iy_value, STEEL_G, torsion,
                    ],
                },
                {
                    "name": "beamIntegration",
                    "args": ["Lobatto", integration_tag, section_tag, 4],
                },
                {
                    "name": "element",
                    "args": [
                        "dispBeamColumn", element.tag, i, j,
                        transform, integration_tag,
                    ],
                },
            ])
            for offset, name in enumerate(("A", "Iy", "Iz")):
                parameter_specs.append((parameter_index + offset + 1, element.tag, name))
            parameter_index += 3
        else:
            commands.append({
                "name": "element",
                "args": [
                    "elasticBeamColumn", element.tag, i, j, area, STEEL_E,
                    STEEL_G, torsion, iy, iz, transform,
                ],
            })

    for shell in mesh.shells:
        commands.append({
            "name": "element",
            "args": ["ShellMITC4", shell.tag, *shell.nodes, 100],
        })
    for tag, element_tag, name in parameter_specs:
        commands.append({
            "name": "parameter",
            "args": [tag, "element", element_tag, name],
        })

    gravity = 9.80665
    commands.extend([
        {
            "name": "timeSeries",
            "args": [
                "Path", 101, "-dt", GROUND_MOTION_DT,
                "-values", *acceleration_x.tolist(), "-factor", gravity,
            ],
        },
        {
            "name": "timeSeries",
            "args": [
                "Path", 102, "-dt", GROUND_MOTION_DT,
                "-values", *acceleration_y.tolist(), "-factor", gravity,
            ],
        },
        {"name": "pattern", "args": ["UniformExcitation", 101, 1, "-accel", 101]},
        {"name": "pattern", "args": ["UniformExcitation", 102, 2, "-accel", 102]},
        {
            "name": "rayleigh",
            "args": [
                2.0 * DAMPING_RATIO * 2.0 * np.pi * REFERENCE_FREQUENCY,
                0.0, 0.0, 0.0,
            ],
        },
        {"name": "constraints", "args": ["Plain"]},
        {"name": "numberer", "args": ["RCM"]},
        {"name": "system", "args": ["UmfPack"]},
        {"name": "test", "args": ["NormDispIncr", 1.0e-9, 20]},
        {"name": "algorithm", "args": ["Linear"]},
        {"name": "integrator", "args": ["Newmark", 0.5, 0.25]},
    ])

    responses = [
        {"type": "node_accel", "node": node, "dof": dof}
        for node in ACCELEROMETER_NODE_TAGS
        for dof in ACCELEROMETER_DOFS
    ]
    return json.dumps({
        "commands": commands,
        "analysis": {
            "type": "Transient",
            "steps": len(time) - 1,
            "dt": GROUND_MOTION_DT,
        },
        "parameter_tags": list(range(1, parameter_index + 1)),
        "responses": responses,
    })


def observe(response_history):
    """Return raw three-axis absolute acceleration histories."""
    response = jnp.asarray(response_history).reshape(
        response_history.shape[0],
        len(ACCELEROMETER_NODE_TAGS),
        len(ACCELEROMETER_DOFS),
    )
    _, acceleration_x, acceleration_y = make_base_excitation()
    ground = jnp.stack((
        jnp.asarray(acceleration_x[1:]) * 9.80665,
        jnp.asarray(acceleration_y[1:]) * 9.80665,
        jnp.zeros(response.shape[0]),
    ), axis=1)
    absolute = response + ground[:, None, :]
    free_decay_start = int(np.ceil(EXCITATION_DURATION / GROUND_MOTION_DT)) - 1
    free_decay = absolute[free_decay_start:]
    return free_decay.reshape(free_decay.shape[0], -1)[::OBSERVATION_STRIDE]
