"""Static SHM frame with four spatially separated local corrosion sites."""

from __future__ import annotations

import json
from itertools import pairwise

import jax.numpy as jnp
import numpy as np

N_SUBELEMENTS = 4

NODE_COORDINATES = {
    1: (0.0, 0.0), 2: (5.0, 0.0), 3: (10.0, 0.0), 4: (15.0, 0.0),
    5: (0.0, 4.0), 6: (5.0, 4.0), 7: (10.0, 4.0), 8: (15.0, 4.0),
    9: (0.0, 8.0), 10: (5.0, 8.0), 11: (10.0, 8.0), 12: (15.0, 8.0),
    13: (0.0, 12.0), 14: (5.0, 12.0), 15: (10.0, 12.0), 16: (15.0, 12.0),
}

MEMBERS = (
    (1, 5), (5, 9), (9, 13),
    (2, 6), (6, 10), (10, 14),
    (3, 7), (7, 11), (11, 15),
    (4, 8), (8, 12), (12, 16),
    (5, 6), (9, 10), (13, 14),
    (6, 7), (10, 11), (14, 15),
    (7, 8), (11, 12), (15, 16),
)

# member, damaged quarter-segment, label
DAMAGE_SITES = (
    (0, 0, "first-story left-column base"),
    (15, 1, "first-story middle-beam interior"),
    (13, 1, "second-story left-beam interior"),
    (11, 0, "third-story right-column base"),
)
DAMAGE_LABELS = tuple(site[2] for site in DAMAGE_SITES)



def make_section_spec() -> str:
    """Nominal I-section and its cross-sectional corrosion morphology."""
    return json.dumps({
        "d": 0.30,
        "b": 0.25,
        "tf": 0.020,
        "tw": 0.020,
        "root_radius": 0.006,
        "root_points": 8,
        "mesh_size": 1.0e-4,
    })


def initial_corrosion() -> np.ndarray:
    """VI initialization: independent flange/web losses at four sites [m]."""
    return np.full((4, 2), 0.002)


def true_corrosion() -> np.ndarray:
    """Synthetic hidden flange/web losses at four sites [m]."""
    return np.asarray([
        [0.0080, 0.0035],
        [0.0050, 0.0060],
        [0.0090, 0.0025],
        [0.0040, 0.0070],
    ])


def corrosion_upper_bounds() -> np.ndarray:
    """Physical upper bounds for flange and web loss [m]."""
    return np.tile(np.asarray([0.012, 0.008]), (len(DAMAGE_SITES), 1))


def _member_nodes(member_index: int) -> tuple[int, ...]:
    node_i, node_j = MEMBERS[member_index]
    first_internal = 17 + member_index * (N_SUBELEMENTS - 1)
    return (node_i, first_internal, first_internal + 1, first_internal + 2, node_j)


# Two targeted tests per damage site: bending dominated, then axial dominated.
LOAD_CASES = (
    ((5, 1, 240.0e3),),
    ((5, 2, -800.0e3),),
    ((63, 2, -250.0e3),),
    ((6, 1, -300.0e3), (7, 1, 300.0e3)),
    ((57, 2, -250.0e3),),
    ((9, 1, -300.0e3), (10, 1, 300.0e3)),
    ((16, 1, 320.0e3),),
    ((12, 2, 800.0e3), (16, 2, -800.0e3)),
)


SENSOR_NODES = tuple(dict.fromkeys(
    node
    for member, _, _ in DAMAGE_SITES
    for node in _member_nodes(member)[1:-1]
))
SENSOR_DOFS = (1, 2, 3)
RAW_RESPONSES = tuple(
    (node, dof)
    for node in SENSOR_NODES
    for dof in SENSOR_DOFS
)
OBSERVATION_LABELS = tuple(
    f"case {case + 1}, node {node}, dof {dof}"
    for case in range(len(LOAD_CASES))
    for node, dof in RAW_RESPONSES
)


def observe(response_history):
    """Return direct nodal translations and rotations without post-processing."""
    return jnp.asarray(response_history).reshape(-1)


def observation_noise_scale(clean_observation, noise_fraction: float):
    """Relative sensor noise with a small absolute noise floor."""
    magnitude = jnp.abs(jnp.asarray(clean_observation))
    floor = 0.1 * jnp.median(magnitude)
    return noise_fraction * jnp.maximum(magnitude, floor)


def _parameter(index: int):
    return {"parameter": index}


def make_opensees_program(nominal_area: float, nominal_inertia: float) -> str:
    commands = [{"name": "model", "args": ["basic", "-ndm", 2, "-ndf", 3]}]
    for tag, (x, y) in NODE_COORDINATES.items():
        commands.append({"name": "node", "args": [tag, x, y]})
    for tag in (1, 2, 3, 4):
        commands.append({"name": "fix", "args": [tag, 1, 1, 1]})
    commands.append({"name": "geomTransf", "args": ["Linear", 1]})

    damage_by_member = {
        member: (site_index, segment)
        for site_index, (member, segment, _) in enumerate(DAMAGE_SITES)
    }
    element_tag = 1
    for member_index, (node_i, node_j) in enumerate(MEMBERS):
        xi, yi = NODE_COORDINATES[node_i]
        xj, yj = NODE_COORDINATES[node_j]
        nodes = _member_nodes(member_index)
        for local_node, fraction in zip(nodes[1:-1], (0.25, 0.50, 0.75)):
            commands.append({
                "name": "node",
                "args": [
                    local_node,
                    xi + fraction * (xj - xi),
                    yi + fraction * (yj - yi),
                ],
            })

        for segment, (start, end) in enumerate(pairwise(nodes)):
            section_tag = element_tag
            integration_tag = element_tag
            damage = damage_by_member.get(member_index)
            active = damage is not None and segment == damage[1]
            if active:
                site_index = damage[0]
                area = _parameter(2 * site_index)
                inertia = _parameter(2 * site_index + 1)
            else:
                area = nominal_area
                inertia = nominal_inertia

            commands.extend([
                {
                    "name": "section",
                    "args": ["Elastic", section_tag, 200.0e9, area, inertia],
                },
                {
                    "name": "beamIntegration",
                    "args": ["Lobatto", integration_tag, section_tag, 4],
                },
                {
                    "name": "element",
                    "args": ["dispBeamColumn", element_tag, start, end, 1, integration_tag],
                },
            ])
            if active:
                commands.extend([
                    {
                        "name": "parameter",
                        "args": [2 * site_index + 1, "element", element_tag, "A"],
                    },
                    {
                        "name": "parameter",
                        "args": [2 * site_index + 2, "element", element_tag, "I"],
                    },
                ])
            element_tag += 1

    n_cases = len(LOAD_CASES)
    for case_index, loads in enumerate(LOAD_CASES, start=1):
        values = [0.0] * (n_cases + 1)
        values[case_index] = 1.0
        commands.extend([
            {
                "name": "timeSeries",
                "args": ["Path", case_index, "-dt", 1.0, "-values", *values, "-useLast"],
            },
            {"name": "pattern", "args": ["Plain", case_index, case_index]},
        ])
        for node, dof, magnitude in loads:
            load = [0.0, 0.0, 0.0]
            load[dof - 1] = magnitude
            commands.append({"name": "load", "args": [node, *load]})

    commands.extend([
        {"name": "constraints", "args": ["Plain"]},
        {"name": "numberer", "args": ["RCM"]},
        {"name": "system", "args": ["UmfPack"]},
        {"name": "algorithm", "args": ["Linear"]},
        {"name": "integrator", "args": ["LoadControl", 1.0]},
    ])
    return json.dumps({
        "commands": commands,
        "analysis": {"type": "Static", "steps": n_cases},
        "parameter_tags": list(range(1, 2 * len(DAMAGE_SITES) + 1)),
        "responses": [
            {"type": "node_disp", "node": node, "dof": dof}
            for node, dof in RAW_RESPONSES
        ],
    })
