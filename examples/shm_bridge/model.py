"""Three-dimensional OpenSees model of a steel pedestrian bridge."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineElement:
    tag: int
    nodes: tuple[int, int]
    group: str
    formulation: str


@dataclass(frozen=True)
class ShellElement:
    tag: int
    nodes: tuple[int, int, int, int]
    group: str = "concrete_deck"


@dataclass(frozen=True)
class BridgeMesh:
    nodes: Mapping[int, tuple[float, float, float]]
    lines: tuple[LineElement, ...]
    shells: tuple[ShellElement, ...]
    supports: Mapping[int, tuple[int, int, int, int, int, int]]
    candidate_damage_elements: tuple[int, ...]


STEEL_E = 200.0e9
STEEL_NU = 0.30
STEEL_G = STEEL_E / (2.0 * (1.0 + STEEL_NU))
CONCRETE_E = 30.0e9
CONCRETE_NU = 0.20
CONCRETE_DENSITY = 2500.0
DECK_THICKNESS = 0.15

# Representative steel section properties.  Each entry is (A, Iy, Iz, J) in SI.
# These group-level properties are the interface that will later be replaced by
# sectionproperties outputs for selected damaged members.
SECTION_PROPERTIES = {
    "bottom_chord": (1.80e-2, 2.00e-4, 4.50e-4, 1.20e-5),
    "top_chord": (1.55e-2, 1.70e-4, 3.80e-4, 1.00e-5),
    "end_post": (1.35e-2, 1.30e-4, 2.80e-4, 8.00e-6),
    "vertical": (8.00e-3, 4.50e-5, 8.50e-5, 3.50e-6),
    "stringer": (1.00e-2, 8.00e-5, 2.20e-4, 6.00e-6),
    "floor_beam": (1.20e-2, 1.00e-4, 3.00e-4, 7.00e-6),
    "top_cross_beam": (8.00e-3, 5.00e-5, 1.20e-4, 4.00e-6),
}
TRUSS_AREAS = {
    "side_diagonal": 6.00e-3,
    "top_bracing": 4.00e-3,
    "bottom_bracing": 4.00e-3,
}


def make_bridge_mesh(
    *,
    n_panels: int = 12,
    panel_length: float = 3.0,
    width: float = 4.0,
    truss_height: float = 3.2,
    n_deck_lines: int = 5,
) -> BridgeMesh:
    """Create the three-dimensional pedestrian-bridge mesh."""
    if n_panels < 4 or n_deck_lines < 3:
        raise ValueError("bridge mesh requires at least four panels and three deck lines")

    nodes: dict[int, tuple[float, float, float]] = {}

    def deck_node(station: int, line: int) -> int:
        return 1 + station * n_deck_lines + line

    y_lines = np.linspace(-0.5 * width, 0.5 * width, n_deck_lines)
    for station in range(n_panels + 1):
        x = station * panel_length
        for line, y in enumerate(y_lines):
            nodes[deck_node(station, line)] = (x, float(y), 0.0)

    def top_node(side: int, station: int) -> int:
        return 1000 + side * 100 + station

    for side, y in enumerate((-0.5 * width, 0.5 * width)):
        for station in range(1, n_panels):
            nodes[top_node(side, station)] = (
                station * panel_length,
                y,
                truss_height,
            )

    lines: list[LineElement] = []
    shells: list[ShellElement] = []
    next_line = 1
    next_shell = 10001

    def add_line(i: int, j: int, group: str, formulation: str = "beam") -> int:
        nonlocal next_line
        tag = next_line
        lines.append(LineElement(tag, (i, j), group, formulation))
        next_line += 1
        return tag

    candidate_by_name: dict[str, int] = {}

    # Longitudinal deck system: outer lines are the lower truss chords.
    for line in range(n_deck_lines):
        group = "bottom_chord" if line in (0, n_deck_lines - 1) else "stringer"
        for station in range(n_panels):
            tag = add_line(
                deck_node(station, line),
                deck_node(station + 1, line),
                group,
            )
            if (line, station) == (n_deck_lines - 1, n_panels // 2):
                candidate_by_name["middle_far_bottom"] = tag

    # Floor beams and concrete deck shell mesh.
    for station in range(n_panels + 1):
        for line in range(n_deck_lines - 1):
            add_line(
                deck_node(station, line),
                deck_node(station, line + 1),
                "floor_beam",
            )
    for station in range(n_panels):
        for line in range(n_deck_lines - 1):
            shells.append(ShellElement(
                next_shell,
                (
                    deck_node(station, line),
                    deck_node(station + 1, line),
                    deck_node(station + 1, line + 1),
                    deck_node(station, line + 1),
                ),
            ))
            next_shell += 1

    # Two side trusses.
    for side, outer_line in enumerate((0, n_deck_lines - 1)):
        # Upper chord.
        for station in range(1, n_panels - 1):
            tag = add_line(
                top_node(side, station),
                top_node(side, station + 1),
                "top_chord",
            )
            if (side, station) == (0, 2):
                candidate_by_name["left_near_top"] = tag
            if (side, station) == (1, n_panels - 3):
                candidate_by_name["right_far_top"] = tag

        # Sloping end posts.
        add_line(deck_node(0, outer_line), top_node(side, 1), "end_post")
        add_line(
            top_node(side, n_panels - 1),
            deck_node(n_panels, outer_line),
            "end_post",
        )

        # Verticals and alternating Warren diagonals.
        for station in range(1, n_panels):
            add_line(
                deck_node(station, outer_line),
                top_node(side, station),
                "vertical",
            )
        for station in range(1, n_panels - 1):
            if station % 2:
                i = top_node(side, station)
                j = deck_node(station + 1, outer_line)
            else:
                i = deck_node(station, outer_line)
                j = top_node(side, station + 1)
            tag = add_line(i, j, "side_diagonal", "truss")

    # Overhead cross beams and X bracing make the model spatially stable.
    for station in range(1, n_panels):
        add_line(
            top_node(0, station),
            top_node(1, station),
            "top_cross_beam",
        )
    for station in range(1, n_panels - 1):
        add_line(
            top_node(0, station),
            top_node(1, station + 1),
            "top_bracing",
            "truss",
        )
        add_line(
            top_node(1, station),
            top_node(0, station + 1),
            "top_bracing",
            "truss",
        )

    # Under-deck lateral X bracing between the lower chords.
    for station in range(n_panels):
        add_line(
            deck_node(station, 0),
            deck_node(station + 1, n_deck_lines - 1),
            "bottom_bracing",
            "truss",
        )
        add_line(
            deck_node(station, n_deck_lines - 1),
            deck_node(station + 1, 0),
            "bottom_bracing",
            "truss",
        )

    left_a = deck_node(0, 0)
    left_b = deck_node(0, n_deck_lines - 1)
    right_a = deck_node(n_panels, 0)
    right_b = deck_node(n_panels, n_deck_lines - 1)
    supports = {
        left_a: (1, 1, 1, 1, 1, 1),
        left_b: (1, 1, 1, 1, 1, 1),
        right_a: (1, 1, 1, 1, 1, 1),
        right_b: (1, 1, 1, 1, 1, 1),
    }

    return BridgeMesh(
        nodes=nodes,
        lines=tuple(lines),
        shells=tuple(shells),
        supports=supports,
        candidate_damage_elements=tuple(
            candidate_by_name[name]
            for name in (
                "left_near_top",
                "middle_far_bottom",
                "right_far_top",
            )
        ),
    )
