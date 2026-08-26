"""Visualise the three-dimensional pedestrian bridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.shm_bridge.experiment import (
    corrosion_from_severity,
    make_section_spec,
    make_transient_program,
    section_properties_to_parameters,
    true_severity,
)
from examples.shm_bridge.model import (
    BridgeMesh,
    make_bridge_mesh,
)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 17,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

STEEL_BLUE = "#315D8A"
BRACING_BLUE = "#6E9FC5"
DECK_GRAY = "#D9D9D6"
DECK_EDGE = "#A6A6A2"
DAMAGE_RED = "#D45A4A"
UNDEFORMED_GRAY = "#A8A8A8"


def _line_color(group: str) -> str:
    if group in {"side_diagonal", "top_bracing", "bottom_bracing"}:
        return BRACING_BLUE
    if group in {"stringer", "floor_beam", "top_cross_beam"}:
        return "#6686A3"
    return STEEL_BLUE


def _line_width(group: str) -> float:
    if group in {"bottom_chord", "top_chord", "end_post"}:
        return 2.2
    if group in {"stringer", "floor_beam", "top_cross_beam"}:
        return 1.25
    return 1.0



def _project(points):
    """Project bridge coordinates to an oblique engineering view."""
    points = np.asarray(points, dtype=float)
    return np.stack(
        (
            points[..., 0] - 0.62 * points[..., 1],
            1.35 * points[..., 2] + 0.36 * points[..., 1],
        ),
        axis=-1,
    )


def _projected_support_points(mesh: BridgeMesh):
    """Project bearings below their nodes along the physical vertical axis."""
    support_coordinates = np.asarray([mesh.nodes[tag] for tag in mesh.supports])
    return _project(support_coordinates - np.asarray((0.0, 0.0, 0.18)))



def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transient_displacement_history(mesh: BridgeMesh):
    """Run the damaged bridge through the prescribed transient record."""
    section_solver = _load_module(
        "bridge_section_solver",
        ROOT / "tesseracts" / "section_properties" / "section_solver.py",
    )
    opensees_solver = _load_module(
        "bridge_opensees_solver",
        ROOT / "tesseracts" / "opensees_ddm" / "solver.py",
    )

    section_spec = make_section_spec()
    pristine = section_solver.apply(section_spec, np.zeros((1, 2)))[0]
    damaged = section_solver.apply(
        section_spec,
        np.asarray(corrosion_from_severity(true_severity())),
    )
    parameters = np.asarray(
        section_properties_to_parameters(mesh, damaged, pristine),
        dtype=float,
    )
    program = opensees_solver.parse_program(make_transient_program(mesh))
    opensees_solver.build_domain(program, parameters, sensitivity=False)

    node_tags = tuple(mesh.nodes)
    history = np.empty((program["analysis"]["steps"], len(node_tags), 3))
    for step in range(history.shape[0]):
        opensees_solver.analyze_step(program["analysis"])
        history[step] = [ops.nodeDisp(tag)[:3] for tag in node_tags]

    peak_frame = int(np.argmax(np.max(np.linalg.norm(history, axis=2), axis=1)))
    time = (np.arange(history.shape[0]) + 1) * float(program["analysis"]["dt"])
    peak_displacement = float(np.max(np.linalg.norm(history[peak_frame], axis=1)))
    return node_tags, history, time, peak_frame, peak_displacement


def animate_transient_response(
    mesh: BridgeMesh,
    node_tags,
    displacement_history,
    time,
    output: Path,
    *,
    scale: float = 500.0,
    interval: int = 80,
):
    """Animate the complete bridge deformation history under transient input."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    node_tags = tuple(node_tags)
    node_index = {tag: index for index, tag in enumerate(node_tags)}
    coordinates = np.asarray([mesh.nodes[tag] for tag in node_tags], dtype=float)
    history = np.asarray(displacement_history, dtype=float)
    deformed = coordinates[None, :, :] + scale * history

    projected_undeformed = _project(coordinates)
    projected_history = _project(deformed)

    figure, axes = plt.subplots(figsize=(15.5, 5.2))
    support_points = _projected_support_points(mesh)
    axes.scatter(
        support_points[:, 0], support_points[:, 1],
        marker="^", s=92, color="#444444", edgecolors="#444444", zorder=0,
    )

    reference_segments = [
        projected_undeformed[[node_index[node] for node in element.nodes]]
        for element in mesh.lines
    ]
    axes.add_collection(LineCollection(
        reference_segments,
        colors=UNDEFORMED_GRAY,
        linewidths=0.9,
        linestyles="dashed",
        alpha=0.48,
        zorder=1,
    ))

    shell_indices = [
        [node_index[node] for node in shell.nodes] for shell in mesh.shells
    ]
    deck = PolyCollection(
        [projected_history[0, indices] for indices in shell_indices],
        facecolors=DECK_GRAY,
        edgecolors=DECK_EDGE,
        linewidths=0.35,
        alpha=0.48,
        zorder=2,
    )
    axes.add_collection(deck)

    candidates = set(mesh.candidate_damage_elements)
    grouped_elements: dict[tuple[str, float], list] = {}
    damaged_elements = []
    for element in mesh.lines:
        if element.tag in candidates:
            damaged_elements.append(element)
        else:
            grouped_elements.setdefault(
                (_line_color(element.group), _line_width(element.group)), []
            ).append(element)

    line_collections = []
    for (color, width), elements in grouped_elements.items():
        collection = LineCollection(
            [
                projected_history[0, [node_index[node] for node in element.nodes]]
                for element in elements
            ],
            colors=color,
            linewidths=width,
            zorder=3,
        )
        axes.add_collection(collection)
        line_collections.append((collection, elements))

    damaged_collection = LineCollection(
        [
            projected_history[0, [node_index[node] for node in element.nodes]]
            for element in damaged_elements
        ],
        colors=DAMAGE_RED,
        linewidths=3.2,
        zorder=4,
    )
    axes.add_collection(damaged_collection)

    all_points = np.concatenate((
        projected_undeformed[None, :, :],
        projected_history,
    ), axis=0)
    lower = all_points.min(axis=(0, 1))
    upper = all_points.max(axis=(0, 1))
    padding_x = 0.025 * np.ptp(all_points[..., 0])
    padding_y = 0.12 * np.ptp(all_points[..., 1])
    axes.set_xlim(lower[0] - padding_x, upper[0] + padding_x)
    axes.set_ylim(lower[1] - padding_y, upper[1] + padding_y)
    axes.set_aspect("equal", adjustable="box")
    axes.axis("off")
    axes.set_title(
        rf"Bridge response under designed transient excitation ($\times {scale:.0f}$)",
        pad=8,
    )
    annotation = figure.text(
        0.5, 0.205, "",
        ha="center", va="center", fontsize=21,
    )

    figure.legend(
        handles=[
            Line2D([0], [0], color=STEEL_BLUE, lw=2.2, label="Primary steel members"),
            Line2D([0], [0], color=BRACING_BLUE, lw=1.2, label="Bracing"),
            Line2D([0], [0], color=DECK_GRAY, lw=6, alpha=0.7, label="Concrete deck"),
            Line2D([0], [0], color=DAMAGE_RED, lw=3.0, label="Candidate damage segments"),
            Line2D(
                [0], [0], color="#444444", marker="^", linestyle="none",
                markersize=11, label="Fixed supports",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=5,
        frameon=True,
    )
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.245, top=0.91)

    def update(frame):
        current = projected_history[frame]
        deck.set_verts([current[indices] for indices in shell_indices])
        for collection, elements in line_collections:
            collection.set_segments([
                current[[node_index[node] for node in element.nodes]]
                for element in elements
            ])
        damaged_collection.set_segments([
            current[[node_index[node] for node in element.nodes]]
            for element in damaged_elements
        ])
        annotation.set_text(rf"$t = {time[frame]:.2f}\,\mathrm{{s}}$")
        return (deck, damaged_collection, annotation) + tuple(
            collection for collection, _ in line_collections
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=history.shape[0],
        interval=interval,
        blit=False,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=1000.0 / interval))
    plt.close(figure)
    return output

def main():
    mesh = make_bridge_mesh()
    animation_scale = 500.0
    node_tags, history, time, peak_frame, peak_displacement = (
        transient_displacement_history(mesh)
    )
    peak_time = time[peak_frame]
    animation = animate_transient_response(
        mesh,
        node_tags,
        history,
        time,
        ROOT / "figs" / "shm_bridge_transient.gif",
        scale=animation_scale,
    )

    print("Steel pedestrian bridge transient response")
    print(f"Nodes                       : {len(mesh.nodes)}")
    print(f"Steel line elements         : {len(mesh.lines)}")
    print(f"Concrete shell elements     : {len(mesh.shells)}")
    print(f"Candidate damage elements   : {mesh.candidate_damage_elements}")
    print(f"Peak response time          : {peak_time:.3f} s")
    print(f"Peak translational response : {peak_displacement:.6e} m")
    print(f"Deformation scale factor    : {animation_scale:.1f}")
    print(f"Transient-response GIF     : {animation}")


if __name__ == "__main__":
    main()
