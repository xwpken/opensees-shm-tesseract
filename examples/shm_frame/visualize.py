"""Visualise the frame, damage locations, and local corroded sections."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.shm_frame.experiment import (
    DAMAGE_SITES,
    make_section_spec,
    true_corrosion,
)
from examples.shm_frame.model import MEMBERS, N_SUBELEMENTS, NODE_COORDINATES
from utils.plot_style import set_plot_style

set_plot_style(27)

STEEL_BLUE = "#315D8A"
DAMAGE_RED = "#D45A4A"
SECTION_FILL = "#D8E7F0"
SUPPORT_GRAY = "#444444"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segment_endpoints(member_index: int, segment: int):
    node_i, node_j = MEMBERS[member_index]
    start = np.asarray(NODE_COORDINATES[node_i], dtype=float)
    end = np.asarray(NODE_COORDINATES[node_j], dtype=float)
    fractions = np.linspace(0.0, 1.0, N_SUBELEMENTS + 1)
    return (
        start + fractions[segment] * (end - start),
        start + fractions[segment + 1] * (end - start),
    )


def element_node_coordinates():
    """Return every end and internal node used by the beam-column elements."""
    coordinates = []
    fractions = np.linspace(0.0, 1.0, N_SUBELEMENTS + 1)
    for node_i, node_j in MEMBERS:
        start = np.asarray(NODE_COORDINATES[node_i], dtype=float)
        end = np.asarray(NODE_COORDINATES[node_j], dtype=float)
        coordinates.extend(start + fraction * (end - start) for fraction in fractions)
    return np.unique(np.asarray(coordinates), axis=0)


def draw_section(axis, geometry, label):
    polygon = geometry.geom
    x, y = polygon.exterior.xy
    axis.fill(x, y, facecolor=SECTION_FILL, edgecolor=STEEL_BLUE, linewidth=1.15)
    for interior in polygon.interiors:
        x_hole, y_hole = interior.xy
        axis.fill(x_hole, y_hole, facecolor="white", edgecolor=STEEL_BLUE, linewidth=0.9)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.010, 0.260)
    axis.set_ylim(-0.010, 0.310)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(label, color=DAMAGE_RED, fontsize=23, pad=1)
    axis.set_facecolor((1.0, 1.0, 1.0, 0.92))
    for spine in axis.spines.values():
        spine.set_color(DAMAGE_RED)
        spine.set_linewidth(0.65)
        spine.set_alpha(0.55)


def main():
    section_solver = load_module(
        "section_properties_solver",
        ROOT / "tesseracts" / "section_properties" / "section_solver.py",
    )
    spec = section_solver.parse_spec(make_section_spec())
    geometries = [
        section_solver.build_geometry(spec, row) for row in true_corrosion()
    ]

    figure, axis = plt.subplots(figsize=(15.5, 10.0))
    figure.subplots_adjust(left=0.095, right=0.975, bottom=0.105, top=0.925)

    for node_i, node_j in MEMBERS:
        xy = np.asarray((NODE_COORDINATES[node_i], NODE_COORDINATES[node_j]))
        axis.plot(xy[:, 0], xy[:, 1], color=STEEL_BLUE, linewidth=1.9, zorder=1)

    element_nodes = element_node_coordinates()
    axis.scatter(
        element_nodes[:, 0],
        element_nodes[:, 1],
        s=24,
        facecolor="white",
        edgecolor=STEEL_BLUE,
        linewidth=1.15,
        zorder=3,
    )
    support_x = np.asarray([NODE_COORDINATES[tag][0] for tag in (1, 2, 3, 4)])
    axis.scatter(
        support_x,
        np.full_like(support_x, -0.17),
        marker="^",
        s=78,
        color=SUPPORT_GRAY,
        zorder=3,
    )

    # Bounds are in frame data coordinates. Each section sits in unused space
    # next to its damaged segment, without covering axes, labels, or the legend.
    inset_boxes = (
        (0.65, 0.55, 2.15, 2.45),
        (7.15, 0.75, 2.15, 2.45),
        (0.65, 8.65, 2.15, 2.45),
        (11.65, 8.65, 2.15, 2.45),
    )

    for site, ((member, segment, _), geometry, box) in enumerate(
        zip(DAMAGE_SITES, geometries, inset_boxes), start=1
    ):
        start, end = segment_endpoints(member, segment)
        midpoint = 0.5 * (start + end)
        axis.plot(
            (start[0], end[0]),
            (start[1], end[1]),
            color=DAMAGE_RED,
            linewidth=4.2,
            solid_capstyle="round",
            zorder=4,
        )

        inset = axis.inset_axes(box, transform=axis.transData, zorder=6)
        draw_section(inset, geometry, rf"$S_{{{site}}}$")
        connector = ConnectionPatch(
            xyA=midpoint,
            coordsA=axis.transData,
            xyB=(0.5, 0.5),
            coordsB=inset.transAxes,
            color=DAMAGE_RED,
            linewidth=0.8,
            alpha=0.70,
            zorder=5,
        )
        figure.add_artist(connector)

    legend = (
        Line2D([0], [0], color=STEEL_BLUE, lw=2.0, label="Member"),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor="white", markeredgecolor=STEEL_BLUE,
            label="Node",
        ),
        Line2D([0], [0], color=DAMAGE_RED, lw=4.0, label="Damage"),
        Line2D(
            [0], [0], marker="^", linestyle="none", markersize=8,
            color=SUPPORT_GRAY, label="Support",
        ),
    )
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        fontsize=24,
        frameon=True,
        framealpha=0.94,
        handlelength=1.5,
        handletextpad=0.55,
        columnspacing=1.15,
        borderpad=0.35,
        labelspacing=0.35,
    )
    axis.set_title("Frame model and local corrosion sites", fontsize=30, pad=10)

    axis.set_xlim(-1.75, 16.75)
    axis.set_ylim(-1.45, 13.75)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x$ [m]", fontsize=28)
    axis.set_ylabel(r"$y$ [m]", fontsize=28)
    axis.tick_params(axis="both", labelsize=25)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axis.grid(True, alpha=0.16)

    output = ROOT / "figs" / "shm_frame_model.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
