"""Finite-element section analysis for locally corroded I-sections."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sectionproperties.analysis import Section
from sectionproperties.pre.library.primitive_sections import rectangular_section
from sectionproperties.pre.library.steel_sections import i_section

PROPERTY_LABELS = ("A", "Ixx", "Iyy", "Ixy", "cx", "cy")
CORROSION_LABELS = ("flange_loss", "web_loss")


def parse_spec(section_spec: str | dict[str, Any]) -> dict[str, Any]:
    spec = json.loads(section_spec) if isinstance(section_spec, str) else section_spec
    required = {"d", "b", "tf", "tw", "root_radius", "mesh_size"}
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"section specification is missing: {sorted(missing)}")
    return spec


def build_geometry(spec: dict[str, Any], corrosion: np.ndarray):
    d = float(spec["d"])
    b = float(spec["b"])
    tf = float(spec["tf"])
    tw = float(spec["tw"])
    flange_loss, web_loss = map(float, corrosion)

    if not (
        0.0 <= flange_loss < tf
        and 0.0 <= web_loss < 0.5 * tw
    ):
        raise ValueError("corrosion losses exceed the affected plate thickness")

    geometry = i_section(
        d=d,
        b=b,
        t_f=tf,
        t_w=tw,
        r=float(spec["root_radius"]),
        n_r=int(spec.get("root_points", 8)),
    )

    # Local, non-uniform, double-symmetric corrosion. Flange and web loss
    # affect A and Ixx differently, making the two variables identifiable in
    # the present 2D reduction.
    if flange_loss > 0.0:
        notch_width = 0.22 * b
        for x_offset in (0.0, b - notch_width):
            geometry = geometry - rectangular_section(
                d=flange_loss, b=notch_width
            ).shift_section(x_offset=x_offset, y_offset=d - flange_loss)
            geometry = geometry - rectangular_section(
                d=flange_loss, b=notch_width
            ).shift_section(x_offset=x_offset, y_offset=0.0)

    if web_loss > 0.0:
        notch_height = 0.28 * d
        y_offset = 0.5 * (d - notch_height)
        geometry = geometry - rectangular_section(
            d=notch_height, b=web_loss
        ).shift_section(x_offset=0.5 * b - 0.5 * tw, y_offset=y_offset)
        geometry = geometry - rectangular_section(
            d=notch_height, b=web_loss
        ).shift_section(
            x_offset=0.5 * b + 0.5 * tw - web_loss,
            y_offset=y_offset,
        )

    return geometry


def analyse_one(section_spec: str | dict[str, Any], corrosion) -> np.ndarray:
    spec = parse_spec(section_spec)
    geometry = build_geometry(spec, np.asarray(corrosion, dtype=float))
    geometry.create_mesh(mesh_sizes=float(spec["mesh_size"]))
    section = Section(geometry)
    section.calculate_geometric_properties()

    ixx, iyy, ixy = section.get_ic()
    cx, cy = section.get_c()
    return np.asarray([section.get_area(), ixx, iyy, ixy, cx, cy], dtype=float)


def apply(section_spec: str | dict[str, Any], corrosion) -> np.ndarray:
    corrosion = np.asarray(corrosion, dtype=float)
    if corrosion.ndim != 2 or corrosion.shape[1] != len(CORROSION_LABELS):
        raise ValueError("corrosion must have shape (n_sections, 2)")
    return np.stack([analyse_one(section_spec, row) for row in corrosion])


def vector_jacobian_product(section_spec, corrosion, property_cotangent) -> np.ndarray:
    """Finite-difference pullback for all local corrosion parameters."""
    corrosion = np.asarray(corrosion, dtype=float)
    cotangent = np.asarray(property_cotangent, dtype=float)
    if cotangent.shape != (corrosion.shape[0], len(PROPERTY_LABELS)):
        raise ValueError("property cotangent has an incompatible shape")

    gradient = np.empty_like(corrosion)
    for section_index, row in enumerate(corrosion):
        for parameter_index, value in enumerate(row):
            step = max(1.0e-6, 1.0e-4 * abs(value))
            plus = row.copy()
            minus = row.copy()
            plus[parameter_index] += step
            minus[parameter_index] -= step
            if minus[parameter_index] < 0.0:
                base = analyse_one(section_spec, row)
                perturbed = analyse_one(section_spec, plus)
                derivative = (perturbed - base) / step
            else:
                derivative = (
                    analyse_one(section_spec, plus)
                    - analyse_one(section_spec, minus)
                ) / (2.0 * step)
            gradient[section_index, parameter_index] = (
                cotangent[section_index] @ derivative
            )
    return gradient
