"""Shared gradient-check metrics and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.plot_style import BLUE, ERROR_CMAP, RED, set_plot_style


def compute_gradient_metrics(actual, reference) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    difference = actual - reference
    scale = max(float(np.max(np.abs(reference))), 1.0)
    active_threshold = 1.0e-10 * scale
    active = np.maximum(np.abs(actual), np.abs(reference)) > active_threshold
    denominator = np.maximum(np.maximum(np.abs(actual), np.abs(reference)), active_threshold)
    relative = np.abs(difference) / denominator

    return {
        "max_absolute": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1.0e-30)),
        "max_active_relative": float(np.max(relative[active])) if np.any(active) else 0.0,
        "active_components": int(np.count_nonzero(active)),
        "sign_mismatches": int(np.count_nonzero(np.signbit(actual[active]) != np.signbit(reference[active]))),
    }


def plot_gradient_check(
    actual,
    reference,
    *,
    parameter_shape: tuple[int, int],
    parameter_labels: tuple[str, ...],
    times: tuple[float, float],
    output,
    actual_label: str = "Tesseract/DDM gradient",
    method_label: str = "JAX → Tesseract\n→ OpenSees DDM",
    gradient_unit: str | None = None,
):
    """Plot parity, componentwise error, and timing comparison."""
    set_plot_style()
    actual = np.asarray(actual, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    difference = np.abs(actual - reference)
    scale = max(float(np.max(np.abs(reference))), 1.0)
    floor = 1.0e-14 * scale
    active = np.maximum(np.abs(actual), np.abs(reference)) > floor
    relative = np.zeros_like(actual)
    relative[active] = difference[active] / np.maximum(
        np.maximum(np.abs(actual[active]), np.abs(reference[active])), floor
    )

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))

    # Use one common scale factor rather than repeating scientific notation on
    # both axes.
    parity = axes[0]
    x = np.abs(reference[active])
    y = np.abs(actual[active])
    exponent = int(np.floor(np.log10(max(float(x.max()), float(y.max())))))
    common_scale = 10.0**exponent
    x_scaled = x / common_scale
    y_scaled = y / common_scale
    lower = min(x_scaled.min(), y_scaled.min()) * 0.96
    upper = max(x_scaled.max(), y_scaled.max()) * 1.04
    parity.plot(x_scaled, y_scaled, "o", ms=4, alpha=0.78, color=BLUE)
    parity.plot([lower, upper], [lower, upper], "--", color=RED, lw=1)
    parity.set_xlim(lower, upper)
    parity.set_ylim(lower, upper)
    parity.set_aspect("equal", adjustable="box")
    parity.set_xlabel(r"$|\,\mathrm{finite\!\!-\!difference\ gradient}\,|$", fontsize=24)
    actual_label_math = actual_label.replace(" ", r"\ ")
    parity.set_ylabel(
        rf"$|\,\mathrm{{{actual_label_math}}}\,|$",
        fontsize=24,
    )
    parity.set_title("Gradient parity", fontsize=26)
    scale_text = rf"$\times 10^{{{exponent}}}$"
    if gradient_unit:
        scale_text += rf" $\mathrm{{{gradient_unit}}}$"
    parity.text(
        0.04,
        0.94,
        scale_text,
        transform=parity.transAxes,
        ha="left",
        va="top",
        fontsize=20,
    )
    parity.tick_params(axis="both", which="major", labelsize=21)
    parity.grid(True, which="both", alpha=0.2)

    heatmap = axes[1]
    error_matrix = relative.reshape(parameter_shape)
    image = heatmap.imshow(
        np.log10(np.maximum(error_matrix, 1.0e-16)),
        aspect="auto",
        cmap=ERROR_CMAP,
        vmin=-16,
        vmax=-4,
    )
    heatmap.set_xlabel("Parameter type", fontsize=24)
    heatmap.set_ylabel("Physical member", fontsize=24)
    heatmap.set_xticks(np.arange(len(parameter_labels)), parameter_labels, fontsize=21)
    tick_count = min(6, parameter_shape[0])
    member_ticks = np.unique(
        np.rint(np.linspace(0, parameter_shape[0] - 1, tick_count)).astype(int)
    )
    heatmap.set_yticks(member_ticks, member_ticks + 1, fontsize=21)
    heatmap.set_title(r"$\log_{10}$ relative error", fontsize=26)
    colorbar = figure.colorbar(image, ax=heatmap, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=21)

    timing = axes[2]
    seconds = np.asarray(times, dtype=float)
    if float(seconds.max()) >= 1.0:
        timing_values = seconds
        timing_unit = "s"
    else:
        timing_values = 1.0e3 * seconds
        timing_unit = "ms"
    bars = timing.bar(
        [method_label, "Finite\ndifference"],
        timing_values,
        color=[BLUE, RED],
    )
    timing.set_ylabel(f"Median gradient time [{timing_unit}]", fontsize=24)
    timing.set_title("Gradient evaluation time", fontsize=26)
    timing.tick_params(axis="both", labelsize=21)
    timing.set_ylim(0.0, 1.18 * float(timing_values.max()))
    timing.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, timing_values):
        timing.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=21,
        )

    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
