"""Shared Matplotlib style and color palette."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_BASE_CMAP = plt.get_cmap("RdBu_r")
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "soft_red_blue",
    _BASE_CMAP(np.linspace(0.18, 0.82, 256)),
)
BLUE = ERROR_CMAP(0.0)
RED = ERROR_CMAP(1.0)


def set_plot_style(font_size: int = 16) -> None:
    """Apply the project's LaTeX-rendered Matplotlib style."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    })
