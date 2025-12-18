"""
Visualization utilities for HP protein folding results.

Provides functions for 3D structure plots.
"""

import matplotlib.pyplot as plt

from hp_model import Conformation


def plot_conformation_3d(
    conf: Conformation,
    title: str = "Protein Conformation",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot a 3D protein conformation.

    Args:
        conf: Conformation to plot
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    coords = conf.coords
    sequence = conf.sequence

    # Plot monomers
    h_x, h_y, h_z = [], [], []
    p_x, p_y, p_z = [], [], []

    for i, (x, y, z) in enumerate(coords):
        if sequence[i] == "H":
            h_x.append(x)
            h_y.append(y)
            h_z.append(z)
        else:
            p_x.append(x)
            p_y.append(y)
            p_z.append(z)

    # Plot H monomers in red
    if h_x:
        ax.scatter(h_x, h_y, h_z, c="red", s=100, label="H (hydrophobic)", alpha=0.7)

    # Plot P monomers in blue
    if p_x:
        ax.scatter(p_x, p_y, p_z, c="blue", s=100, label="P (polar)", alpha=0.7)

    # Plot bonds (chain connections)
    for i in range(len(coords) - 1):
        x1, y1, z1 = coords[i]
        x2, y2, z2 = coords[i + 1]
        ax.plot([x1, x2], [y1, y2], [z1, z2], "k-", alpha=0.3, linewidth=1)

    # Mark first monomer
    if coords:
        ax.scatter(
            [coords[0][0]],
            [coords[0][1]],
            [coords[0][2]],
            c="green",
            s=200,
            marker="*",
            label="Start",
            zorder=5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    # Set equal aspect ratio
    max_range = max(
        max([abs(c[0]) for c in coords]) if coords else 0,
        max([abs(c[1]) for c in coords]) if coords else 0,
        max([abs(c[2]) for c in coords]) if coords else 0,
    )
    if max_range > 0:
        ax.set_xlim([-max_range - 1, max_range + 1])
        ax.set_ylim([-max_range - 1, max_range + 1])
        ax.set_zlim([-max_range - 1, max_range + 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
