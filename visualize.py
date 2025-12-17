"""
Visualization utilities for HP protein folding results.

Provides functions for 3D structure plots and convergence curves.
"""

import matplotlib.pyplot as plt
import numpy as np

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


def plot_convergence_curve(
    fitness_history: list[float],
    title: str = "Convergence Curve",
    xlabel: str = "Generation",
    ylabel: str = "Best Fitness (H-H Contacts)",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot convergence curve (fitness over time).

    Args:
        fitness_history: List of best fitness values over time
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_comparison(
    ga_history: list[float],
    mcts_history: list[float],
    title: str = "GA vs MCTS Convergence",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot convergence curves for both algorithms.

    Args:
        ga_history: GA fitness history
        mcts_history: MCTS reward history
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))

    # Normalize lengths - pad shorter history
    max_len = max(len(ga_history), len(mcts_history))
    ga_padded = (
        ga_history + [ga_history[-1]] * (max_len - len(ga_history))
        if ga_history
        else []
    )
    mcts_padded = (
        mcts_history + [mcts_history[-1]] * (max_len - len(mcts_history))
        if mcts_history
        else []
    )

    plt.plot(ga_padded, label="GA", linewidth=2, alpha=0.8)
    plt.plot(mcts_padded, label="MCTS", linewidth=2, alpha=0.8)
    plt.xlabel("Evaluations")
    plt.ylabel("Best H-H Contacts")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_bar_chart(
    comparison_results: dict, save_path: str | None = None, show: bool = True
):
    """
    Plot bar chart comparing GA and MCTS results.

    Args:
        comparison_results: Results from compare_algorithms()
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    ga = comparison_results["ga"]
    mcts = comparison_results["mcts"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Best contacts comparison
    algorithms = ["GA", "MCTS"]
    mean_contacts = [ga["mean_best_contacts"], mcts["mean_best_contacts"]]
    std_contacts = [ga["std_best_contacts"], mcts["std_best_contacts"]]

    ax1.bar(
        algorithms,
        mean_contacts,
        yerr=std_contacts,
        capsize=10,
        alpha=0.7,
        color=["red", "blue"],
    )
    ax1.set_ylabel("Mean Best H-H Contacts")
    ax1.set_title("Solution Quality Comparison")
    ax1.grid(True, alpha=0.3, axis="y")

    if ga["optimal_contacts"] is not None:
        ax1.axhline(
            y=ga["optimal_contacts"],
            color="green",
            linestyle="--",
            label=f"Optimal ({ga['optimal_contacts']})",
        )
        ax1.legend()

    # Success rate comparison (if available)
    if ga["success_rate"] is not None and mcts["success_rate"] is not None:
        success_rates = [ga["success_rate"] * 100, mcts["success_rate"] * 100]
        ax2.bar(algorithms, success_rates, alpha=0.7, color=["red", "blue"])
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Success Rate Comparison")
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        # Evaluations comparison instead
        mean_evals = [ga["mean_evaluations_used"], mcts["mean_evaluations_used"]]
        ax2.bar(algorithms, mean_evals, alpha=0.7, color=["red", "blue"])
        ax2.set_ylabel("Mean Evaluations Used")
        ax2.set_title("Computational Cost Comparison")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Comparison for {comparison_results['sequence_name']}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_sequences(
    results_list: list[dict],
    metric: str = "mean_best_contacts",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot comparison across multiple sequences.

    Args:
        results_list: List of comparison results dictionaries
        metric: Metric to plot ('mean_best_contacts' or 'success_rate')
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    sequence_names = [r["sequence_name"] for r in results_list]
    ga_values = [r["ga"][metric] for r in results_list]
    mcts_values = [r["mcts"][metric] for r in results_list]

    x = np.arange(len(sequence_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, ga_values, width, label="GA", alpha=0.7, color="red")
    ax.bar(x + width / 2, mcts_values, width, label="MCTS", alpha=0.7, color="blue")

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Algorithm Comparison Across Sequences ({metric})")
    ax.set_xticks(x)
    ax.set_xticklabels(sequence_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
