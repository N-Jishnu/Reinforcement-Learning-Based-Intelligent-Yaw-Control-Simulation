"""Architecture reporting and visualization for the Q-network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from rl.agents import QNetwork


def print_architecture(model: QNetwork) -> None:
    """Print layer shapes, activations and parameter counts."""

    print(model.describe())


def plot_architecture(model: QNetwork, output_path: str | Path) -> Path:
    """Draw the actual network topology using Matplotlib.

    Up to 24 representative neurons are drawn per layer so the figure remains
    readable; the true neuron count is printed above each layer.  Sampled
    connections are drawn to indicate full connectivity.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    layer_sizes: List[int] = [model.input_size, *model.hidden_sizes, model.output_size]
    layer_names = ["Input", *[f"Hidden {i + 1}" for i in range(len(model.hidden_sizes))], "Output"]
    activations = ["-", *["ReLU"] * len(model.hidden_sizes), "Linear"]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.7, len(layer_sizes) - 0.3 + 0.7)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")

    x_positions = list(range(len(layer_sizes)))
    shown_positions = []

    for layer_index, (size, name, activation) in enumerate(zip(layer_sizes, layer_names, activations)):
        x = x_positions[layer_index]
        shown = min(size, 24)
        ys = [0.0] if shown == 1 else [0.85 - 1.7 * i / (shown - 1) for i in range(shown)]
        shown_positions.append((x, ys, size))

        # Background box indicates the true layer extent.
        box = FancyBboxPatch(
            (x - 0.16, -0.95),
            0.32,
            1.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0,
            edgecolor="#4C72B0",
            facecolor="#EAF0F8",
            alpha=0.55,
            zorder=1,
        )
        ax.add_patch(box)

        for y in ys:
            ax.add_patch(Circle((x, y), 0.035, facecolor="#4C72B0", edgecolor="white", linewidth=0.5, zorder=3))

        label = f"{name}\n{size} neurons"
        if size > shown:
            label += f"\n({shown} drawn)"
        if activation != "-":
            label += f"\n{activation}"
        ax.text(x, 1.12, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Sampled all-to-all connections.
    rng = __import__("random").Random(7)
    for layer_index in range(len(shown_positions) - 1):
        x0, ys0, _ = shown_positions[layer_index]
        x1, ys1, _ = shown_positions[layer_index + 1]
        for y0 in ys0:
            targets = rng.sample(ys1, min(3, len(ys1)))
            for y1 in targets:
                ax.plot([x0 + 0.04, x1 - 0.04], [y0, y1], color="#9AA7B8", linewidth=0.35, alpha=0.45, zorder=2)

    summary = model.parameter_summary()
    ax.text(
        0.5,
        -0.93,
        "Fully connected feed-forward network | "
        f"{summary['total_trainable_parameters']:,} trainable parameters | "
        f"Input {model.input_size} -> Hidden {list(model.hidden_sizes)} -> Output {model.output_size}",
        ha="center",
        va="bottom",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_title("DQN Q-Network Architecture (actual PyTorch implementation)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved architecture diagram: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Print and visualize the Q-network architecture")
    parser.add_argument("--output", default="results/plots/network_architecture.png")
    args = parser.parse_args()
    model = QNetwork()
    print_architecture(model)
    plot_architecture(model, args.output)


if __name__ == "__main__":
    main()
