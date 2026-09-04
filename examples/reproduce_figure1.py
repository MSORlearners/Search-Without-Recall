"""Reproduce Figure 1 of the paper with the paper-wide grid resolution."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm import PAPER_GRID_SIZE, _S_c0_compute


def tau(k):
    lambda_k = math.sqrt((1.0 - 1.0/k**2)/(k - 1.0))
    return (1.0 - 1.0/k)/lambda_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("figure_1.png"))
    args = parser.parse_args()

    s20 = _S_c0_compute(0, 0, -0.5, 0.0, 20, 1.0, 1.0, PAPER_GRID_SIZE)[0]
    s50 = _S_c0_compute(0, 0, -0.5, 0.0, 50, 1.0, 1.0, PAPER_GRID_SIZE)[0]

    k_tau = np.arange(3, 30)
    k20 = np.arange(3, 20)
    k50 = np.arange(3, 50)

    fig, ax = plt.subplots(figsize=(10.25, 6.0))
    ax.plot(k_tau, [tau(k) for k in k_tau], color="#c40000", marker="D",
            markersize=5.5, linewidth=2.7)
    ax.plot(k20, s20[k20, 1], color="#2094cf", marker="s", markersize=5.5,
            linewidth=2.7)
    ax.plot(k50, s50[k50, 1], color="#102f3f", marker="s", markersize=5.5,
            linewidth=2.7)

    ax.axvline(5, color="#58afe0", linestyle=(0, (4, 5)), linewidth=2)
    ax.axvline(7, color="#486678", linestyle=(0, (4, 5)), linewidth=2)
    ax.text(4.2, 0.2, r"$k^*_{20}=5$", color="#2094cf", fontsize=18,
            rotation=90, ha="right", va="bottom")
    ax.text(7.6, 0.2, r"$k^*_{50}=7$", color="#102f3f", fontsize=18,
            rotation=90, ha="left", va="bottom")
    ax.text(13.1, 4.25, r"$\tau_k^*$", color="#c40000", fontsize=30)
    ax.text(17.6, 0.72, r"$S_{20,k}$", color="black", fontsize=28)
    ax.text(37.0, 1.58, r"$S_{50,k}$", color="black", fontsize=28)
    ax.text(1.05, -0.02, r"$k$", transform=ax.transAxes, fontsize=28)

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.grid(color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b7b7b7")
        ax.spines[side].set_linewidth(1.6)
    ax.tick_params(axis="both", colors="#555555", labelsize=20, length=0, pad=8)
    fig.tight_layout(pad=0.8)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
