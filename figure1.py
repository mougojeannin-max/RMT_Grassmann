"""Recompute the RICE HT18 spectrum in Figure 1 from the grain SCMs."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def spectrum(data):
    """This descriptive figure pools all 961 grains, with their pixel counts."""
    folder = Path(data) / "rice"
    with np.load(folder / "subject00.npz", allow_pickle=False) as archive:
        counts = np.asarray(archive["dof"], dtype=np.int64) + 1
        labels = archive["labels"]
        denominator = str(archive["source_denominator"])
        covariance = archive["covariances"] if "covariances" in archive else None
    if covariance is None:
        covariance = np.load(folder / "subject00_covariances.npy", mmap_mode="r")
    if (covariance.shape != (961, 256, 256) or counts.shape != (961,)
            or labels.shape != (961,) or np.any(counts <= 1)
            or counts[769] != 396 or labels[769] != 3):
        raise ValueError("Figure 1 requires the released RICE object order (HT18-01_000 at index 769).")
    if denominator not in ("n", "n-1"):
        raise ValueError("Unknown covariance denominator")
    pooled = np.zeros((256, 256), dtype=np.float64)
    for i, count in enumerate(counts):
        scm = np.asarray(covariance[i], dtype=np.float64)
        # Figure 1 uses X.T @ X / n, as in the original descriptive plot.
        if denominator == "n-1":
            scm = scm * ((count - 1) / count)
        pooled += count * scm
    pooled /= counts.sum()
    values, vectors = np.linalg.eigh((pooled + pooled.T) / 2)
    if values[0] <= 0:
        raise ValueError("The pooled covariance is not positive definite")
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    target = np.asarray(covariance[769], dtype=np.float64)
    if denominator == "n-1":
        target = target * (395 / 396)
    whitened = whitening @ target @ whitening.T
    return np.maximum(np.linalg.eigvalsh((whitened + whitened.T) / 2), 0), 256 / 396


def plot(values, ratio, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    low, high = (1 - np.sqrt(ratio)) ** 2, (1 + np.sqrt(ratio)) ** 2
    x = np.linspace(low, high, 600)
    density = np.sqrt(np.maximum((high - x) * (x - low), 0)) / (2 * np.pi * ratio * x)
    heights, edges = np.histogram(values, bins=35, density=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
                         "axes.labelsize": 9, "axes.linewidth": .6,
                         "pdf.fonttype": 42, "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(3.25, 1.95))
    fig.subplots_adjust(left=.16, right=.985, bottom=.24, top=.96)
    ax.axvspan(low, high, color="crimson", alpha=.08, linewidth=0)
    ax.bar(edges[:-1], heights, width=np.diff(edges), align="edge", color="darkorange",
           alpha=.7, edgecolor="white", linewidth=.4, label="Empirical spectrum")
    ax.plot(x, density, "--", color="crimson", lw=1.2, label="MP density")
    ax.axvline(high, color=".25", ls=":", lw=.8, label=r"MP upper edge $x_+(c)$")
    ax.set(xlim=(0, 8), ylim=(0, 1.2), xlabel="Eigenvalue", ylabel="Density")
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, .4, .8, 1.2])
    ax.tick_params(length=2.5, width=.6, pad=2)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, borderaxespad=.3, labelspacing=.35)
    outliers = np.flatnonzero((edges[:-1] > high) & (heights > 0))
    ax.text(6.2, .27, "Outliers", ha="center", va="bottom")
    for index, start in zip(outliers, [(5.8, .26), (6.6, .26)]):
        ax.annotate("", xy=((edges[index] + edges[index + 1]) / 2, heights[index]),
                    xytext=start, arrowprops={"arrowstyle": "->", "color": ".3", "lw": .6})
    fig.savefig(output / "figure1.pdf", metadata={"Author": "", "Creator": "",
                "CreationDate": None, "ModDate": None, "Title": "Whitened RICE HT18 spectrum"})
    fig.savefig(output / "figure1.png", dpi=250)
    plt.close(fig)
    np.savetxt(output / "figure1_eigenvalues.csv", values, delimiter=",", header="eigenvalue", comments="")
    np.savetxt(output / "figure1_histogram.csv", np.c_[edges[:-1], np.diff(edges), heights],
               delimiter=",", header="left,width,density", comments="")
    report = dict(ratio=ratio, upper_edge=high, n_pixels=396, grain_index=769,
                  eigenvalues_above_edge=int(np.sum(values > high)),
                  largest_eigenvalues=values[-3:].tolist(),
                  histogram_integral=float(np.sum(np.diff(edges) * heights)))
    (output / "figure1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/figure1"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(plot(*spectrum(args.data), args.output), indent=2))


if __name__ == "__main__":
    main()
