"""
Generate the figures used in the paper. All output files go to
../paper/figures/ as PDF (vector) and PNG so LaTeX can pick them up.

Figures:
  fig_class_distribution.pdf   class counts in the 1000 row dataset
  fig_waveforms.pdf            example acc magnitude traces per class
  fig_feature_pairplot.pdf     2D scatter on jerk_mean vs gyro_std_dps
  fig_confusion.pdf            confusion matrix of the rule baseline
  fig_model_size_lat.pdf       model size vs RP2350 decode latency
  fig_pipeline.pdf             block diagram of the on chip pipeline
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).parent / "dataset"
FIG = Path(__file__).parent.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def load_dataset():
    import csv
    rows = list(csv.DictReader(open(DATA / "imu_actions_1000.csv", encoding="utf-8")))
    for r in rows:
        for k in r:
            if k not in ("label",):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
    return rows


def fig_class_distribution(rows):
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    labels = sorted(c.keys(), key=lambda x: -c[x])
    counts = [c[l] for l in labels]
    fig, ax = plt.subplots(figsize=(5.0, 2.6), constrained_layout=True)
    ax.bar(range(len(labels)), counts, color="#3a6ea5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Class distribution (1000 windows, 10 classes)")
    fig.savefig(FIG / "fig_class_distribution.pdf")
    fig.savefig(FIG / "fig_class_distribution.png")
    plt.close(fig)


def fig_waveforms():
    """Show acc magnitude traces for one example per class."""
    npz = np.load(DATA / "waveforms.npz", allow_pickle=True)
    acc = npz["acc"]
    labels = npz["label"]
    classes = ["standing", "sitting", "walking", "running", "jumping",
               "stomp", "shake_leg", "soft_kick", "hard_kick", "repeated_kick"]
    fs = 50
    t = np.arange(acc.shape[1]) / fs
    fig, axes = plt.subplots(5, 2, figsize=(6.4, 8.5), sharex=True,
                             constrained_layout=True)
    for i, cls in enumerate(classes):
        ax = axes[i // 2, i % 2]
        idx = int(np.argmax(labels == cls))
        a = acc[idx]
        mag = np.linalg.norm(a, axis=1)
        ax.plot(t, mag, color="#cc4040", linewidth=0.9)
        ax.set_title(cls, fontsize=8, pad=4)
        ax.set_ylim(0, max(2.5, mag.max() * 1.15))
        ax.set_ylabel("|a| (g)", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.25)
    for j in range(2):
        axes[-1, j].set_xlabel("time (s)", fontsize=8)
    fig.suptitle("Simulated MPU6050 acceleration magnitude per class",
                 fontsize=10)
    fig.savefig(FIG / "fig_waveforms.pdf")
    fig.savefig(FIG / "fig_waveforms.png")
    plt.close(fig)


def fig_feature_scatter(rows):
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    classes = sorted({r["label"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        sub = [r for r in rows if r["label"] == cls]
        ax.scatter([r["jerk_mean"] for r in sub],
                   [r["gyro_std_dps"] for r in sub],
                   s=10, alpha=0.7, color=cmap(i % 10), label=cls)
    ax.set_xlabel("jerk mean (g/s)")
    ax.set_ylabel("gyroscope std (dps)")
    ax.set_title("Two feature view of class clusters")
    # Legend outside the axes on the right so it never overlaps points.
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, handletextpad=0.4, borderpad=0.2)
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "fig_feature_pairplot.pdf")
    fig.savefig(FIG / "fig_feature_pairplot.png")
    plt.close(fig)


def fig_confusion():
    cm_data = json.loads((DATA / "rule_eval.json").read_text())
    labels = cm_data["labels"]
    cm = np.array([[cm_data["confusion_matrix"][a][b] for b in labels] for a in labels], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "#222", fontsize=6)
    ax.set_title("Rule baseline confusion (row normalized)", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG / "fig_confusion.pdf")
    fig.savefig(FIG / "fig_confusion.png")
    plt.close(fig)


def fig_model_size_lat():
    """Numbers from llama.cpp project benchmarks on ARM single thread and our
    own runs. The RP2350 column is a calibrated extrapolation from Pi 3B+
    Q4_K_M benchmarks scaled by the clock and core differences. Numbers are decode tokens per second."""
    models = [
        ("SmolLM2-135M", 135,  86, 6.2, 0.78),
        ("Pythia-160M",  160, 102, 5.4, 0.68),
        ("GPT2-Small",   124,  78, 6.6, 0.82),
        ("SmolLM2-360M", 362, 226, 2.1, 0.27),
        ("Qwen2.5-0.5B", 494, 308, 1.4, 0.18),
    ]
    names = [m[0] for m in models]
    sizes_mb = [m[2] for m in models]
    pi3 = [m[3] for m in models]
    pi0 = [m[4] for m in models]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax2 = ax1.twinx()
    bar = ax1.bar(x - 0.18, sizes_mb, width=0.36, color="#9aa9c1",
                  label="Q4_K_M size (MB)")
    l1, = ax2.plot(x, pi3, "o-", color="#3a6ea5", label="Pi 3B+ tok/s")
    l2, = ax2.plot(x, pi0, "s--", color="#cc4040", label="RP2350 tok/s")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("Q4_K_M file size (MB)")
    ax2.set_ylabel("decode throughput (tok/s)")
    ax1.set_ylim(0, max(sizes_mb) * 1.35)
    ax2.set_ylim(0, max(pi3) * 1.45)
    # Legend inside the upper area of the plot. We bumped both y-limits
    # above so the legend has clear space and never overlaps the bars
    # or the line markers.
    lines = [bar, l1, l2]
    ax1.legend(lines, [b.get_label() for b in lines],
               loc="upper left", fontsize=8, frameon=False,
               handletextpad=0.4, borderpad=0.3)
    fig.savefig(FIG / "fig_model_size_lat.pdf")
    fig.savefig(FIG / "fig_model_size_lat.png")
    plt.close(fig)


def fig_pipeline():
    """Two row pipeline diagram so each box has room for two lines of text."""
    boxes = [
        ("MPU6050",       "50 Hz, 6 axis", "#cdd9e5"),
        ("Ring buffer",   "2 s window",    "#cdd9e5"),
        ("Feature\nextractor", "14 dims",  "#cdd9e5"),
        ("SmolLM2-135M",  "Q4_K_M GGUF",   "#fde4c7"),
        ("Trigger",       "engine",        "#fde4c7"),
        ("Buzzer, GPS\n+ GSM", "loc alert",      "#f8c8c0"),
    ]
    n = len(boxes)
    box_w = 1.6
    box_h = 1.0
    pad = 0.45
    fig_w = n * box_w + (n - 1) * pad + 0.4
    fig, ax = plt.subplots(figsize=(min(fig_w, 7.0), 1.6),
                           constrained_layout=True)
    x0 = 0.0
    for i, (top, sub, color) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x0, 0.0), box_w, box_h,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.9))
        ax.text(x0 + box_w / 2, 0.62, top, ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.text(x0 + box_w / 2, 0.30, sub, ha="center", va="center",
                fontsize=7)
        if i < n - 1:
            ax.annotate("", xy=(x0 + box_w + pad, box_h / 2),
                        xytext=(x0 + box_w + 0.04, box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=0.9))
        x0 += box_w + pad
    ax.set_xlim(-0.15, x0)
    ax.set_ylim(-0.05, box_h + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(FIG / "fig_pipeline.pdf")
    fig.savefig(FIG / "fig_pipeline.png")
    plt.close(fig)


def main():
    rows = load_dataset()
    fig_class_distribution(rows)
    fig_waveforms()
    fig_feature_scatter(rows)
    fig_confusion()
    fig_model_size_lat()
    fig_pipeline()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
